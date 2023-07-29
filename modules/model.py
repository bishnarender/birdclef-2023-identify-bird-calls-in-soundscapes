import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from torch.distributions import Beta
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import sklearn
import timm
import torchaudio
import os
from torch_audiomentations import Compose, PitchShift, Shift, OneOf, AddColoredNoise
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
)
import sklearn.metrics
from torch.cuda.amp import autocast




def padded_cmap(solution, submission, padding_factor=5):
    '''
    padded_cmap => padded "cmap" (classification mean average precision).
    '''
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')    
    new_rows = []
    
    
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
        
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
        
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    
    # padded_solution.shape => (9, 834)
    # padded_submission.shape => (9, 834)
    
    score = sklearn.metrics.average_precision_score(
        # .astype(int) resolved the error "continuous-multioutput format is not supported".
        padded_solution.values.astype(int),
        padded_submission.values,
        average="macro",# calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    )
    return score


def map_score(solution, submission):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average="micro",
    )
    return score


def compute_deltas(specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate") -> torch.Tensor:
    r"""
    Delta coefficients are used to represent the rate of change of each element in the input tensor over time. This is useful for capturing the dynamic changes in audio signals, especially in speech and audio processing tasks.    
    
    Compute delta coefficients of a tensor, usually a spectrogram:
       
    d(t) = summation_from_n_to_N[ n{ c(t+n) - c(t-n) } ] /  2*summation_from_n_to_N[ n^2 ]

    where : d(t) is the deltas at time t i.e., delta coefficient from frame t,
    `c_t` is the spectrogram static coeffcients at time t,
    `N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3
    
    # represents the number of elements to the left and right of the central element in the window.
    n = (win_length - 1) // 2
    
    # denominator "denom" is computed as a constant value, which is used to normalize the deltas.
    # twice sum of integer squared i.e., denominator of mathematical expression present in docstring.
    denom = n * (n + 1) * (2 * n + 1) / 3

    # .pad(...) => pad some dimensions of input are described starting from the last dimension and moving forward.
    # pads the input tensor by "replicating" (default mode) the tensor values.
    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)
    
    # .arange(start=0, end, step=1, ...) => returns a 1-D tensor.
    # a kernel is created with values ranging from -n to n, repeated for each frequency bin in the input tensor. 
    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )
    
    # the groups parameter is set to specgram.shape[1] to ensure that each frequency bin is convolved independently.
    output = (
        torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


def make_delta(input_tensor: torch.Tensor):
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


def image_delta(x):
    delta_1 = make_delta(x)
    # compute second-order delta coefficients. 
    delta_2 = make_delta(delta_1)
    x = torch.cat([x, delta_1, delta_2], dim=1)
    return x


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator."""
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32))


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    
    # .interpolate(input, size=None, scale_factor=None, ...) => down/up samples the input to either the given size or the given scale_factor.
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


def gem(x: torch.Tensor, p=3, eps=1e-6):
    # .clamp(min=eps) => values less than eps are set to eps.
    # avg_pool2d(input, kernel_size,...)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    '''
    Generalized Mean Pooling. it generalizes the concept of average pooling i.e., generalizaton of average pooling by the parameter p. 
    '''
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        # torch.ones(1), torch.ones(1)*3 => tensor([1.]), tensor([3.])
        # .Parameter(...) => a kind of Tensor that is to be considered a module parameter i.e., power parameter p becomes trainable and can be learned during the model training process. .
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        '''
        The primary purpose of __repr__ is to provide a clear, unambiguous, and informative representation of the object, so that developers can understand its content and state. if you create an instance like this: gem_instance = GeM(p=2.5, eps=1e-5), then calling print(gem_instance) will produce the output: GeM(p=2.5000, eps=1e-05).
        '''
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        # initialization.
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixup_prob, mixup_double):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup_prob = mixup_prob
        self.mixup_double = mixup_double

    def forward(self, X, Y, weight=None):
        # .rand(size, ...) => returns a tensor filled with random numbers from a uniform distribution on the interval [0,1).
        p = torch.rand((1,))[0]
        if p < self.mixup_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            
            # torch.randperm(n, *, ...) => returns a random permutation of integers from 0 to n-1.
            # torch.randperm(6) => tensor([0, 4, 1, 3, 5, 2])
            # torch.randperm(6).shape => torch.Size([6])
            # random permutation "perm" of indices for shuffling the input batch. 
            perm = torch.randperm(bs)

            p1 = torch.rand((1,))[0]
            if p1 < self.mixup_double:
                X = X + X[perm]
                Y = Y + Y[perm]
                
                # torch.clamp(input, min=None, max=None, ...) => clamps all elements in input into the range [ min, max ].
                # since the labels are assumed to be in the range [0, 1].
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = 0.5 * weight + 0.5 * weight[perm]
                    return X, Y, weight
            else:
                perm2 = torch.randperm(bs)
                X = X + X[perm] + X[perm2]
                Y = Y + Y[perm] + Y[perm2]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = (
                        1 / 3 * weight + 1 / 3 * weight[perm] + 1 / 3 * weight[perm2]
                    )
                    return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


class Mixup2(nn.Module):
    def __init__(self, mix_beta, mixup2_prob):
        super(Mixup2, self).__init__()
        # Beta(concentration1 (alpha), concentration0 (beta), ...) => Beta distribution parameterized by concentration1 and concentration0. Beta distribution is a family of continuous probability distributions defined on the interval [0, 1] or (0, 1) in terms of two positive parameters, denoted by alpha (α) and beta (β).
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup2_prob = mixup2_prob

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup2_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            
            # Beta(torch.tensor([0.5]), torch.tensor([0.5])).rsample((4,)) => tensor([[0.9311], [0.0017], [0.4144], [0.9786]])
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
            
            # coefficients determine the mixing ratio between the original and shuffled samples.
            if n_dims == 2:                
                X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
            elif n_dims == 3:
                X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
            else:
                X = (
                    coeffs.view(-1, 1, 1, 1) * X
                    + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
                )
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
            # Y = Y + Y[perm]
            # Y = torch.clamp(Y, 0, 1)

            if weight is None:
                return X, Y
            else:
                weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
                return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


class BirdClefModelBase(pl.LightningModule):
    def __init__(self, cfg, stage):
        super().__init__()
        self.num_classes = len(cfg.bird_cols)
        self.birds = [bird for bird in cfg.bird_cols]
        self.stage = stage
        self.cfg = cfg
        self.loss = cfg.loss[stage]
        self.lr = cfg.lr[stage]
        self.epochs = cfg.epochs[stage]
        self.in_chans = cfg.in_chans # 3

        if self.loss == "ce":
            self.loss_function = nn.CrossEntropyLoss(
                label_smoothing=self.cfg.label_smoothing, reduction="none"
            )
        elif self.loss == "bce":
            self.loss_function = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise NotImplementedError
            
        self.mixup = Mixup(
            mix_beta=self.cfg.mix_beta, # 5
            mixup_prob=self.cfg.mixup_prob, # 0.7
            mixup_double=self.cfg.mixup_double, # 0.5
        )
        self.mixup2 = Mixup2(
            mix_beta=self.cfg.mix_beta2, mixup2_prob=self.cfg.mixup2_prob
        )
        self.ema = None

        # https://github.com/asteroid-team/torch-audiomentations
        self.audio_transforms = Compose(
            [
                # AddColoredNoise(p=0.5),
                # pitch shift the sound up or down "without changing the tempo" (while keeping the original speed).
                PitchShift(
                    min_transpose_semitones=-4, # unit: semitones.
                    max_transpose_semitones=4,
                    sample_rate=self.cfg.SR,
                    p=0.4,
                ),
                # shift the samples forwards or backwards, with or without rollover. Rollover refers to the behavior when the shifted samples exceed the original audio duration. When set to rollover=True, samples that roll beyond the first or last position are re-introduced at the last or first. 
                Shift(min_shift=-0.5, max_shift=0.5, p=0.4),
            ]
        )
        
        # torchaudio.transforms.TimeMasking(...) => apply masking to a spectrogram in the time domain.
        # time_mask_param => maximum possible length of the mask.
        # iid_masks => whether to apply different masks to each example/channel in the batch.
        # p => maximum proportion of time steps that can be masked.
        self.time_mask_transform = torchaudio.transforms.TimeMasking(
            time_mask_param=60, iid_masks=True, p=0.5
        )
        
        # torchaudio.transforms.FrequencyMasking(...) => apply masking to a spectrogram in the frequency domain.
        # freq_mask_param => maximum possible length of the mask.
        self.freq_mask_transform = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=24, iid_masks=True
        )
        
        # self.cfg.hop_length => 417
        #window = torch.hann_window(window_length = 2048, device = self.cfg.DEVICE)
        # .MelSpectrogram() => create MelSpectrogram for a raw audio signal.
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.SR,
            # hop_length refers to the number of samples between the starting points of consecutive Short-Time Fourier Transform (STFT) windows.
            hop_length=self.cfg.hop_length, # length of hop between STFT windows.
            n_mels=self.cfg.n_mels,# number of mel filterbanks.
            f_min=self.cfg.f_min,# minimum frequency. 
            f_max=self.cfg.f_max,# maximum frequency.
            n_fft=self.cfg.n_fft, # size of FFT, creates n_fft // 2 + 1 frequency bins.
            center=True, # whether to pad waveform on both sides so that the t-th frame is centered at time t×hop_length. 
            pad_mode="constant",
            
            # When computing the Mel filterbank weights, the filterbanks are typically represented as triangular filters. Each triangular filter corresponds to a particular Mel frequency and spans a specific range of frequencies. The width of the triangular filter corresponds to the width of the Mel band centered at that Mel frequency. Area normalization means that the area under each triangular filter in the Mel domain is normalized to a value of 1.
            norm="slaney", # if “slaney”, divide the triangular mel weights by the width of the mel band (area normalization).
            onesided=True,
            mel_scale="slaney",
            #win_length="" # window size (Default: n_fft). specifies the number of samples in each window/frame. 
        )

        if self.cfg.DEVICE.type == "cuda":
            self.melspec_transform = self.melspec_transform.cuda()
        else:
            self.melspec_transform = self.melspec_transform.cpu()

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        
    def lower_upper_freq(self, images):
        #images = images - images.min(1,keepdim=True).values.min(2,keepdim=True).values.min(3,keepdim=True).values
        
        # torch.randint(low=0, high, size, ...) => returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        r = torch.randint(self.cfg.n_mels // 2, self.cfg.n_mels, size=(1,))[0].item()
        # torch.rand(*size, *, ...) => Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1).
        x = (torch.rand(size=(1,))[0] / 2).item()
        
        # pink_noise tensor represents a "pink noise" pattern with decreasing values in the first half and constant values in the second half. random parameters (r and x) ensures that the noise is different for each batch.
        pink_noise = (
            torch.from_numpy(
                np.array(
                    [
                        np.concatenate(
                            (
                                1 - np.arange(r) * x / r,
                                np.zeros(self.cfg.n_mels - r) - x + 1,
                            )
                        )
                    ]
                ).T
            )
            .float()
            .to(self.cfg.DEVICE)
        )
        
        # attenuating the frequencies in the first half of the images. this can help the model learn to be more robust to missing or attenuated frequency information, which can be useful for improving the model's performance on noisy or incomplete audio data. 
        images = images * pink_noise
        
        # images_max = images.max(1,keepdim=True).values.max(2,keepdim=True).values.max(3,keepdim=True).values
        # images = images/images_max
        return images

    def transform_to_spec(self, audio):
        '''
        transformation section.
        '''
        if self.training:
            audio = self.audio_transforms(audio, sample_rate=self.cfg.SR)

        # normalize
        #max_amplitude = torch.abs(audio).max()
        #audio = audio/max_amplitude

        spec = self.melspec_transform(audio)
        
        # transformation to convert the power spectrogram into decibels (dB) scale. 
        spec = self.db_transform(spec)
        
        # normalization of the spectrogram. 
        if self.cfg.normal == 80:
            spec = (spec + 80) / 80
        elif self.cfg.normal == 255:
            spec = spec / 255
        else:
            raise NotImplementedError
            
        #spec = qtransform(audio)
        #spec = spec.unsqueeze(1)
        if self.training:
            # time masking masks random consecutive time steps in the spectrogram.
            spec = self.time_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                # frequency masking masks random consecutive frequency bins in the spectrogram, providing robustness to variations in the frequency domain.
                spec = self.freq_mask_transform(spec)
            if torch.rand(size=(1,))[0] < 0.5:
                # pink noise.
                spec = self.lower_upper_freq(spec)
        return spec

    def set_ema(self, ema):
        self.ema = ema

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema is not None:
            if (self.global_step + 1) % 10 == 0:
                self.ema.update(self)

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.cfg.weight_decay,
        )
        interval = "epoch"

        lr_scheduler = CosineAnnealingWarmRestarts(
            model_optimizer, T_0=self.epochs, T_mult=1, eta_min=1e-6, last_epoch=-1
        )

        return {
            "optimizer": model_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": interval,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def freeze(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.freeze()
        
        # tricky forward pass through the model to obtain predictions. 
        y_pred, target, loss = self(batch)                

        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.ema is not None:
            y_pred, target, val_loss = self.ema.module(batch)
        else:
            y_pred, target, val_loss = self(batch)
            
        # print(y_pred)
        # self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return {"val_loss": val_loss, "logits": y_pred, "targets": target}

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def validation_epoch_end(self, validation_step_outputs): # def on_validation_epoch_end(self, *arg, **kwargs):
        if len(validation_step_outputs): # len(outputs)
            # calculate the average validation loss across all batches in the epoch.
            avg_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean() # x in outputs
            
            # concatenate and process the model's predictions for the entire epoch.
            output_val = (
                torch.cat([x["logits"] for x in validation_step_outputs], dim=0) # for x in outputs
                .sigmoid().cpu().detach().numpy()
            )
            
            # concatenate and process the target labels for the entire epoch.
            target_val = (
                torch.cat([x["targets"] for x in validation_step_outputs], dim=0).cpu().detach().numpy() # for x in outputs
            )            

            # print(output_val.shape)
            val_df = pd.DataFrame(target_val, columns=self.birds)
            pred_df = pd.DataFrame(output_val, columns=self.birds)
            
            # val_df.shape => [4,834]
            # pred_df.shape => [4,834]
        
            if self.current_epoch > -1:
                avg_score = padded_cmap(val_df, pred_df, padding_factor=5)
                avg_score2 = padded_cmap(val_df, pred_df, padding_factor=3)
                
                # LRAP => assess how well the model ranks the ground-truth labels for each instance in the predicted list. for each ground truth label, what fraction of higher-ranked labels were true labels.
                avg_score3 = sklearn.metrics.label_ranking_average_precision_score(
                    target_val.astype(int), output_val
                )
                self.log("val_loss", avg_loss,  on_step=False, on_epoch=True, logger=True, prog_bar=True,)
                self.log("validation C-MAP score pad 5", avg_score, on_step=False, on_epoch=True, logger=True, prog_bar=True,)
                self.log("validation C-MAP score pad e", avg_score2, on_step=False, on_epoch=True, logger=True,
                    prog_bar=True,
                )
                self.log("validation AP score", avg_score3, on_step=False, on_epoch=True, logger=True, prog_bar=True,)
                # competition_metrics(output_val,target_val)
                print(f"epoch {self.current_epoch} validation loss {avg_loss}")
                print(f"epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}")
                print(f"epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}")
                print(f"epoch {self.current_epoch} validation AP score {avg_score3}")
            else:
                self.log("val_loss", avg_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True,)
                avg_score = 0
                print(f"epoch {self.current_epoch} validation loss {avg_loss}")
            val_df.to_pickle("val_df.pkl")
            pred_df.to_pickle("pred_df.pkl")
        else:
            avg_loss = 0
            avg_score = 0
        return {"val_loss": avg_loss, "val_cmap": avg_score}    
    

    def training_epoch_end(self, training_step_outputs): # def on_train_epoch_end(self, *arg, **kwargs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean() # x in outputs
        self.log("train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        print(f"epoch {self.current_epoch} train loss {avg_loss}")
        
        if (self.ema is not None) & ((self.current_epoch > self.epochs - 3 - 1)):
            if not os.path.exists(self.cfg.output_path[stage]):
                os.makedirs(self.cfg.output_path[stage])
            torch.save(
                {
                    "state_dict": self.ema.module.state_dict(),
                },
                os.path.join(self.cfg.output_path[stage], f"ema_{self.current_epoch}.ckpt"),
            )


class BirdClefTrainModelSED(BirdClefModelBase):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        self.bn0 = nn.BatchNorm2d(cfg.n_mels)

        base_model = timm.create_model(
            cfg.model_name,
            pretrained=True,
            in_chans=self.in_chans,
            drop_path_rate=0.2,
            drop_rate=0.5,
        )
        
        #base_model.conv_stem.stride = (1,1)
        layers = list(base_model.children())[:-2]
        # base_model.children() => <generator object Module.children at 0x7fe004fa1e50>        
        # list(base_model.children()) => [..., Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False), BatchNormAct2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True (drop): Identity() (act): SiLU(inplace=True))], SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1)), Linear(in_features=1280, out_features=21843, bias=True)]
        # list(base_model.children())[:-2] => [..., Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False), BatchNormAct2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True (drop): Identity() (act): SiLU(inplace=True))]                        
    
        self.encoder = nn.Sequential(*layers)

        if "efficientnet" in self.cfg.model_name:
            in_features = base_model.classifier.in_features
        elif "eca" in self.cfg.model_name:
            in_features = base_model.head.fc.in_features
        elif "res" in self.cfg.model_name:
            in_features = base_model.fc.in_features
            
        # bias => the layer will learn an additive bias. the values are initialized from uniform distribution.
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def freeze(self):
        if self.stage == "finetune":
            self.encoder.eval()
            self.fc1.eval()
            self.bn0.eval()
            for param in self.encoder.parameters():
                # freezing the part of the model as no changes happen to its parameters.
                param.requires_grad = False
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.bn0.parameters():
                param.requires_grad = False
        return

    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2))
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # if self.training:
        #    x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        
        return x, frames_num

    def forward(self, batch):
        x = batch[0]
        y = batch[1]
        weight = batch[2]
                
#         self.training=True # my_ for torchview
                
        if not self.training:
            # x.shape[0], x.shape[1], x.shape[2], x.shape[3] => 2, 1, 6, 320000
            # test_batch_size => 2
            bs, channel, parts = x.shape[0], x.shape[1], x.shape[2]
            x = x.reshape((bs * parts, channel, -1))
            # x.shape => (12,6,320000)

        if self.training:
            if self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
                
        #with autocast(enabled=False):
        x = self.transform_to_spec(x)
        if self.in_chans == 3:
            x = image_delta(x)

        if self.training:
            if self.cfg.mixup2:
                x, y, weight = self.mixup2(x, y, weight)

        x, frames_num = self.extract_feature(x)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output,
        }
        if not self.training:
            clipwise_output = clipwise_output.reshape((bs, parts, -1)).max(dim=1).values
            seg_num = segmentwise_logit.shape[1]
            fram_num = framewise_logit.shape[1]
            segmentwise_logit = (
                segmentwise_logit.reshape((bs, parts, seg_num, -1)).max(dim=1).values
            )
            framewise_logit = (
                framewise_logit.reshape((bs, parts, fram_num, -1)).max(dim=1).values
            )

        loss = 0.5 * self.loss_function(
            torch.logit(clipwise_output), y
        ) + 0.5 * self.loss_function(segmentwise_logit.max(1)[0], y)
        
        # loss = 0.5*self.loss_function(torch.logit(clipwise_output), y) + 0.5*self.loss_function(framewise_logit.max(1)[0], y)
        if self.loss == "ce":
            # loss.shape => [BS,]
            loss = (loss * weight) / weight.sum()
            # loss.shape => [BS,]
        elif self.loss == "bce":
            # loss.shape => [BS, 834]
            loss = loss.sum(dim=1) * weight
            # loss.shape => [BS,]
        else:
            raise NotImplementedError
            
        loss = loss.sum()
        # loss.shape => [] i.e, single value.
                
        return torch.logit(clipwise_output), y, loss

class BirdClefInferModelSED(BirdClefTrainModelSED):
    def forward(self,x,tta_delta=2):
        
        #x = x[0]
        #y = batch[1]
        #weight = batch[2]        
        
        x,_ = self.extract_feature(x)
        time_att = torch.tanh(self.att_block.att(x))
        feat_time = x.size(-1)
        start = (
            feat_time / 2 - feat_time * (self.cfg.infer_duration / self.cfg.DURATION) / 2
        )
        end = start + feat_time * (self.cfg.infer_duration / self.cfg.DURATION)
        start = int(start)
        end = int(end)
        pred = self.attention_infer(start,end,x,time_att)
        
        # tta => test time augmentation. tta involves making predictions on multiple augmented versions of the same input and then averaging or combining those predictions. 
        start_minus = start-tta_delta
        end_minus=end-tta_delta
        pred_minus = self.attention_infer(start_minus,end_minus,x,time_att)

        start_plus = start+tta_delta
        end_plus=end+tta_delta
        pred_plus = self.attention_infer(start_plus,end_plus,x,time_att)

        pred = 0.5*pred + 0.25*pred_minus + 0.25*pred_plus
        return pred

    def attention_infer(self,start,end,x,time_att):
        feat = x[:, :, start:end]
        #att = torch.softmax(time_att[:, :, start:end], dim=-1)
        #             print(feat_time, start, end)
        #             print(att_a.sum(), att.sum(), time_att.shape)
        framewise_pred = torch.sigmoid(self.att_block.cla(feat))
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        #clipwise_output = torch.sum(framewise_pred * att, dim=-1)
        #logits = torch.sum(
        #    self.att_block.cla(feat) * att,
        #    dim=-1,
        #)

        #return clipwise_output
        return framewise_pred_max

class BirdClefTrainModelCNN(BirdClefModelBase):
    def __init__(self, cfg, stage):
        super().__init__(cfg, stage)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=True,
            in_chans=self.in_chans,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,
            drop_rate=0.5,
        )

        if "efficientnet" in self.cfg.model_name:
            backbone_out = self.backbone.num_features
        elif "eca" in self.cfg.model_name:
            backbone_out = self.backbone.num_features
        elif "res" in self.cfg.model_name:
            backbone_out = self.backbone.num_features

        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.num_classes)

        self.big_dropout = nn.Dropout(p=0.5)

    def freeze(self):
        if self.stage == "finetune":
            self.backbone.eval()
            self.global_pool.eval()
            for param in self.global_pool.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, batch):
        x = batch[0].squeeze(1)
        y = batch[1]
        weight = batch[2]
        
#         self.training = True # my_ for torchview
        if self.training:
            self.factor = self.cfg.train_part
        else:
            self.factor = self.cfg.valid_part

        if not self.training:
            bs = x.shape[0]
            x = x.reshape((bs, -1))
        else:
            if self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
                
        bs, time = x.shape
        x = x.reshape(bs * self.factor, time // self.factor)
        
        #with autocast(enabled=False):
        x = self.transform_to_spec(x.unsqueeze(1))
        if self.in_chans == 3:
            x = image_delta(x)

        x = x.permute(0, 1, 3, 2)
        # x = x.permute(0, 2, 1)
        # x = x[:, None, :, :]

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if self.cfg.mixup2:
                x, y, weight = self.mixup2(x, y, weight)
            # if self.cfg.mixup:
            #    x, y, weight = self.mixup(x, y, weight)
            # if self.cfg.mixup2:
            #    x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)

        x = self.backbone(x)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b // self.factor, self.factor * t, c, f)
        x = x.permute(0, 2, 1, 3)
        x = self.global_pool(x)
        x = x[:, :, 0, 0] # __getitem__
        
        if self.training:
            logits = torch.mean(
                torch.stack([self.head(self.big_dropout(x)) for _ in range(5)], dim=0),
                dim=0,
            )
        else:
            logits = self.head(x)

        loss = self.loss_function(logits, y)
        
        if self.loss == "ce":
            loss = (loss * weight) / weight.sum()
        elif self.loss == "bce":
            loss = loss.sum(dim=1) * weight
        else:
            raise NotImplementedError
        loss = loss.sum()

        return logits, y, loss

class BirdClefInferModelCNN(BirdClefTrainModelCNN):
    def forward(self,x):
        x = x.permute(0, 1, 3, 2)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x)
        return logits

def load_model(cfg,stage,train=True):
    if train:
        model_ckpt = cfg.model_ckpt[stage]
    else:
        model_ckpt = cfg.final_model_path

    if model_ckpt is not None:
        state_dict = torch.load(model_ckpt,map_location=cfg.DEVICE)['state_dict']
        print('loading model from checkpoint')
    else:
        state_dict = None

    # drop ema because did not used in the competition
    if state_dict is not None:
        keys = list(state_dict.keys())
        for k in keys:
            if 'ema' in k:
                state_dict.pop(k)

    if cfg.model_type=='sed':
        if train:
            model = BirdClefTrainModelSED(cfg, stage)
            if state_dict is not None:
                # pretrain to train
                if stage == 'train_ce':
                    # pop(key[, default]) => if key is in the dictionary, remove it and return its value, else return default. if default is not given and key is not in the dictionary, a KeyError is raised.
                    state_dict.pop('att_block.att.weight')
                    state_dict.pop('att_block.att.bias')
                    state_dict.pop('att_block.cla.weight')
                    state_dict.pop('att_block.cla.bias')
                    model.load_state_dict(state_dict,strict=False)
                else:
                    model.load_state_dict(state_dict,strict=False)
        else:
            model = BirdClefInferModelSED(cfg, stage)
            model.load_state_dict(state_dict,strict=False)

    elif cfg.model_type=='cnn':
        if train:
            model = BirdClefTrainModelCNN(cfg, stage)
            if state_dict is not None:
                # pretrain to train
                if stage == 'train_ce':
                    state_dict.pop('head.weight')
                    state_dict.pop('head.bias')
                    model.load_state_dict(state_dict,strict=False)
                else:
                    model.load_state_dict(state_dict,strict=False)
        else:
            model = BirdClefInferModelCNN(cfg, stage)
            model.load_state_dict(state_dict,strict=False)
    else:
        raise NotImplementedError

    if not train:
        model.eval()
    return model


