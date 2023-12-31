import copy
from configs.common import common_cfg
from modules.augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)
if cfg.WANDB_API_KEY=='your key':
    print('input your wandb api key!')
    raise NotImplementedError

cfg.model_type = "sed"
cfg.model_name = "tf_efficientnetv2_s_in21k"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 16 # 96, 16 (my train), 8 (torchview)
cfg.PRECISION = 32
cfg.seed = {
    "pretrain_ce": 20231121,
    "pretrain_bce": 20230503,
    "train_ce": 20231019,
    "train_bce": 20231911,
    "finetune": 20230523,
}
cfg.DURATION_TRAIN = 10
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = True
cfg.mixup2 = True
cfg.mixup_prob = 0.7
cfg.mixup_double = 0.5
cfg.mixup2_prob = 0.15
cfg.mix_beta = 5
cfg.mix_beta2 = 2
cfg.in_chans = 3
cfg.epochs = {
    "pretrain_ce": 70,# 70
    "pretrain_bce": 40,
    "train_ce": 60,
    "train_bce": 30,
    "finetune": 10,
}
cfg.lr = {
    "pretrain_ce": 3e-4,
    "pretrain_bce": 1e-3,
    "train_ce": 3e-4,
    "train_bce": 1e-3,
    "finetune": 6e-4,
}

cfg.model_ckpt = {
    "pretrain_ce": None,
    "pretrain_bce": "outputs/sed_v2s/pytorch/pretrain_ce/last.ckpt",
    "train_ce": "outputs/sed_v2s/pytorch/pretrain_bce/last.ckpt",
    "train_bce": "outputs/sed_v2s/pytorch/train_ce/last.ckpt",
    "finetune": "outputs/sed_v2s/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    "pretrain_ce": "outputs/sed_v2s/pytorch/pretrain_ce",
    "pretrain_bce": "outputs/sed_v2s/pytorch/pretrain_bce",
    "train_ce": "outputs/sed_v2s/pytorch/train_ce",
    "train_bce": "outputs/sed_v2s/pytorch/train_bce",
    "finetune": "outputs/sed_v2s/pytorch/finetune",
}

cfg.final_model_path = "outputs/sed_v2s/pytorch/finetune/last.ckpt"
cfg.onnx_path = "outputs/sed_v2s/onnx"
cfg.openvino_path = "outputs/sed_v2s/openvino"

cfg.loss = {
    "pretrain_ce": "ce",
    "pretrain_bce": "bce",
    "train_ce": "ce",
    "train_bce": "bce",
    "finetune": "bce",
}

cfg.img_size = 384
cfg.n_mels = 128 # number of mel filterbanks.
cfg.n_fft = 2048 # size of FFT (fast fourier transform), creates n_fft // 2 + 1 frequency bins.
cfg.f_min = 0 # minimum frequency in FFT.
cfg.f_max = 16000 # maximum frequency in FFT.

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)

# hop_length refers to the number of samples between the starting points of consecutive Short-Time Fourier Transform (STFT) windows.
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1) # length of hop between STFT windows. 

cfg.normal = 80 # normalization.

cfg.tta_delta = 3

# https://iver56.github.io/audiomentations/waveform_transforms/add_background_noise/
cfg.am_audio_transforms = amCompose(
    [
        # sed        
        AddBackgroundNoise(
            # list containing path to .wav files
            cfg.birdclef2021_nocall + cfg.birdclef2020_nocall, 
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.6,
        ),
        AddBackgroundNoise(
            cfg.freefield + cfg.warblrb + cfg.birdvox,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.3,
        ),
        AddBackgroundNoise(
            cfg.rainforest + cfg.environment, min_snr_in_db=0, max_snr_in_db=3, p=0.4
        ),
        amOneOf(
            [
                # multiply the audio by a random amplitude factor to reduce or increase the volume.
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                # gradually change the volume up or down over a random time span. Also known as fade in and fade out. 
                GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            ],
        ),
    ]
)

# manually written augmentations (no third party library).
cfg.np_audio_transforms = CustomCompose(
    [
        CustomOneOf(
            [
                NoiseInjection(p=1, max_noise_level=0.04),
                GaussianNoise(p=1, min_snr=5, max_snr=20),
                PinkNoise(p=1, min_snr=5, max_snr=20),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ],
            p=0.3,
        ),
    ]
)

cfg.input_shape = (120,cfg.in_chans,cfg.n_mels,768)
cfg.input_names = [ "x",'tta_delta' ]
cfg.output_names = [ "y" ]
cfg.opset_version = None

basic_cfg = cfg
