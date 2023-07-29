import librosa as lb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from modules.utils import crop_or_pad
import multiprocessing

class BirdTrainDataset(Dataset):

    def __init__(self, df, df_labels, cfg, res_type="kaiser_fast",resample=True, train = True, pseudo=None, transforms=None):
        self.cfg =cfg
        self.df = df
        self.df_labels = df_labels
        self.sr = cfg.SR # 32000
        self.n_mels = cfg.n_mels # 128
        self.fmin = cfg.f_min # 0
        self.fmax = cfg.f_max # 16000

        self.train = train
        # duration of each audio clip .
        self.duration = cfg.DURATION
        
        # audio length in samples
        self.audio_length = self.duration*self.sr
        
        # resampling type of audio files (e.g., "kaiser_fast").        
        self.res_type = res_type
        # whether audio files should be resampled during loading.
        self.resample = resample
        
        # .clip(a, a_min, a_max, ..) => clip (limit) the values in an array. values outside the interval are clipped to the interval edges. 
        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        self.pseudo = pseudo

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def adjust_label(self,labels,filename,sample_ends,target,version,pseudo,pseudo_weights):
        '''
            adjust labels when pseudo=True.
        '''
        adjust_label = {label:0 for label in labels if label in self.cfg.bird_cols}
        labels_comp = list(adjust_label.keys())
        
        for oof,w in zip(pseudo,pseudo_weights):
            for label in labels_comp:
                preds = [oof['pred'][version][filename][label][sample_end] for sample_end in sample_ends]
                thre = oof['thre'][label]
                adjusts = np.zeros(shape=(len(preds),))
                for i,pred in enumerate(preds):
                    q3,q2,q1 = thre['q3'],thre['q2'],thre['q1']
                    if pred>=q3:
                        adjust = 1.0
                    elif pred>=q2:
                        adjust = 0.9
                    elif pred>=q1:
                        adjust = 0.5
                    else:
                        adjust = 0.2
                    adjusts[i] = adjust
                    
                adjust_label[label] += w * (1-np.prod(1-adjusts))
        for label in labels_comp:
            if adjust_label[label] <= 0.6:
                adjust_label[label] = 0.01
            elif adjust_label[label]<=0.75:
                adjust_label[label] = 0.6
            target[label] = target[label] * adjust_label[label]
        return target

    def load_data(self, filepath,target,row):
        filename = row['filename']
        labels = [bird for bird in list(set([row[self.cfg.primary_label_col]] + row[self.cfg.secondary_labels_col])) if bird in self.cfg.bird_cols]
        
        secondary_labels = [bird for bird in row[self.cfg.secondary_labels_col] if bird in self.cfg.bird_cols]
        
        duration = row['duration']
        version = row['version']
        presence = row['presence_type']
        
        
        # self mixup
        self_mixup_part = 1
        if (presence!='foreground') | (len(secondary_labels)>0):
            self_mixup_part = int(self.cfg.background_duration_thre/self.duration) # background_duration_thre=>60, DURATION=>10
            
        work_duration = self.duration * self_mixup_part
        work_audio_length = work_duration*self.sr
        
        # work_duration, work_audio_length => 10, 320000

        max_offset =np.max([0,duration-work_duration])
        # duration , max_offset => 24.8946875, 14.8946875
        
        # infer_duration => 5
        parts = int(duration//self.cfg.infer_duration) if duration%self.cfg.infer_duration==0 else int(duration//self.cfg.infer_duration + 1)
        ends = [(p+1)*self.cfg.infer_duration for p in range(parts)]
         
        # parts, ends => 5, [5, 10, 15, 20, 25]
        pseudo_max_end = ends[-1]
        
        
        if self.train:
            # .rand(*size, ...) => Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1).
            offset = torch.rand((1,)).numpy()[0] * max_offset
            # torch.rand((1,)).numpy()[0], offset => 0.7988023, 12.487610212154687
            
            # .load(...) => load an audio file as a floating point time series.
            # to preserve the native sampling rate of the file, use sr=None.
            # offset => start reading after this time (in seconds).
            # duration => only load up to this much audio (in seconds)
            audio_sample, orig_sr = lb.load(filepath, sr=None, mono=True,offset=offset, duration=work_duration)
            # orig_sr, audio_sample.shape => 32000, (320000,)
            
            if (self.resample)&(orig_sr != self.sr):
                # .resample(y, *, orig_sr, target_sr, ...) => resample a time series from orig_sr to target_sr.
                audio_sample = lb.resample(audio_sample, orig_sr, self.sr, res_type=self.res_type)

            if len(audio_sample) < work_audio_length:
                audio_sample = crop_or_pad(audio_sample, length=work_audio_length, is_train=self.train)
            
            # self_mixup_part, => 1
            # audio_sample.shape => (320000,)
            audio_sample = audio_sample.reshape((self_mixup_part,-1))
            # audio_sample.shape => (1, 320000)            
            audio_sample = np.sum(audio_sample,axis=0)

            if self.transforms is not None:
                audio_sample = self.transforms(audio_sample)

            if len(audio_sample) != self.audio_length:
                audio_sample = crop_or_pad(audio_sample, length=self.audio_length,is_train=self.train)
            
            # offset => 12.487610212154687
            # pseudo is made every 5s. For example, if offset=7 then the nearest_offset=5
            nearest_offset = int(np.round(offset/self.cfg.infer_duration) * self.cfg.infer_duration)
            # nearest_offset => 10           
            
            sample_ends = [str(nearest_offset+(i+1)*self.cfg.infer_duration) for i in range(int(work_duration/self.cfg.infer_duration)) if nearest_offset+(i+1)*self.cfg.infer_duration<=pseudo_max_end]
            # sample_ends => ['5', '10']
            
            # self.pseudo => None                    
            # use pseudo and hand label if the total duration of the audio is larger than clip duration
            if (work_duration < duration)&(self.pseudo is not None):
                if (version=='2023') | (version=='add') | (version=='scrap') | (version=='scrap_add') | (version=='scrap_add_add'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset1']['pseudo'],self.pseudo['subset1']['weight'])
                elif (version=='scrap_data'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset2']['pseudo'],self.pseudo['subset2']['weight'])
                elif (version=='scrap_data_add'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset3']['pseudo'],self.pseudo['subset3']['weight'])
                elif (version=='scrap_data_0515'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset4']['pseudo'],self.pseudo['subset4']['weight'])

        else:
            audio, orig_sr = lb.load(filepath, sr=None, mono=True,offset=0,duration=self.cfg.valid_duration)#valid_duration=>60
            # audio.shape => (796630,)
            if self.resample and orig_sr != self.sr:
                audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
                        
            audio_parts = int(np.ceil(len(audio)/self.audio_length))
            # audio_parts => 3
            # self.audio_length => 320000
            #audio_sample = [audio[i:(i+1)*self.audio_length] for i in range(audio_parts)]
            audio_sample = [audio[i*self.audio_length:(i+1)*self.audio_length] for i in range(audio_parts)]
            # audio_sample[0].shape, audio_sample[1].shape, audio_sample[2].shape => (320000,) (320000,) (156630,)
            
            # len(audio_sample[-1]) => 156630
            if len(audio_sample[-1])<self.audio_length:
                audio_sample[-1] = crop_or_pad(audio_sample[-1],length=self.audio_length,is_train=self.train)
                
            # audio_sample[-1].shape => (320000,)                    
            valid_len = int(self.cfg.valid_duration/self.duration)
            # valid_len => 6
            # self.audio_length => 320000
            if len(audio_sample)> valid_len:
                audio_sample = audio_sample[0:valid_len]
            elif len(audio_sample)<valid_len:
                diff = valid_len-len(audio_sample)
                padding = [np.zeros(shape=(self.audio_length,))] * diff
                audio_sample += padding

            audio_sample = np.stack(audio_sample)
            
            # audio_sample.shape => (6, 320000)
            
            sample_end = np.min([audio_parts * self.cfg.infer_duration, pseudo_max_end])
            sample_ends = [str(sample_end-i*self.cfg.infer_duration) for i in range(valid_len) if sample_end-i*self.cfg.infer_duration>0]

            if (work_duration < duration)&(self.pseudo is not None):
                if (version=='2023') | (version=='add') | (version=='scrap') | (version=='scrap_add') | (version=='scrap_add_add'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset1']['pseudo'],self.pseudo['subset1']['weight'])
                elif (version=='scrap_data'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset2']['pseudo'],self.pseudo['subset2']['weight'])
                elif (version=='scrap_data_add'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset3']['pseudo'],self.pseudo['subset3']['weight'])
                elif (version=='scrap_data_0515'):
                    target = self.adjust_label(labels,filename,sample_ends,target,version,self.pseudo['subset4']['pseudo'],self.pseudo['subset4']['weight'])

        audio_sample = torch.tensor(audio_sample[np.newaxis]).float()        
        # audio_sample.shape (train) => torch.Size([1, 320000])
        # audio_sample.shape (valid) => torch.Size([1, 6, 320000])
        
        # target =>
        # abethr1    0.0
        #           ... 
        # zebdov     0.0
        # Name: 1, Length: 834, dtype: float64
        
        return audio_sample, target

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = self.df_labels.loc[idx]
        weight = self.df.loc[idx,"weight"]
        
        if row['presence_type']!='foreground': weight = weight * 0.8        
            
        audio, target = self.load_data(self.df.loc[idx, "path"],target,row)
        target = torch.tensor(target).float()
        # target.shape, weight => torch.Size([834]), 0.6
        return audio, target , weight

def get_train_dataloader(df_train, df_valid, df_labels_train, df_labels_valid, sample_weight,cfg,pseudo=None,transforms=None):
    num_workers = multiprocessing.cpu_count()
    sample_weight = torch.from_numpy(sample_weight)
    
    # .WeightedRandomSampler(weights, num_samples, replacement=True, generator=None) => Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    # torch.DoubleTensor
    # replacement (bool) – if True, samples are drawn with replacement. 
    sampler = WeightedRandomSampler(sample_weight.type('torch.FloatTensor'), len(sample_weight),replacement=True)

    ds_train = BirdTrainDataset(
      df_train,
      df_labels_train,
      cfg,
      train = True,
      pseudo = pseudo,
      transforms = transforms,
    )
    
    ds_val = BirdTrainDataset(
      df_valid,
      df_labels_valid,
      cfg,
      train = False,
      pseudo = None,
      transforms=None,
    ) 
    
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size , sampler=sampler, num_workers = num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.test_batch_size, num_workers = num_workers, pin_memory=True) 
    
    return dl_train, dl_val, ds_train, ds_val