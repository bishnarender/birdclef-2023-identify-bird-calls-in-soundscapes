import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
import torch

import os

def prepare_cfg(cfg,stage):
    if stage in ["pretrain_ce","pretrain_bce"]:
        cfg.bird_cols = cfg.bird_cols_pretrain
    elif stage in ["train_ce","train_bce","finetune"]:
        cfg.bird_cols = cfg.bird_cols_train
    else:
        raise NotImplementedError

    if stage == 'finetune':
        cfg.DURATION = cfg.DURATION_FINETUNE
        cfg.freeze = True
    elif stage in ["pretrain_ce","pretrain_bce","train_ce","train_bce"]:
        cfg.DURATION = cfg.DURATION_TRAIN
    else:
        raise NotImplementedError

    cfg.test_batch_size = int(
        np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
    )
    
    # cfg.test_batch_size => 2
    cfg.train_part = int(cfg.DURATION / cfg.infer_duration)
    # cfg.train_part => 2
    return cfg

def train_test_split_(df,df_labels,seed):
    df_train = df
    df_labels_train = df_labels
    
    df_valid = pd.DataFrame(columns=df_train.columns)    
    # df_valid.head(3) =>
    # Empty DataFrame
    # Columns: [id, duration, version, ..., filename, path, presence_type]
    # Index: []
    # [0 rows x 54 columns]
    df_labels_valid = pd.DataFrame(columns=df_labels_train.columns)
    
    ## .... where are our weights???
    #df_train,df_valid,df_labels_train,df_labels_valid = train_test_split(df_train, df_labels_train, test_size=0.20, random_state=seed)
    
    return df_train.reset_index(drop=True),df_valid.reset_index(drop=True) ,df_labels_train.reset_index(drop=True) ,df_labels_valid.reset_index(drop=True)

def preprocess(cfg,stage):
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)
        audio = cfg.am_audio_transforms(audio,sample_rate=cfg.SR)
        return audio

    # primary_label_2023: used in birdclef2023, containing wrong label
    # primary_label_very_strict: original label from xeno-canto
    # primary_label_strict: fuse the ebird_code with same name but different number. ex: categr1	to categr
    # primary_label: fuse the same species with different ebird code: ['grbcam1',  'blkkit3',  'whcshr1', 'barowl8','barowl7','egwtea1','foxsp1','euhgul1']
    df = pd.read_csv(cfg.train_data)
    
    # df.columns => 
    #     Index(['id', 'duration', 'version', 'class', 'gen', 'sp', 'ssp', 'group', 'en',
    #        'rec', 'cnt', 'loc', 'lat', 'lng', 'alt', 'type', 'sex', 'stage',
    #        'method', 'url', 'file', 'file-name', 'sono', 'osci', 'lic', 'q',
    #        'length', 'time', 'date', 'uploaded', 'also', 'rmk', 'bird-seen',
    #        'animal-seen', 'playback-used', 'temp', 'regnr', 'auto', 'dvc', 'mic',
    #        'smp', 'rating', 'primary_label', 'secondary_labels',
    #        'primary_label_2023', 'secondary_labels_2023',
    #        'primary_label_very_strict', 'secondary_labels_very_strict',
    #        'recollect', 'primary_label_strict', 'secondary_labels_strict'],
    #       dtype='object')
    
    # df[['secondary_labels','secondary_labels_2023','secondary_labels_strict','secondary_labels_very_strict']].tail(3) =>
    #        secondary_labels secondary_labels_2023 secondary_labels_strict  secondary_labels_very_strict 
    # 119289               []                    []                      []                            []
    # 119290      ['eaywag1']                    []             ['eaywag1']                   ['eaywag1'] 
    # 119291      ['eaywag1']                    []             ['eaywag1']                   ['eaywag1']
    

    # literal_eval(node_or_string) => evaluate an expression node or a string containing only a Python literal or container display. The string or node provided may only consist of the following Python literal structures: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None and Ellipsis.
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: literal_eval(x))    
    df['secondary_labels_2023'] = df['secondary_labels_2023'].apply(lambda x: literal_eval(x))
    df['secondary_labels_strict'] = df['secondary_labels_strict'].apply(lambda x: literal_eval(x))
    df['secondary_labels_very_strict'] = df['secondary_labels_very_strict'].apply(lambda x: literal_eval(x))    
    
    df['version'] = df['version'].astype(str)    
    
    # df['q'][0:3] =>
    # 0    C
    # 1    C
    # 2    B
    # Name: q, dtype: object    
    
    # df['q'].map({'A':5,'B':4,'C':3,'D':2,'E':1,'no score':0})[0:3] =>
    # 0    3.0
    # 1    3.0
    # 2    4.0
    # Name: q, dtype: float64
    
    # .mask(cond, other=_NoDefault.no_default, *, inplace=False, axis=None, level=None) => Replace values where the condition is True. Entries where cond is True are replaced with corresponding value from other. 
    df['rating'] = df['rating'].mask(np.isnan(df['rating'].values),df['q'].map({'A':5,'B':4,'C':3,'D':2,'E':1,'no score':0}))
    df['filename'] = df['id'].apply(lambda x: f'XC{x}')    
    df['path'] = df['id'].apply(lambda x: os.path.join(cfg.train_dir,f'XC{x}.ogg'))
    
    # .all(...) => return whether all elements are True, potentially over an axis.
    # ensure all the train data is available
    if not df['path'].apply(lambda x:os.path.exists(x)).all():
        print('===========================================================')
        print('warning: missing audio files in ./inputs/train_audios')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df = df[df['path'].apply(lambda x:os.path.exists(x))].reset_index(drop=True) # my comment 

    labels = np.zeros(shape=(len(df),len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels,columns=cfg.bird_cols)
    class_sample_count = {col:0 for col in cfg.bird_cols}
    # class_sample_count => {'abethr1': 0, 'abhori1': 0, 'abythr1': 0, ...}
    
    include_in_train = []
    presence_type = []
    
    # cfg.primary_label_col => 'primary_label_2023'
    # cfg.secondary_labels_col => 'secondary_labels_2023'
    
    # df[cfg.secondary_labels_col].values[5:10] => [list([]) list(['vimwea1']) list([]) list([]) list([])]
    # df[cfg.primary_label_col] =>
    # 0         yebapa1
    # 1         yebapa1
    #            ...   
    # 119287    tacsun1
    # 119291     skylar
    # Name: primary_label_2023, Length: 119292, dtype: object    
    
    # df[cfg.primary_label_col].unique() => ['yebapa1' 'combuz1' 'chibat1' 'carcha1' 'beasun2' 'carwoo1' 'wlwwar' ...]
    
    
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        include = False
        presence = 'background' if primary_label!='soundscape' else 'soundscape'
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i,primary_label] = 1
            class_sample_count[primary_label] += 1
            
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i,secondary_label] = cfg.secondary_label
                class_sample_count[secondary_label] += cfg.secondary_label_weight #  cfg.secondary_label_weight => 0.5
        presence_type.append(presence)
        include_in_train.append(include)
        
    # len(include_in_train) => 119292
    df['presence_type'] = presence_type
    
    # include training rows.
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)
    
    # df['presence_type'].unique() => ['foreground']
    # ((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')
    df_labels[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)
    df = df[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)

    df_train,df_valid,df_labels_train,df_labels_valid = train_test_split_(df,df_labels,cfg.seed[stage])
    
    sample_weight = np.zeros(shape=(len(df_train,)))
    for i,(primary_label, secondary_labels) in enumerate(zip(df_train[cfg.primary_label_col].values,df_train[cfg.secondary_labels_col].values)):
        if primary_label in cfg.bird_cols:
            sample_weight[i] = 1.0/(class_sample_count[primary_label])
        else:
            secondary_labels_include = [secondary_label for secondary_label in secondary_labels if secondary_label in cfg.bird_cols]
            sample_weight[i] = np.mean([1.0/class_sample_count[secondary_label] for secondary_label in secondary_labels_include])

    return df_train, df_valid, df_labels_train, df_labels_valid, sample_weight, transforms