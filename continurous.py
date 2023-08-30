from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import pickle
from torchvision import transforms
import pytorch_lightning as pl
import torch

# load own code
import sys
sys.path.append('../')
from sleeplib.Resnet_15.model import FineTuning
from sleeplib.datasets import BonoboDataset, ContinousToSnippetDataset
# this holds all the configuration parameters
from sleeplib.config import Config
import pickle

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from sleeplib.datasets import BonoboDataset , ContinousToSnippetDataset
from sleeplib.montages import CDAC_bipolar_montage,CDAC_common_average_montage,CDAC_combine_montage,con_combine_montage
from sleeplib.transforms import cut_and_jitter, channel_flip

path_model = 'Models/YOUR_MODEL_NAME'
# load config file
with open(path_model+'/config.pkl', 'rb') as f:
   config = pickle.load(f)
# load dataset
df = pd.read_csv(config.PATH_LUT_BONOBO,sep=';')
# fraction filter
frac_filter = (df['fraction_of_yes'] > 6/8) | (df['fraction_of_yes'] < 2/8)
mode_filter = df['Mode'] == 'Test'
extreme_quality_filter = df['total_votes_received'] >= 8
quality_filter = df['total_votes_received'] > 2

test_df = df[mode_filter]
print(f'there are {len(test_df)} test samples')

# set up dataloader to predict all samples in test dataset
transform_val = transforms.Compose([cut_and_jitter(windowsize=config.WINDOWSIZE,max_offset=0,Fq=config.FQ)])
combine_montage = CDAC_combine_montage()
con_combine_montage = con_combine_montage()
'''
test_dataset = BonoboDataset(test_df, config.PATH_FILES_BONOBO, 
                           transform=transform_val,
                           montage = combine_montage
                          )
test_dataloader = DataLoader(test_dataset, batch_size=32,shuffle=False,num_workers=os.cpu_count())
for x, y in test_dataloader:
    with torch.no_grad():
        print(x.shape)
        print(y)
        break

'''
Bonobo_con = ContinousToSnippetDataset('/home/ubuntu/data/Bonobo01742_0.mat',montage=con_combine_montage)
con_dataloader = DataLoader(Bonobo_con, batch_size=config.BATCH_SIZE,shuffle=False,num_workers=os.cpu_count())

# load pretrained model
model = FineTuning.load_from_checkpoint('/home/ubuntu/code/Spike_37chan/Models/YOUR_MODEL_NAME/37chan_weights-v2.ckpt',
                                        lr=config.LR,
                                        head_dropout=config.HEAD_DROPOUT,
                                        n_channels=config.N_CHANNELS,
                                        n_fft=config.N_FFT,
                                        hop_length=config.HOP_LENGTH,
                                       )
                                        #map_location=torch.device('cpu') add this if running on CPU machine
# init trainer
trainer = pl.Trainer(fast_dev_run=False,enable_progress_bar=False)

# store results
path_controls = os.path.join('/home/ubuntu/code/vergleicher/Judge/testset_controls.csv')
controls = pd.read_csv(path_controls)
i = 0
for eeg_file, timesteps in zip(controls.eeg_file, controls.timesteps):
    path = '/home/ubuntu/data/bonobo/con/'+eeg_file+'.mat'
    Bonobo_con = ContinousToSnippetDataset(path,montage=con_combine_montage)
    con_dataloader = DataLoader(Bonobo_con, batch_size=128,shuffle=False,num_workers=os.cpu_count())
    
    preds = trainer.predict(model,con_dataloader)
    preds = np.concatenate(preds)
    
    preds = pd.DataFrame(preds)
    preds.to_csv(path_model+'/con/'+ eeg_file +'.csv',index=False)
    i = i + 1
    if i % 100 == 0:
      print(i)
    