# config.py

from dataclasses import dataclass

@dataclass
class Config:

    # Data params
    PATH_FILES_BONOBO: str = '/home/ubuntu/data/bonobo/npy/' #'/home/ubuntu/data/bonobo/pseudo/'
    PATH_LUT_BONOBO: str = '/home/ubuntu/code/Spike_37chan/soft_label.csv' #'lut_labelled_20230628.csv'
    PATH_CONTINOUS_EEG: str = '/bdsp/staging/Bonobo/Datasets/continuousEEG'

    FQ: int = 128 # Hz
    
    
    # Preprocessing 
    MONTAGE: str = 'bipolar'
    WINDOWSIZE: float = 3 # 1 seconds
    
	# Model parameters
    N_FFT: int = 128
    HOP_LENGTH: int = 64
    HEAD_DROPOUT: int = 0.3
    EMB_SIZE: int = 256
    HEADS: int = 8
    DEPTH: int = 4
    N_CHANNELS: int = 37 #19+18

    # training parameters
    BATCH_SIZE: int = 128 #test 128
    LR: float = 1e-4 #test 1e-4

    def print_config(self):
        print('THIS CONFIG FILE CONTAINS THE FOLLOWING PARAMETERS :\n')
        for key, value in self.__dict__.items():
            print(key, value)