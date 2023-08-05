import numpy as np

# this class is used to convert a signal from monopolar to bipolar montage, using the CDAC data convention
#input 20 channels, 19 monopolar, 1 EKG
#output 18 channels, all bipolar
class CDAC_bipolar_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

    
    def __call__(self,signal):
        signal = signal[self.bipolar_ids[:,0]] - signal[self.bipolar_ids[:,1]]
        return signal