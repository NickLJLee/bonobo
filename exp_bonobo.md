# Experiment of BONOBO



#### NO.89 

channel=18 epoch=100 focal loss



AUC=91.4 **SOTA**



#### NO.90

channel=18 epoch=100 BCE loss



AUC=89.2



#### NO.91

filter_list=[64,160,160,400,400],

m_blocks_list=[2,2,2,3,3],

channel=18 epoch=100 focal loss



AUC=90.5



#### NO.92

channel=**37** epoch=100 focal loss

wrong



#### NO.93

channel=**37** epoch=100 focal loss



AUC=89

problem: flip's p=0.5



#### NO.94

channel=**37** epoch=100 focal loss

wrong



#### NO. 95 96 97 98 99

channel=**37** epoch=100 focal loss

flip's p=0.5

AUC=87



#### NO. 100 1 2 3 4

channel=**37** epoch=100 focal loss

flip's p=1

AUC=89



#### NO. 105-8

wrong



#### NO. 109

channel=**37** epoch=100 **GHM**

flip's p=1

AUC=86



#### NO. 110-4

channel=**37** epoch=100 **GHM**

bipolar flip's p=0.5

common average flip's p=1

AUC=84



#### NO. 115

channel=**37** epoch=**50** focal loss

bipolar flip's p=1

common average flip's p=1

AUC=79



#### NO. 116

channel=**37** epoch=**100** focal loss

bipolar flip's p=0.5

common average flip's p=1

AUC=79



#### NO. 117

channel=**37** epoch=**100** focal loss

bipolar flip's p=1

common average flip's p=1

AUC=89



#### NO. 118

channel=**37** epoch=**100** focal loss

bipolar flip's p=0

common average flip's p=0

AUC=85



#### NO. 119

wrong



#### NO. 120

channel=**37** epoch=**100**  focal loss

​          base_filters=32, 

​          ratio=1, 

​          filter_list = [64, 128, 128, 160, 160, 256, 256],

AUC=90.36



#### NO. 121

wrong



#### NO. 122 pseudo

**Big dataset**

channel=**37** epoch=**100**  focal loss

​          base_filters=32, 

​          ratio=1, 

​          filter_list = [64, 128, 128, 160, 160, 256, 256],





#### NO. 123 pseudo

**Big dataset**

channel=**37** epoch=**20** focal loss

​          base_filters=32, 

​          ratio=1, 

​          filter_list = [64, 128, 128, 160, 160, 256, 256],

lr=1e-2



#### NO. 124 pseudo

**Big dataset**

channel=**37** epoch=**20** focal loss

​          base_filters=32, 

​          ratio=1, 

​          filter_list = [64, 128, 128, 160, 160, 256, 256],

lr=1e-3



#### NO. 125 pseudo

**Big dataset**

channel=**37** epoch=**20** focal loss

​          **base_filters=64,** 

​          **ratio=1,** 

​          **filter_list=[64,160,160,400,400,1024,1024],** 

lr=1e-4



#### NO. 126 pseudo

**Big dataset**

channel=**37** epoch=**20** focal loss

​          **base_filters=64,** 

​          **ratio=1,** 

​          **filter_list=[64,160,160,400,400,1024,1024],** 

lr=1e-1



#### NO. 126 pseudo_test

**Big dataset** 

channel=**37** epoch=**10** focal loss

​          **base_filters=64,** 

​          **ratio=1,** 

​          **filter_list=[64,160,160,400,400,1024,1024],** 

lr=1e-2 batchsize=256





#### NO. 24 

channel=**37**（modify） epoch=**50** focal loss

AUC=91.5 **SOTA**



#### NO. 25

channel=**37**（modify） epoch=**50** focal loss

​          **base_filters=64,** 

​          **ratio=1,** 

​          **filter_list=[64,160,160,400,400,1024,1024],** 

AUC=86.4



#### NO. 26

wrong



#### NO. 27

channel=**37**（modify） epoch=100 focal loss

WINDOWSIZE: float= **2**

AUC=91.86 **SOTA**



#### NO. 28 （93.71）

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **2**

AUC=93.71 **SOTA**



#### NO. 29 （94.39）

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

AUC=94.39 **SOTA**



#### NO. 30

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **5**

AUC=91



#### NO. 31

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **10**

AUC=92.78



#### NO. 32

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

AUC=90.7



#### NO. 33-37

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

AUC=



#### NO. 38-42

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **10**

AUC=



#### NO. 43 Hard_label

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

optimization: SGD

AUC=



#### NO. 44-46 Hard_label

wrong



#### NO. 47 Hard_label

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

optimization: Adam

AUC=93.36



#### NO. 48 Hard_label

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

optimization: Adam

AUC=92.18



#### NO. 49 

wrong



#### NO. 50 Soft_label but delete 0.3-0.7(94.28)

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **10**

Soft_label but delete 0.3-0.7

AUC=94.28



#### NO. 51

low



#### NO. 52

channel=**37**（modify） epoch=150 focal loss

WINDOWSIZE: float= **3**

AUC=92.87



#### NO. 53

channel=19 only ref epoch=150 focal loss

WINDOWSIZE: float= **2**

AUC=88.4



#### NO. 54

channel=19 only ref epoch=150 focal loss

WINDOWSIZE: float= **10**

AUC=87.2



# Spike localization task

#### NO.161

1Focal+1MSE epoch=150 WINDOWSIZE: float= 2

AUC=

MAE=



#### NO.162

1Focal+10MSE epoch=150 37chan **spike_det_weights-v11**

AUC=92.05

MAE={'Fp1': 0.08658641843686415,
 'F3': 0.08416371820066008,
 'C3': 0.07853006364937162,
 'P3': 0.08418371861413525,
 'F7': 0.09125622545627768,
 'T3': 0.09754284981396767,
 'T5': 0.09133072181885099,
 'O1': 0.08464283781852183,
 'Fz': 0.08489445516201675,
 'Cz': 0.07967883485454817,
 'Pz': 0.07772454432359067,
 'Fp2': 0.08779408646271693,
 'F4': 0.08565538049557152,
 'C4': 0.08613122042305804,
 'P4': 0.09069318759994015,
 'F8': 0.0957346618186104,
 'T4': 0.09955740034945541,
 'T6': 0.09061809114410532,
 'O2': 0.08431843773064032}



#### NO.163

1Focal+100MSE epoch=150 37chan

AUC=87

MAE={'Fp1': 0.09118203404694196,
 'F3': 0.087797770330383,
 'C3': 0.08252587631807301,
 'P3': 0.08965777214580492,
 'F7': 0.0951355596550389,
 'T3': 0.10072176009687611,
 'T5': 0.0982968121725015,
 'O1': 0.09238413621748175,
 'Fz': 0.08866784351080792,
 'Cz': 0.08561069153371034,
 'Pz': 0.08561353007265111,
 'Fp2': 0.09432719966074209,
 'F4': 0.09091474052999401,
 'C4': 0.09096884142244165,
 'P4': 0.09864776463548773,
 'F8': 0.0997747615712268,
 'T4': 0.10325471138491919,
 'T6': 0.09620517295272292,
 'O2': 0.09545883383077199}



# Hard mine

#### NO.210

AUC = 95

/run/media/exx/Expansion2/code/Spike_37chan/Models/YOUR_MODEL_NAME/hardmine-chan_weights-v1.ckpt





#### NO.234(95.95)

AUC=95.95

**select_10000** samples as hard mining dataset

/run/media/exx/Expansion2/code/Spike_37chan/Models/YOUR_MODEL_NAME/select-hardmine-chan_weights-v2.ckpt



#### NO.2xx

AUC=95.65

whole negative samples

/run/media/exx/Expansion2/code/Spike_37chan/Models/YOUR_MODEL_NAME/round3-hardmine-chan_weights-v7.ckpt





#### NO.342

AUC=96.16

round3 hard mining, multiply the positive 4 times

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round3-hardmine-chan_weights-9616.ckpt





#### NO.342（96.68）

AUC=96.68

round4 hard mining, multiply the positive 4 times

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round4-hardmine-chan_weights.ckpt



#### NO.356（96.82）

AUC=96.82

round5 hard mining, multiply the positive 4 times, BCEloss

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round5-hardmine-chan_weights-9682.ckpt



#### NO.365

AUC=

**round5** hard mining, 

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round5-hardmine-chan_weights-SOTA_in_controlset.ckpt



#### NO.366（97.11）

AUC=97.11

**round5** hard mining, BCEloss

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round5-hardmine-chan_weights-9711.ckpt



#### NO.367（97.11）

AUC=97.11

**round6** hard mining,, **BCEloss**

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round5-hardmine-chan_weights-v2.ckpt



#### NO.368（97.13）

AUC=97.13

**round6** hard mining, **BCEloss**

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round5-hardmine-chan_weights-v3.ckpt





#### NO.369（96.91）

AUC=96.91

**round6** hard mining,  **BCEloss**

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round6-hardmine-chan_weights.ckpt





#### NO.387（97.92）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round8-hardmine-chan_weights.ckpt

8 round hard mining 

not better than 7 rounds



#### NO.406-SCDNN（96.54）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/SCDNN-hardmine-chan_weights-v4.ckpt

8 round hard mining 

training loss lowest



#### NO.413-SCDNN（96.68）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round9-SCDNN-hardmine-chan_weights.ckpt

9 rounds hard mining。SCDNN

```
Specificity: 0.9766
```



#### NO.418-net1d（96.61）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round9-net1d-hardmine-chan_weights.ckpt

9 rounds hard mining。net1d

```
Specificity: 0.9766
```



#### NO.420-net1d（97.06）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round10-net1d-hardmine-chan_weights.ckpt

10 rounds hard mining。net1d

num_pos_augmentations=6



#### NO.421-net1d（96.99）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round10-net1d-hardmine-chan_weights-v1.ckpt

10 rounds hard mining。net1d

num_pos_augmentations=7

```
Specificity: 0.9727
```





#### NO.422-net1d（96.81）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round10-net1d-hardmine-chan_weights-v2.ckpt

10 rounds hard mining。net1d

num_pos_augmentations=7

```
Specificity: 0.9844
```



#### NO.422-net1d（97.01）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round11-net1d-hardmine-chan_weights-v1.ckpt

11 rounds hard mining。net1d

num_pos_augmentations=7

```
Specificity: 0.9766
```





#### NO.4xx-net1d（96.87）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/1s-round11-hardmine-chan_weights-v1.ckpt

11 rounds hard mining。net1d

w**indow size=1second**

BCEloss

```
Specificity: 0.9648
Recall:0.8747
```



#### NO.436-net1d（97.18）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/round12-BCE-hardmine-chan_weights.ckpt

12 rounds hard mining。w**indow size=1second**

BCEloss

```
Specificity: 0.9609
Recall: 0.8891
```





#### NO.486-net1d（95.82）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/focal_loss_10s_Resnet_chan_weights.ckpt

NO hard mining。**window size=10 second**

Focal loss

```
Specificity: 0.9141
Recall: 0.8914
```





#### NO.491-net1d（86.66/96.68）

/data/0shared/lijun/code/eeg/Spike_37chan/Models/YOUR_MODEL_NAME/member_iiic_12hm_BCE/member_iiic_12hm_BCE.ckpt

**member+12 hm + iiic**

BCE loss

epoch: 10

roc_auc: 0.8666                                                                                                                                                                                                                                                                                                                                                         

Specificity: 0.9414                                                                                                                                                                                                                                                                                                                                                     

Recall: 0.6109



epoch: 30

roc_auc: 0.9668                                                                                                                                                                                                                                                                                                                                                        

Specificity: 0.9727                                                                                                                                                                                                                                                                                                                                                     

Recall: 0.8060  

