from torch.nn import BCELoss
from pytorch_lightning import LightningModule
import torch
import numpy as np
# "." allows to import from the same directory
from .net1d import Net1D
from pytorch_metric_learning import losses
from .heads import RegressionHead
from .losses import WeightedFocalLoss,GHM_Loss

weights = [2.58306541, 11.210725, 3.30010517, 10.2310366, 17.13606388, 15.49333892]
weights_num = np.array(weights)
class_weights = torch.from_numpy(weights_num).float().to(torch.device('cuda:{}'.format(0)))
# create a specialized finetuning module


class FineTuning(LightningModule):
    def __init__(self,lr,n_channels=37, **kwargs):
        super().__init__()
        self.lr = lr

        self.model = Net1D(
                    in_channels=n_channels, 
                    base_filters=64, 
                    #base_filters=32, 
                    ratio=1, 
                    #filter_list = [64,128,128,160,160,256,256],#
                    filter_list=[64,160,160,400,400,1024,1024], 
                    m_blocks_list=[2,2,2,3,3,4,4], 
                    kernel_size=16, 
                    stride=2, 
                    groups_width=16,
                    verbose=False, 
                    use_bn=True,
                    #n_classes=1, #soft target
                    n_classes=1 #hard target
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        device = 'cuda:0'
        logits = self.forward(x)
        #loss_function = losses.SphereFaceLoss(num_classes=2, embedding_size=128, margin=3, scale=1).to(device)
        loss_function = WeightedFocalLoss()
        #loss_function = WeightedKLDivWithLogitsLoss(class_weights)
        loss = loss_function(logits, y)
        self.log('train_loss', loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        device = 'cuda:0'
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        #loss_function = losses.SphereFaceLoss(num_classes=2, embedding_size=128, margin=3, scale=1).to(device)
        loss_function = WeightedFocalLoss()
        #loss_function = WeightedKLDivWithLogitsLoss(class_weights)
        loss = loss_function(logits, y)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        preds = self.forward(signals)
        # compute and log loss
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
