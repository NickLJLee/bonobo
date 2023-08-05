from torch.nn import BCELoss
from pytorch_lightning import LightningModule
import torch

# "." allows to import from the same directory
from .BIOTEncoder import BIOTEncoder
from .heads import RegressionHead

# create a specialized finetuning module
class FineTuning(LightningModule):
    def __init__(self,lr,head_dropout=0.3, emb_size=256, heads=8, depth=4,n_channels=18, **kwargs):
        super().__init__()
        self.lr = lr
        # create the BIOT encoder
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth,n_channels=n_channels)
        # create the regression head
        self.head = RegressionHead(emb_size,head_dropout)

    def forward(self, x):
        x = self.biot(x)
        x = self.head(x)
        return x

    def training_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        loss_function = BCELoss()
        loss = loss_function(logits, y)
        self.log('train_loss', loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        # flatten label
        y = y.view(-1, 1).float()
        logits = self.forward(x)
        loss_function = BCELoss()
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
        return optimizer