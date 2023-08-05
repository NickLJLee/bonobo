import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedKLDivWithLogitsLoss(nn.KLDivLoss):
    def __init__(self, weight):
        super(WeightedKLDivWithLogitsLoss, self).__init__(size_average=None, reduce=None, reduction='none')
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        # TODO: For KLDivLoss: input should 'log-probability' and target should be 'probability'
        # TODO: input for this method is logits, and target is probabilities
        batch_size = input.size(0)
        log_prob = F.log_softmax(input, 1)
        element_loss = super(WeightedKLDivWithLogitsLoss, self).forward(log_prob, target)

        sample_loss = torch.sum(element_loss, dim=1)
        sample_weight = torch.sum(target * self.weight, dim=1)

        weighted_loss = sample_loss * sample_weight
        # Average over mini-batch, not element-wise
        avg_loss = torch.sum(weighted_loss) / batch_size

        return avg_loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()