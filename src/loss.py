import torch.nn.functional as F
from torch import nn



#def recon_loss():
#    mse = nn.MSELoss(reduction='mean')
 #   l1 = nn.L1Loss(reduction='mean')
  #  def loss(logits, target):
 #       return 0.99*mse(logits, target) +\
 #                0.01*l1(logits, target)
  #  return loss

def recon_loss(logits, target):
    return F.mse_loss(logits, target)
            #    0.01 * F.l1_loss(logits, target)


def ce_loss(logits, target):
    return F.cross_entropy(logits, target)

# Entropy regularisation loss 
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
