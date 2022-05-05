import torch.nn.functional as F




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
