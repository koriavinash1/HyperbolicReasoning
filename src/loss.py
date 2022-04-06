import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from hpenalty import hessian_penalty
from torch.autograd import grad as torch_grad



#def recon_loss():
#    mse = nn.MSELoss(reduction='mean')
 #   l1 = nn.L1Loss(reduction='mean')
  #  def loss(logits, target):
 #       return 0.99*mse(logits, target) +\
 #                0.01*l1(logits, target)
  #  return loss

def recon_loss(logits, target):
    return 0.99 * F.mse_loss(logits, target) + \
               0.01 * F.l1_loss(logits, target)


def ce_loss(logits, target, label_smoothing=0.0):

    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    normalization = images.shape[2] * images.shape[3]
        
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(normalization)
    outputs = (images*pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, 
                          retain_graph=True, 
                          only_inputs=True)[0]

    if len(pl_grads.shape) > 2:
        return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
    else:
        return (pl_grads ** 2).mean(dim=1).sqrt()


def hpenalty(model, inputs, G_z=None, **model_kwargs):
    penalty = hessian_penalty(model, inputs, G_z=G_z, **model_kwargs)
    return penalty 


def orthonormal_loss(output, max_eigen=2):
    output_ = output.clone()
    shape = output.shape
    # if shape[0] < shape[1]:
    #     baseline = torch.zeros_like(output_)
    #     for i in range(shape[1]//shape[0] + 1):
    #         if i*shape[0] > shape[1]: break
    #         subspace = output_[:, i*shape[0]:(i+1)*shape[0]]
    #         a, b = torch.linalg.qr(subspace) 
    #         baseline[:, i*shape[0]:(i+1)*shape[0]] = a
    # else:
    #     a, b = torch.linalg.qr(output_) 
    #     baseline = a

    if shape[0] > shape[1]:
        u, s, _ = torch.svd(output_)
        s = max_eigen*s/torch.norm(s)
        baseline = torch.mm(u, torch.diag(s)).detach()
    else:
        _, s, u = torch.svd(output_)
        s = max_eigen*s/torch.norm(s)
        baseline = torch.mm(u, torch.diag(s)).detach().T
    return ((output - baseline)**2).sum(dim=0).sqrt()
