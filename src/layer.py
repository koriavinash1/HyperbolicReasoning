import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
import einops
from einops import rearrange
import geomstats.backend as gs
from geomstats.geometry.hypersphere import \
Hypersphere

from src.manifolds import PoincareBall
from src.mathutils import artanh, tanh, arcosh, cosh, sinh

from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
import torch.nn.init as init

from src.reasoning import BinaryLinear
from torch.nn.modules.module import Module

def nonlinearity(x):
    # swish
    return F.relu6(x)#*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.BatchNorm2d(in_channels)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, 
                            scale_factor=2.0, 
                            mode="bilinear", 
                            align_corners=False)
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.conv1(h)
        h = nonlinearity(h)

        h = self.norm2(h)
        h = self.conv2(h)
        h = nonlinearity(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class Decoder(nn.Module):
        def __init__(self, *, ch = 32, 
                        out_ch = 3, 
                        in_channels=3,
                        z_channels = 64, 
                        ch_mult=(1, 2, 4, 8), 
                        num_res_blocks=2,
                        dropout=0.0, 
                        resamp_with_conv=True, 
                        input=(64, 4, 4)):
            super(Decoder, self).__init__()
            self.ch = ch
            
            self.num_resolutions = len(ch_mult)
            block_in = ch * ch_mult[self.num_resolutions - 1]

            self.input = input
            self.inchannel = np.prod(self.input)
            self.num_res_blocks = num_res_blocks
            self.in_channels = in_channels


            self.conv_in = torch.nn.Conv2d(z_channels,
                                           block_in,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

            self.midblock_1 = ResnetBlock2D(in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

            self.midblock_2 = ResnetBlock2D(in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

            # downsampling




            upl = []
            for i_level in reversed(range(self.num_resolutions)):
                if i_level != 0:
                    upl.append(Upsample(block_in, resamp_with_conv))

                block_out = ch * ch_mult[i_level]
                for i_block in range(self.num_res_blocks):
                    upl.append(ResnetBlock2D(in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                    block_in = block_out





            self.up = nn.Sequential(*upl)

            # middle


            # end
            self.norm_out = Normalize(block_in)
            self.conv_out = torch.nn.Conv2d(block_in,
                                            out_ch,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        def forward(self, x):
            # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

            # timestep embedding
            x1 = self.conv_in(x)
            x1 = nonlinearity(x1)


            x2 = self.midblock_1(x1)
            x3 = self.midblock_2(x2)
            
            x4 = self.up(x3)
            x5 = self.norm_out(x4)
            x6 = self.conv_out(x5)

            return  nonlinearity(x6)


class VQmodulator(nn.Module):

    def __init__(self, *, features,dropout=0.0, z_channels, codebooksize, device):
        super(VQmodulator, self).__init__()
        self.norm1 = nn.BatchNorm2d(features, affine=True)

        self.conv1 = torch.nn.Conv2d(features,
                                       z_channels,
                                       kernel_size=1,
                                       stride=1,
                                       )

        self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                      z_channels,
                                      kernel_size=1,
                                      stride=1,
                                      )
        
        self.quantize = VectorQuantizer2DHS(device, codebooksize, z_channels, beta=1.0, sigma=0.1)

    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x1)
        z_q, loss, distances, info, zqf, ce, td, hrc, r = self.quantize(x2)

        return loss, z_q, zqf, ce, td, hrc, r, None, None


class VectorQuantizer2DHS(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self,
                    device,  
                    n_e = 128, 
                    e_dim = 16, 
                    beta = 0.9, 
                    ignorezq = False,
                    disentangle = True,
                    remap=None, 
                    unknown_index="random",
                    sane_index_shape=False, 
                    legacy=True, 
                    sigma = 0.1):

        super().__init__()
        '''
        n_e : total number of codebook vectors
        e_dim: codebook vector dimension
        beta: factor for legacy term
        '''
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma
        self.device = device
        self.ignorezq = ignorezq
        self.disentangle = disentangle
        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        # uniformly sampled initialization
        sphere = Hypersphere(dim=self.e_dim - 1)
        self.embedding = nn.Embedding(self.n_e, 
                                            self.e_dim)

        self.h =  HNNLayer(e_dim, e_dim, 1, 0, True)
        # self.h1 =  HypLinear(e_dim, e_dim, 1, 0, True)        


        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True

        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


        self.hsreg = lambda x: [ torch.norm(x[i]) for i in range(x.shape[0])]
        self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(device)
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]
        

        # remap
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.clamp_class = Clamp()


    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None, None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def rbf(self, d):
        return (d ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

    def projp(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.epsilon)
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
    
    def dist(self, u, v):
        sqdist = torch.sum(u ** 2, dim=1,  keepdim=True) + \
                 torch.sum(v ** 2, dim=1) - 2 * \
                 torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        squnorm = torch.sum(v ** 2, dim=1,   keepdim=True)
        sqvnorm = torch.sum(u ** 2, dim=1,  keepdim=True)
        l = torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        #x = 1 + 2 * sqdist / torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        x = 1 + 2 * sqdist/torch.t(l)
        y =  torch.clamp(x ** 2, min=1.0 + self.epsilon)
        #x = 1 + 2 * sqdist / torch.mm((1 - squnorm), (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(y - 1)
        return torch.log(x + z)


    
    def disth(self, u, v):
        sqdist = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        """        
        u = torch.t(u[:,0].repeat(u.shape[1], 1))
        v = (v[:, 0].repeat(v.shape[1], 1))
        uv = torch.einsum('bd,dn->bn', u, v)
        theta = sqdist - 2* uv
        theta = torch.clamp(theta, min=1.0 + self.epsilon)
        return arcosh(theta) ** 2
        """
        theta = torch.clamp(sqdist, min=1.0 + self.epsilon)
        return arcosh(theta) ** 2
    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.epsilon))
    
    def sdisth(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(prod / K, min=1.0 + self.epsilon)
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)
    

    def sdisthh(self, u, v):
        res = torch.sum(u * v, dim = 1)
    
    def HLoss(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, 
                    prev_cb = None,
                    attention_w = None,
                    temp=None, rescale_logits=False, return_logits=False):
        
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        z_flattened = z.view(-1, self.e_dim)


        loss = 0
        prevcb = cb_attn = None
                
        if (prev_cb is None):       
            # reshape z -> (batch, height, width, channel) and flatten
            # z = rearrange(z, 'b c h w -> b h w c').contiguous()

            # intra distance (gdes-distance) between codebook vector 
            d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
            ed1 = torch.tensor(self.ed(self.embedding.weight))
            ed1 = ed1.repeat(self.n_e, 1)
            ed2 = ed1.transpose(0,1)
            ed3 = ed1 * ed2

            edx = d1/ed3.to(self.device)
            edx = torch.clamp(edx, min=-0.99999, max=0.99999)
            # d = torch.acos(d)
            d1 = torch.acos(edx)
            

            min_distance = torch.kthvalue(d1, 2, 0)
            total_min_distance = torch.mean(min_distance[0])
            codebookvariance = torch.mean(torch.var(d1, 1))
            # codebookvariance = torch.var(min_distance[0])
 
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
            
            min_encoding_indices = torch.argmin(d, dim=1)
            
            z_q = self.embedding(min_encoding_indices).view(z.shape)
            zn = torch.nn.functional.normalize(z_q).view(z.shape) # get quantized vector and normalize
        else:
            #prevcb = self.to_poincare(self.h1(self.expmap0(torch.nn.functional.normalize(prev_cb.clone().detach()),1)),1)
            #prevcb = self.to_poincare(self.h1(self.expmap0(prev_cb.clone().detach(),1)),1)
            cb_attn = torch.einsum('md,mn->nd', 
                                    self.logmap0(self.h(self.expmap0(prev_cb.clone().detach(),1)),1), 
                                    attention_w)
            cb_attnx = self.to_poincare(self.expmap0(cb_attn, 1), 1)
            zfl = self.to_poincare(self.expmap0(z_flattened,1), 1)
            dd = self.dist(zfl, cb_attnx)
            min_encoding_indices = torch.argmin(dd, dim =1)
            
            
            z_q = self.logmap0(self.to_hyperboloid(cb_attnx[min_encoding_indices], 1),1).view(z.shape)
            zn = torch.nn.functional.normalize(z_q)            
            #cb_loss = F.mse_loss(z_q2, z_q2)
            #print(cb_loss)

            d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))

            min_distance = torch.kthvalue(d1, 2, 0)
            total_min_distance = torch.mean(min_distance[0])
            codebookvariance = torch.mean(torch.var(d1, 1))



        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw))


        # compute loss for embedding
        if  (prev_cb is None):
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
                if not self.ignorezq:
                    loss += torch.mean((z_q - z.detach()) ** 2) 
            else:
                loss = torch.mean((z_q.detach() - z) ** 2)  
                if not self.ignorezq:
                    loss += self.beta * torch.mean((z_q - z.detach()) ** 2)

            disentanglement_loss = codebookvariance - total_min_distance
            if self.disentangle:
                loss += hsw
                loss += disentanglement_loss


        # preserve gradients
        z_q = z + (z_q - z).detach()


        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return (z_q, loss,
                    (sampled_idx, min_encoding_indices.view(z.shape[0], -1)), 
                    codebookvariance, 
                    total_min_distance,  
                    hsw, 
                    torch.mean(self.r), prevcb, attention_w,  cb_attn, zn)
                    

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.9, max=1.1) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class weightConstraint2(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w = torch.heaviside(w, torch.tensor([0.0]))
            x = w.shape[0]
            module.weight.data=w



class GumbelQuantize2DHS(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, device,  
                    n_e = 128, 
                    e_dim = 16, 
                    beta = 0.9, 
                    ignorezq = False,
                    disentangle = True,
                    remap=None, 
                    unknown_index="random",
                    sane_index_shape=False, 
                    legacy=True, 
                    sigma = 0.1,
                    straight_through=True,
                    kl_weight=5e-4, 
                    temp_init=1.0):
        super().__init__()


        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma
        self.device = device
        self.ignorezq = ignorezq
        self.disentangle = disentangle

        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.straight_through = straight_through

        self.proj = nn.Linear(self.e_dim, self.n_e, 1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        sphere = Hypersphere(dim=self.e_dim - 1)


        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True

        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


        self.hsreg = lambda x: [ torch.norm(x[i]) for i in range(x.shape[0])]
        self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(device)
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]
        

        # remap
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.clamp_class = Clamp()


    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z,
                    prev_cb = None,
                    attention_w = None, 
                    temp=None, 
                    rescale_logits=False, 
                    return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. 
        # actually, always true seems to work

        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        

        z_flattened = z.view(-1, self.e_dim)
        cb = self.embedding.weight


        # intra distance (gdes-distance) between codebook vector 
        d1 = torch.einsum('bd,dn->bn', 
                            cb, 
                            rearrange(cb, 'n d -> d n'))
        ed1 = torch.tensor(self.ed(cb))
        ed1 = ed1.repeat(self.n_e, 1)
        ed2 = ed1.transpose(0,1)
        ed3 = ed1 * ed2
        edx = d1/ed3.to(self.device)
        edx = torch.clamp(edx, min=-0.99999, max=0.99999)
        # d = torch.acos(d)
        d1 = torch.acos(edx)
        

        min_distance = torch.kthvalue(d1, 2, 0)
        total_min_distance = torch.mean(min_distance[0])
        codebookvariance = torch.mean(torch.var(d1, 1))

        # get quantized vector and normalize
        hsw = torch.Tensor(self.hsreg(cb)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw.clone().detach()))
        self.r = self.clamp_class.apply(self.r)
        disentanglement_loss = codebookvariance - total_min_distance


        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z_flattened)


        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros

        z_q = torch.einsum('b n, n d -> b d', soft_one_hot, cb)

        cb_loss = 0
        if not (prev_cb is None):
            cb_attn = torch.einsum('md,mn->nd', prev_cb, attention_w)
            z_q2 = torch.einsum('b n, n d -> b d', soft_one_hot, cb_attn)
            cb_loss = F.mse_loss(z_q, z_q2)


        z_q = z_q.view(z.shape)
        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        kl_loss = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_e + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)


        loss = kl_loss + cb_loss

        if self.disentangle:
           loss += hsw + disentanglement_loss

        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[ind] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return (z_q, loss,
                    (sampled_idx, ind), 
                    codebookvariance, 
                    total_min_distance,  
                    hsw, 
                    cb_loss,
                    torch.mean(self.r))


    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_e).permute(0, 3, 1, 2).float()
        z_q = torch.einsum('b n h w, n d -> b d h w', one_hot, self.embedding.weight)
        return z_q


class HierarchyVQmodulator(nn.Module):

    def __init__(self, *, 
                    features, 
                    z_channels, 
                    codebooksize, 
                    emb_dim,
                    device,
                    dropout=0.0, 
                    nclasses=10,
                    ignorezq=False,
                    gumble = False,
                    trim=False,
                    combine=True,
                    reasoning=True):
        super(HierarchyVQmodulator, self).__init__()

        self.ignorezq = ignorezq
        self.codebook_conditional = 0.1 if self.ignorezq else 0.8 

        #=====================================================================
        # modulator layers
        self.norm1 = nn.BatchNorm2d(features, affine=True)        
        self.conv1 = torch.nn.Conv2d(features,
                                       z_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1
                                       )
        self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                      z_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding =1,
                                      )

        #=====================================================================


        self.trim = trim
        self.reasoning = reasoning
        self.combine = combine
        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        self.z_channels = z_channels
        self.emb_dim = emb_dim 

        # ====================================================================
        # quantizer blocks
        self.quantizeBlocks = nn.ModuleList([])
        if not gumble:
            self.quantizeBlocks.append(VectorQuantizer2DHS(device, 
                                            codebooksize[0], 
                                            emb_dim, 
                                            beta=1.0, 
                                            disentangle=False,
                                            sigma=0.1))
            for cbsize in codebooksize[1:]:
                self.quantizeBlocks.append(VectorQuantizer2DHS(device, 
                                            cbsize, 
                                            emb_dim, 
                                            beta=1.0, 
                                            sigma=0.1, 
                                            disentangle=False,
                                            ignorezq = self.ignorezq))
        else:
            self.quantizeBlocks.append(GumbelQuantize2DHS(device, 
                                            codebooksize[0], 
                                            emb_dim, 
                                            beta=1.0, 
                                            disentangle=False,
                                            sigma=0.1))

            for cbsize in codebooksize[1:]:
                self.quantizeBlocks.append(GumbelQuantize2DHS(device, 
                                            cbsize, 
                                            emb_dim, 
                                            beta=1.0, 
                                            disentangle=False,
                                            sigma=0.1))

        # ====================================================================
        # attention layers
        if self.trim:
            zchannels = [(z_channels//(2**i), z_channels//(2**(i+1))) for i in range(len(codebooksize) - 1)]
        else:
            zchannels = [(z_channels, z_channels) for _ in range(len(codebooksize) - 1)]


        self.featureattns = nn.ModuleList([])
        self.hyperbolicLayers = nn.ModuleList([])
        for zch in zchannels:
            self.featureattns.append(BinaryLinear(zch[0],zch[1]))
            self.hyperbolicLayers.append(HNNLayer(emb_dim, emb_dim, 1, 0, True ))


        if self.trim:
            self.trimLayers = nn.ModuleList([])
            for zch in zchannels:
                self.trimLayers.append(torch.nn.Conv2d(zch[0],
                                                zch[1],
                                                kernel_size=1,
                                                stride=1,
                                                ))
            if self.combine:
                f_channels = 0
                for zch in zchannels:
                    f_channels += zch[0]

                self.combine_conv = torch.nn.Conv2d(f_channels,
                                            z_channels,
                                            kernel_size=1,
                                            stride=1,
                                            )
     

        wc = weightConstraint2()
        if self.reasoning:
            self.reasoningLayers = nn.ModuleList([])  
            for i in range(len(codebooksize) - 1):
                self.rattn = BinaryLinear(codebooksize[i],
                                            codebooksize[i+1],
                                        )
                self.rattn.apply(wc)
                self.reasoningLayers.append(self.rattn)   


        # ====================================================================
        # poincare layers for poincare projection
        self.poincareLayers = nn.ModuleList([])
        for _ in codebooksize:
            self.poincareLayers.append(HNNLayer(emb_dim, 3, 1, 0, True))

        
        print (len(self.poincareLayers), len(self.reasoningLayers), len(self.featureattns), len(self.hyperbolicLayers))


    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.epsilon)
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res
    
    def dist(self, u, v):
        sqdist = torch.sum(u ** 2, dim=1,  keepdim=True) + \
                 torch.sum(v ** 2, dim=1) - 2 * \
                 torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        squnorm = torch.sum(v ** 2, dim=1,   keepdim=True)
        sqvnorm = torch.sum(u ** 2, dim=1,  keepdim=True)
        l = torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        #x = 1 + 2 * sqdist / torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        x = 1 + 2 * sqdist/torch.t(l)
        y =  torch.clamp(x ** 2, min=1.0 + self.epsilon)
        #x = 1 + 2 * sqdist / torch.mm((1 - squnorm), (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(y - 1)
        return torch.log(x + z)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

    def attention(self, y, conv=None, h = None):
        # x = input_ + (w * x.view(-1, self.emb_dim)).view(x.shape)

        if not (conv is None):
            x = y.view(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])
            x = self.expmap0(x, 1)
            x = h(x)
            x = self.logmap0(x, 1)
            x = x.view(y.shape[0], y.shape[1], y.shape[2] * y.shape[3])
            x = rearrange(x, 'b c h -> b h c').contiguous()
            x = torch.einsum('bhc, cc-> bhc', x, conv.weight)
            x = rearrange(x, 'b h c -> b c h').contiguous()
            x = x.view(y.shape[0], y.shape[1], y.shape[2], y.shape[3])
            return x
        return y

    def other_parameters(self):
        for name, layer in self.named_parameters():
            if not ("reasoning" in name.lower()):
                yield layer

    def reasoning_parameters(self):
        for name, layer in self.named_parameters():
            if "reasoning" in name.lower():
                yield layer

    def reasoning_attn(self, cb, w):
        if self.reasoning_attn:
            return torch.einsum('md,mn->nd', cb, w.T)
        else:
            return cb

    def forward(self, x):
        x = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x)
        x1 = self.norm2(x1)
        x2 = self.conv2(x1)

        shape = x2.shape
        assert self.emb_dim == shape[2]*shape[3]

        attention_ws = []; cbattns = []; sampled_features = []; sampled_symbols = []
        qinput = x2
        total_loss = 0

        # logging parameters
        total_cv = 0; total_cbd = 0 ; total_r = 0; total_hrc = 0
        for i in range(len(self.quantizeBlocks)):
            if i == 0:
                z_q, loss, sampled_idx, ce, td, hrc, r, \
                    prevcb, attention_w, cb_attnp, zn = self.quantizeBlocks[i](qinput)
                cb_attnp = self.quantizeBlocks[0].embedding.weight.clone()
            else: 
                z_q, loss, sampled_idx, ce, td, hrc, r, \
                    prevcb, attention_w, cb_attnp, zn = self.quantizeBlocks[i](qinput,
                                                            cb_attnp,
                                                            self.reasoningLayers[i-1].weight.T)

            if i < (len(self.quantizeBlocks) -1):
                qinput = self.attention(z_q, self.featureattns[i], self.hyperbolicLayers[i])

            total_loss += loss
            total_cbd += td
            total_cv += ce
            total_r += r
            total_hrc += hrc 

            sampled_features.append(z_q)
            sampled_symbols.append(sampled_idx)
            attention_ws.append(attention_w)
            cbattns.append(cb_attnp)


        # poincare loss computation
        nr_sum = 0
        dr_sum = 0
        attentionList = []; projections = []
        for i in range(len(cbattns)):
            # may need to detach weights
            poincare_projection1 = self.to_poincare(self.poincareLayers[i](self.proj(\
                                    self.expmap0(cbattns[i],1),1)),1)
            
            projections.append(poincare_projection1)
            attention = 1
            if not (attention_ws[i] is None):
                attention = torch.where(attention_ws[i] > 0.5, 1, 0)
                attentionList.append(attention)

            for j in range(len(cbattns)):
                if i > j: continue
                if i == j: attention = 1

                # may need to detach weights
                poincare_projection2 = self.to_poincare(self.poincareLayers[j](self.proj(\
                                        self.expmap0(cbattns[j],1),1)),1)
                
                
                # print (i, j, poincare_projection1.shape, poincare_projection2.shape)
                # try: print(attention.shape)
                # except: pass
                wcoeff = 1
                dist = self.dist(poincare_projection2, poincare_projection1)
                if i ==j:
                    dr_sum += 4 * torch.sum(attention*dist)
                else:
                    nr_sum += wcoeff * torch.sum(attention*dist)
                    dr_sum += wcoeff * torch.sum((1-attention)*dist)

        p_loss = (nr_sum + 1)/(dr_sum + 1)
        # reasoning weights regularizations:
        # all_linear1_params = torch.cat([x.view(-1) for x in list(self.reasoning_parameters())])
        # l1_regularization = 1e-5*torch.norm(all_linear1_params, 1)
        # loss += l1_regularization

        if self.trim and self.combine:
            z_combine = torch.cat(sampled_features, dim=1)
            z_q = self.combine_conv(z_combine)
        else:
            z_q = sampled_features

        return (total_loss, p_loss,
                    z_q,  
                    sampled_symbols,
                    ce, td, hrc, r, attentionList, projections)


class modulator(nn.Module):

    def __init__(self, *,
                 features,
                 z_channels,
                 ):
        super(modulator, self).__init__()
        self.norm1 = nn.BatchNorm2d(features, affine=True)

        self.conv1 = torch.nn.Conv2d(features,
                                     z_channels,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     )

        self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                     z_channels,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     )
    def forward(self, x):
        x1 = self.norm1(x)
        # x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x2)

        return x2



class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.max_norm = 1e6
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)
    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)
    
    
    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.mobius_matvec(m = drop_weight,x=x, c =self.c)
        res = self.proj(mv, self.c)
        if self.use_bias:
            bias = self.proj_tan0(self.bias.view(1, -1), c = self.c)
            hyp_bias = self.expmap0(bias, c =self.c)
            hyp_bias = self.proj(hyp_bias, c =self.c)
            res = self.mobius_add(res, hyp_bias, c=self.c)
            res = self.proj(res, c =self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self,  c_in, c_out):
        super(HypAct, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act = nn.ReLU()
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
    
    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)
    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res
    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x
    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u
    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals


    def forward(self, x):
        xt = self.act(self.logmap0(x, c=self.c_in))
        xt = self.proj_tan0(xt, c=self.c_out)
        return self.proj(self.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self,  in_features, out_features, c, dropout, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear( in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct( c, c)

    def forward(self, x):
        h = self.linear(x)
        #h = self.hyp_act(h)
        return h