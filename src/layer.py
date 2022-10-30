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

from src.mathutils import artanh, tanh, arcosh, cosh, sinh
from src.hyplinear import HNNLayer
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
import torch.nn.init as init

from src.reasoning import BinaryLinear
from torch.nn.modules.module import Module

def nonlinearity(x):
    return F.relu6(x)


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

            return  torch.sigmoid(x6)




class GeometricVQ(nn.Module):
    """
    This implements Vector Quantisation for Symbol formation and the formation of 
    hyperbolic codebooks for sampling in hyperbolic space. We assume a constant negative curvature of -1
    """
    def __init__(self,
                    device,  
                    n_e = 128, 
                    e_dim = 16, 
                    beta = 0.9, 
                    legacy=True, 
                    sigma = 0.1,
                    quantise = 'feature'):

        super().__init__()
        '''
        n_e : total number of codebook vectors
        e_dim: codebook vector dimension
        beta: factor for legacy term
        '''
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.sigma = sigma
        self.device = device
        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        self.quantise = quantise
        self.legacy = legacy
        # uniformly sampled initialization on hypersphere
        sphere = Hypersphere(dim=self.e_dim - 1)
        self.embedding = nn.Embedding(self.n_e, 
                                            self.e_dim)
        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]
        
        # Hyperboloid linear layer applied to each codebook vector before learning weighted edges between features
        self.h =  HNNLayer(e_dim, e_dim, 0, True)



    # Radial Basis function for distance uncertainty
    def rbf(self, d):
        return (d ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()
    
    #=====================================================================
    # Hyperbolic Projections. The following functions are adapted from https://github.com/HazyResearch/hgcn 
    # https://arxiv.org/abs/1910.12933 (Hyperbolic Convolutional Neural Networks)
    
    # Poincare Projection contraint
    def projp(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) 
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    # Hyperboloid Projection Contraint
    
    def proj(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(1 + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x
    
    #Hyperboloid exponential map at tangent at origin
    def expmap0(self, u):

        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm
        res = torch.ones_like(u)
        res[:, 0:1] = cosh(theta)
        res[:, 1:] = sinh(theta) * x / x_norm
        return self.proj(res)


    #Hyperboloid logarithmic  map at tangent at origin
    def logmap0(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1], min=1.0 + self.epsilon)
        res[:, 1:] = arcosh(theta) * y / y_norm
        return res
    
    # Mapping from the hyperboloid to poincare
    def to_poincare(self, x):
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x[:, 0:1] + 1)

    # Mapping from Poincare to hyperboloid
    def to_hyperboloid(self, x):
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return torch.cat([1 + sqnorm, 2 * x], dim=1) / (1 - sqnorm)


    #=====================================================================
    # Hyperbolic distances. The following functions are adapted from https://github.com/HazyResearch/hgcn
    # https://arxiv.org/abs/1910.12933 (Hyperbolic Convolutional Neural Networks)
    
    # Poincare distance
    def distp(self, u, v):
        sqdist = torch.sum(u ** 2, dim=1,  keepdim=True) + \
                 torch.sum(v ** 2, dim=1) - 2 * \
                 torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        squnorm = torch.sum(v ** 2, dim=1,   keepdim=True)
        sqvnorm = torch.sum(u ** 2, dim=1,  keepdim=True)
        l = torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        x = 1 + 2 * sqdist/torch.t(l)
        y =  torch.clamp(x ** 2, min=1.0 + self.epsilon)
        z = torch.sqrt(y - 1)
        return torch.log(x + z)


    # Hyperboloid distance
    def disth(self, u, v):       
        u = torch.t(u[:,0].repeat(u.shape[1], 1))
        v = (v[:, 0].repeat(v.shape[1], 1))
        uv = torch.einsum('bd,dn->bn', u, v)
        theta = sqdist - 2* uv
        theta = torch.clamp(theta, min=1.0 + self.epsilon)
        return arcosh(theta) ** 2
    
    #=====================================================================

    # Geodesic distance on hypersphere
    def geodist(self, u, v):
        d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        ed1 = torch.tensor(self.ed(self.embedding.weight)).repeat(self.n_e, 1)
        ed2 = ed1.transpose(0,1)
        geod = torch.clamp(d1/(ed1*ed2), min=-0.99999, max=0.99999)

        return torch.acos(geod)
    
    # Euclidean distance
    def dist(self, u, v):
         d = torch.sum(u ** 2, dim=1, keepdim=True) + \
             torch.sum(v**2, dim=1) - 2 * \
             torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))

         return d
    def forward(self, z, 
                    prev_cb = None,
                    attention_w = None,
                    si = None,
                    ):
        if self.quantise == 'spatial':
           # reshape z  and flatten
            z = rearrange(z, 'b c h w -> b h w c').contiguous() 
        z_flattened = z.view(-1, self.e_dim)


        loss = 0
        prevcb = cb_attn = None
                
        if (prev_cb is None):       
            # reshape z -> (batch, height, width, channel) and flatten
            
           
           z_flattened = z.view(-1, self.e_dim)
           # intra distance (gdes-distance) between codebook vector on Euclidean codebook on a hypersphere
           
           # compute mean codebook distances
           cd = self.dist(self.embedding.weight, self.embedding.weight)
           min_distance = torch.kthvalue(cd, 2, 0)
           mean_cb_distance = torch.mean(min_distance[0])
            
           # compute mean codebook variance
           mean_cb_variance = torch.mean(torch.var(cd, 1)) 
           # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
           d = self.dist(z_flattened, self.embedding.weight)
           min_encoding_indices = torch.argmin(d, dim=1)
           z_q = self.embedding(min_encoding_indices).view(z.shape)
           # get quantized vector and normalize
           zn = torch.nn.functional.normalize(z_q).view(z.shape) 
        
        else:
            #attention_w = (attention_w - torch.min(attention_w))/(torch.max(attention_w) - torch.min(attention_w))
            
            cb_attn = torch.einsum('md,mn->nd', 
                                    self.logmap0(self.h(self.expmap0(prev_cb.clone().detach()))), 
                                    attention_w)
            # Map codebook and sample(z_flattened) to Poincare space
            cb_attnx = self.to_poincare(self.expmap0(cb_attn))
            zfl = self.to_poincare(self.expmap0(z_flattened))
            #distance matrix
            dd = self.distp(zfl, cb_attnx)
            """
            dd = torch.ones(si.shape[0], z.shape[1], self.n_e, device = self.device)
            # binary attention masking
            bin_att = torch.where(attention_w !=  0, 1, 0)
            # reshape sample
            zfl2 = zfl.view(si.shape[0], z.shape[1], zfl.shape[1])
            # Sample subset of hyperbolic codebook which contains connected edges with all symbol sampled from previous codebook
            for v in range(si.shape[0]):
                f = torch.ones( z.shape[1],self.n_e,  device = self.device)
                samplesind = torch.squeeze(si[v,:])
                samples = bin_att[samplesind]
                samples = torch.squeeze(torch.sum(samples, dim =0))
                if torch.sum(samples)  ==  0:
                   sampled_i = torch.squeeze(torch.nonzero(samples > -1))
                   samplesf = cb_attnx[sampled_i]
                elif torch.squeeze(torch.nonzero(samples > 0)).dim()  > 0:
                   sampled_i = torch.squeeze(torch.nonzero(samples > 0))
                   samplesf = cb_attnx[sampled_i]
                else:
                   sampled_i = torch.nonzero(samples > 0)
                   samplesf = torch.squeeze(cb_attnx[sampled_i], dim = 0)
                newd= self.distp(zfl2[v],  samplesf)
                f[:,torch.squeeze(sampled_i, dim =0)] = newd
                dd[v]=f
            dd = dd.view(zfl.shape[0], self.n_e)
            """
            """
            # Sample subset of hyperbolic codebook which contains connected edges with each symbol sampled from previous codebook
            i_flattened = si.view(-1, 1)
            for v, w in enumerate(i_flattened):
                f = torch.ones(self.n_e, device = self.device)
                #f.to(self.device)
                f[f>0]= float('inf')
                sampled_i = torch.squeeze(bin_att[w,:])
                if torch.sum(sampled_i)  ==  0:
                   sampled_i = torch.squeeze(torch.nonzero(sampled_i > -1))
                elif torch.sum(sampled_i)  > 1:
                   sampled_i = torch.squeeze(torch.nonzero(sampled_i > 0))
                else:
                   sampled_i = torch.squeeze(torch.nonzero(sampled_i > 0), dim = 0)
                #elif torch.squeeze(torch.nonzero(sampled_i > 0).shape[0]) == 0:

                newd = self.dist(torch.unsqueeze(zfl[v], dim =0), cb_attnx[sampled_i])# if len(sampled_i) > 1 else torch.squeeze(cb_attnx[sampled_i], dim =0))
                f.scatter_(0,sampled_i,torch.squeeze(newd))
                dd[v]=f#.scatter_(0,torch.unsqueeze(sampled_i, dim = 0),torch.squeeze(newd))
             """


            #Indices sampled from codebook reshaped into z (si)
            min_encoding_indices = torch.argmin(dd, dim =1)
            
            z_q = self.logmap0(self.to_hyperboloid(cb_attnx[min_encoding_indices])).view(z.shape)
            zn =torch.nn.functional.normalize(self.logmap0(self.to_hyperboloid(cb_attnx[min_encoding_indices]))).view(z.shape)

            d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
            
            # compute mean hyperbolic codebook distances and variance 
            min_distance = torch.kthvalue(d1, 2, 0)
            mean_cb_distance = torch.mean(min_distance[0])
            mean_cb_variance = torch.mean(torch.var(d1, 1))




        si = min_encoding_indices.view(z.shape[0], z.shape[1])
        # compute loss for embedding
        if  (prev_cb is None):
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
                loss += torch.mean((z_q - z.detach()) ** 2) 
            else:
                loss = torch.mean((z_q.detach() - z) ** 2)  
                loss += self.beta * torch.mean((z_q - z.detach()) ** 2)
        #else:
        #    if not self.legacy:
        #        loss += torch.mean((z_q - z.detach()) ** 2)
        #    else:
        #        loss += self.beta * torch.mean((z_q - z.detach()) ** 2)


        # preserve gradients
        z_q = z + (z_q - z).detach()


        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return (z_q, loss,
                    (sampled_idx, min_encoding_indices.view(z.shape[0], -1)), 
                    mean_cb_variance, 
                    mean_cb_distance,   
                    prevcb, attention_w,  cb_attn, si, zn)
                    


        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class weightConstraint(object):
    # Contrain weights between 0 and 1
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w = torch.heaviside(w, torch.tensor([0.0]))
            x = w.shape[0]
            module.weight.data=w



class HierarchyVQmodulator(nn.Module):
    """
    This class creates a hierarchy of codebook embedded in Poincare space with weighted edges connecting the embeddings between consecutive codebooks.
    Edges determined by a binary weights
    Feature modulation applied before inputting into Euclidean codebook
    """
    def __init__(self, *, 
                    features, 
                    z_channels, 
                    codebooksize, 
                    emb_dim,
                    device,
                    dropout=0.0, 
                    nclasses=10,
                    trim = False,
                    quantise = 'feature',
                    reasoning=True):
        super(HierarchyVQmodulator, self).__init__()


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
        self.quantise = quantise
        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        self.z_channels = z_channels
        self.emb_dim = emb_dim 

        # ====================================================================
        # quantizer blocks
        self.quantizeBlocks = nn.ModuleList([])
        self.quantizeBlocks.append(GeometricVQ(device, 
                                            codebooksize[0], 
                                            emb_dim, 
                                            beta=1.0,
                                            quantise = self.quantise, 
                                            sigma=0.1))
        for cbsize in codebooksize[1:]:
                self.quantizeBlocks.append(GeometricVQ(device, 
                                            cbsize, 
                                            emb_dim, 
                                            beta=1.0,
                                            quantise = self.quantise,
                                            sigma=0.1, 
                                            ))

        # ====================================================================
        # Feature size at each abstraction level
        if self.trim:
            zchannels = [(z_channels//(2**i), z_channels//(2**(i+1))) for i in range(len(codebooksize) - 1)]
        else:
            zchannels = [(z_channels, z_channels) for _ in range(len(codebooksize) - 1)]


        self.featureattns = nn.ModuleList([])
        self.hyperbolicLayers = nn.ModuleList([])
        # Feature attention layers
        for zch in zchannels:
            self.featureattns.append(torch.nn.Linear(zch[0],zch[1]))
            self.hyperbolicLayers.append(HNNLayer(emb_dim, emb_dim,  0, True ))

        # Reduce feature size
        if self.trim:
            self.trimLayers = nn.ModuleList([])
            for zch in zchannels:
                self.trimLayers.append(torch.nn.Conv2d(zch[0],
                                                zch[1],
                                                kernel_size=1,
                                                stride=1,
                                                ))
     

        wc = weightConstraint()
        # Binary attention weights between Hyperbolic codebooks
        if self.reasoning:
            self.reasoningLayers = nn.ModuleList([])
            self.Aggregation = nn.ModuleList([])
            for i in range(len(codebooksize) - 1):

                self.rattn = BinaryLinear(codebooksize[i],
                                            codebooksize[i+1],
                                        )
                self.r = torch.nn.Linear(codebooksize[i],
                                        codebooksize[i+1],
                                        )
                self.rattn.apply(wc)
                self.reasoningLayers.append(self.rattn)   
                self.Aggregation.append(self.r)

        # ====================================================================
        # poincare layers for poincare projection
        self.poincareLayers = nn.ModuleList([])
        for _ in codebooksize:
            self.poincareLayers.append(HNNLayer(emb_dim, 3, 0, True))

        
        print (len(self.poincareLayers), len(self.reasoningLayers), len(self.featureattns), len(self.hyperbolicLayers))

    
    #=====================================================================
    # Hyperboloid Projections. The following functions are adapted from https://github.com/HazyResearch/hgcn
    # https://arxiv.org/abs/1910.12933 (Hyperbolic Convolutional Neural Networks)


    # Hyperboloid Projection Contraint

    def proj(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(1 + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    #Hyperboloid exponential map at tangent at origin
    def expmap0(self, u):

        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm
        res = torch.ones_like(u)
        res[:, 0:1] = cosh(theta)
        res[:, 1:] = sinh(theta) * x / x_norm
        return self.proj(res)


    #Hyperboloid logarithmic  map at tangent at origin
    def logmap0(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1], min=1.0 + self.epsilon)
        res[:, 1:] = arcosh(theta) * y / y_norm
        return res

    # Mapping from the hyperboloid to poincare
    def to_poincare(self, x):
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x[:, 0:1] + 1)
    
    #=====================================================================
    # Poincare distance. The following function is adapted from https://github.com/HazyResearch/hgcn
    # https://arxiv.org/abs/1910.12933 (Hyperbolic Convolutional Neural Networks)

    def dist(self, u, v):
        sqdist = torch.sum(u ** 2, dim=1,  keepdim=True) + \
                 torch.sum(v ** 2, dim=1) - 2 * \
                 torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        squnorm = torch.sum(v ** 2, dim=1,   keepdim=True)
        sqvnorm = torch.sum(u ** 2, dim=1,  keepdim=True)
        l = torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        x = 1 + 2 * sqdist/torch.t(l)
        y =  torch.clamp(x ** 2, min=1.0 + self.epsilon)
        z = torch.sqrt(y - 1)
        return torch.log(x + z)
    # Feature attention function
    def attention(self, y, conv=None, h = None):

        if not (conv is None):
            x = y.view(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])
            x = self.expmap0(x)
            x = h(x)
            x = self.logmap0(x)
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
        x1 = self.conv1(x)
        x1 = self.norm2(x1)
        x2 = self.conv2(x1)

        shape = x2.shape

        attention_ws = []; cbattns = []; sampled_features = []; sampled_symbols = []
        qinput = x2
        total_loss = 0

        # Foward pass for knowledge tree generation
        for i in range(len(self.quantizeBlocks)):
            if i == 0:
                z_q, loss, sampled_idx, cvE, cdE, \
                    prevcb, attention_w, cb_attnp,si, zn = self.quantizeBlocks[i](qinput)
                cb_attnp = self.quantizeBlocks[0].embedding.weight.clone()
            else: 
                z_q, loss, sampled_idx, cvH, cdH, \
                    prevcb, attention_w, cb_attnp, si, zn = self.quantizeBlocks[i](qinput,
                                                            cb_attnp,
                                                            (self.reasoningLayers[i-1].weight*self.Aggregation[i-1].weight).T, si)

            if i < (len(self.quantizeBlocks) -1):
                qinput = self.attention(z_q, self.featureattns[i], self.hyperbolicLayers[i])

            total_loss += loss

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
                                    self.expmap0(cbattns[i]))))
            
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
                                        self.expmap0(cbattns[j]))))
                
                
                wcoeff = 1
                dist = self.dist(poincare_projection2, poincare_projection1)
                if i ==j:
                    dr_sum += 4 * torch.sum(attention*dist)
                else:
                    nr_sum += wcoeff * torch.sum(attention*dist)
                    dr_sum += wcoeff * torch.sum((1-attention)*dist)

        p_loss = (nr_sum + 1)/(dr_sum + 1)

        z_q = sampled_features

        return (total_loss, p_loss,
                    z_q,  
                    sampled_symbols,
                    cvE, cdE, attentionList, projections)





