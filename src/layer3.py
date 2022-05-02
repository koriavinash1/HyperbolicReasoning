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
from manifolds import PoincareBall
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from mathutils import artanh, tanh, arcosh, cosh, sinh
import torch.nn.init as init
from reasoning import BinaryLinear
from torch.nn.modules.module import Module
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

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
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        #h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        #h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

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
        self.dropout = torch.nn.Dropout(dropout)
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
       # h = self.norm1(h)
       # h = nonlinearity(h)
        h = self.conv1(h)

        #h = self.norm2(h)
       # h = nonlinearity(h)
        #h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

        
class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super(StyleVectorizer).__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), nn.ReLU6()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class Encoder(nn.Module):
    def __init__(self, *, ch,  ch_mult=(1,2,4,8), num_res_blocks,
                 dropout=0.0, resamp_with_conv=True, in_channels,z_channels,
        ):
        super(Encoder, self).__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)


        in_ch_mult = (1,)+tuple(ch_mult)
        downl = []
        for i_level in range(self.num_resolutions):

            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                downl.append(ResnetBlock2D(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            if i_level != self.num_resolutions-1:
                downl.append(Downsample(block_in, resamp_with_conv))


        self.down = nn.Sequential(*downl)

        # middle
        self.midblock_1 = ResnetBlock2D(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        self.midblock_2 = ResnetBlock2D(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)



    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        x1 = self.conv_in(x)
        enc = self.down(x1)

        h1 = self.midblock_1(enc)
        h2 = self.midblock_2(h1)
        h3 = self.conv_out(self.norm_out(h2))


        return h3

class BottleneckMLP(nn.Module):
    def __init__(self, input, hiddendim):
        super(BottleneckMLP, self).__init__()
        self.inchannel = math.prod(input)
        self.hiddendim = hiddendim
        self.fc1 = nn.Linear(self.inchannel, self.hiddendim)


    def forward(self, x):
        x0 = torch.flatten(x, start_dim=1)
        print(x0.shape)
        x1 = self.fc1(x0)
        x2 = nonlinearity(x1)


        return  x2

class Decoder(nn.Module):
        def __init__(self, *, ch, 
                        out_ch, 
                        ch_mult=(1, 2, 4, 8), 
                        num_res_blocks,
                        dropout=0.0, 
                        resamp_with_conv=True, 
                        in_channels,
                        z_channels, 
                        input, 
                        hiddendim):
            super(Decoder, self).__init__()
            self.ch = ch
            self.temb_ch = 0
            self.num_resolutions = len(ch_mult)
            block_in = ch * ch_mult[self.num_resolutions - 1]
            self.input = input
            self.inchannel = np.prod(self.input)
            self.hiddendim = hiddendim
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
                for i_block in range(self.num_res_blocks + 1):
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
            # xf = torch.flatten(x, start_dim=1)
            # xf1 = self.fc1(xf)
            # xf2 = nonlinearity(xf1)
            # xf3 = self.fc2(xf2)
            # xf4 = nonlinearity(xf3)
            # xrs = xf4.view(x.shape[0],self.input[0],  self.input[1], self.input[2])
            x1 = self.conv_in(x)
            x2 = self.midblock_1(x1)
            x3 = self.midblock_1(x2)
            x4 = self.up(x3)
            x5 = self.norm_out(x4)
            x6 = self.conv_out(x5)



            return  x6

class DiscAE(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                     dropout=0.0, resamp_with_conv=True, sigma = 0.1, in_channels,
                      z_channels, input, hiddendim, codebooksize):
        super(DiscAE, self).__init__()
        self.enc = Encoder(ch = ch,  ch_mult=ch_mult, num_res_blocks = num_res_blocks,
                 dropout=dropout, resamp_with_conv=resamp_with_conv, in_channels = in_channels,z_channels = z_channels)
        self.quantize = VectorQuantizer2D(codebooksize, z_channels, beta=0.25, sigma = 0.1)
        #self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)

        self.dec = Decoder(ch = ch,  ch_mult=ch_mult, num_res_blocks = num_res_blocks,input = input, hiddendim = hiddendim,
                 dropout=dropout, out_ch = out_ch, resamp_with_conv=resamp_with_conv, in_channels = in_channels,z_channels = z_channels)

    def forward (self, x):
        x1 = self.enc(x)
        z_q, loss, distances, info, zqf, ce, td, hsr = self.quantize(x1)
        #x3 = self.BottleneckMLP(z_q)
        x4= self.dec(z_q)

        return x4, loss, z_q, self.dec, zqf, ce, td, hsr

class DiscClassifier(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                     dropout=0.0, resamp_with_conv=True, sigma = 0.1, in_channels,
                      z_channels, input, hiddendim, codebooksize, device):
        super(DiscClassifier, self).__init__()
        self.enc = Encoder(ch = ch,  ch_mult=ch_mult, num_res_blocks = num_res_blocks,
                 dropout=dropout, resamp_with_conv=resamp_with_conv, in_channels = in_channels,z_channels = z_channels)
        self.quantize = VectorQuantizer2DHS(device, codebooksize, z_channels, beta=1.0, sigma = 0.1)
        #self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)



    def forward (self, x):
        x1 = self.enc(x)
        z_q, loss, distances, info, zqf, ce, td,hrc, r = self.quantize(x1)



        return loss, z_q, zqf, ce, td, hrc, r


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
        # self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)

    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x1)
        z_q, loss, distances, info, zqf, ce, td, hrc, r = self.quantize(x2)

        return loss, z_q, zqf, ce, td, hrc, r


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.9, max=1.1) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class VectorQuantizer2DHSCW(nn.Module):
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
        sphere = Hypersphere(dim=16-1)
        #sphere = Hypersphere(self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, 16)
        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        #self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
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

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b c h w ').contiguous()
        #z_flattened = z.view(-1, 16)
        z = rearrange(z, 'b c h w -> b c h w').contiguous()
        z_flattened = z.view(-1, 16)
       # z_flattened = torch.nn.functional.normalize(z_flattened)
        z = z_flattened.view(z.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #d =  torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        

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
        #distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1])
        distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], self.e_dim)
        distances = self.rbf(distances)

        # get quantized vector and normalize
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw))


        perplexity = None
        min_encodings = None
        self.r = self.clamp_class.apply(self.r)
        z_flattened1 = z_q.view(z.shape[0],self.e_dim, z_q.shape[2]*z_q.shape[3])
        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_flattened1.detach() - z_flattened) ** 2) + \
                   torch.mean((z_flattened1 - z_flattened.detach()) ** 2) + hsw
        else:
            loss = torch.mean((z_flattened1.detach() - z_flattened) ** 2) + self.beta * \
                   torch.mean((z_flattened1 - z_flattened.detach()) ** 2) + hsw
        
        z_flattened1 = z_flattened + (z_flattened1 - z_flattened).detach()
        # compute loss for embedding
        """
                
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)# + hsw
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)# + hsw 
        

        #disentanglement_loss = codebookvariance - total_min_distance

        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + hsw - total_min_distance
        else:
            loss = self.beta * torch.mean((z_q - z.detach()) ** 2) +hsw - total_min_distance
        """ 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b c h w-> b c h w').contiguous()
        #z_flattened1 = z_q.view(z.shape[0],self.e_dim, z_q.shape[2]*z_q.shape[3])

        z_q = rearrange(z_q, 'b c h w -> b c h w').contiguous()

        #sampled_idx = torch.zeros(z.shape[0]*self.n_e)
        #sampled_idx[min_encoding_indices.detach()] = 1
        #sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), z_flattened1,codebookvariance, total_min_distance,  hsw, self.r, z_flattened1

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
        self.h1 =  HypLinear(e_dim, e_dim, 1, 0, True)        


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

    """    
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1
    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p
    """
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
        z_flattened = torch.nn.functional.normalize(z_flattened)
        #z = z_flattened.view(z.shape)
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
            

            # d1 = torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True) + \
            #      torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            #      torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
            
            min_distance = torch.kthvalue(d1, 2, 0)
            total_min_distance = torch.mean(min_distance[0])
            codebookvariance = torch.mean(torch.var(d1, 1))
            # codebookvariance = torch.var(min_distance[0])
 
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
            
            min_encoding_indices = torch.argmin(d, dim=1)
            # distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1])
            # distances = self.rbf(distances)

            zn= torch.nn.functional.normalize(self.embedding(min_encoding_indices)).view(z.shape)
            # get quantized vector and normalize
            z_q2 = self.embedding(min_encoding_indices).view(z.shape)
        loss = 0
        cb_loss = 0
        prevcb=  cb_attnp = cb_attn = None
                
        if not (prev_cb is None):
            #prevcb = self.to_poincare(self.h1(self.expmap0(torch.nn.functional.normalize(prev_cb.clone().detach()),1)),1)
            #prevcb = self.to_poincare(self.h1(self.expmap0(prev_cb.clone().detach(),1)),1)
            prevcb = prev_cb.clone().detach()
            cb_attn = torch.einsum('md,mn->nd', 
                                    self.logmap0(self.h(self.expmap0(prev_cb.clone().detach(),1)),1), 
                                    attention_w)
            cb_attnx = self.to_poincare(self.expmap0(cb_attn, 1), 1)
            zfl = self.to_poincare(self.expmap0(z_flattened,1), 1)
            dd = self.dist(zfl, cb_attnx)
            mei = torch.argmin(dd, dim =1)
            
            #d1 = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            #torch.sum(cb_attn**2, dim=1) - 2 * \
            #torch.einsum('bd,dn->bn', z_flattened, rearrange(cb_attn, 'n d -> d n'))
            #min_encoding_indices1 = torch.argmin(d1, dim=1)
            #cb_attnp = self.to_poincare(self.h1(self.expmap0(cb_attn, 1)), 1)
            #zn= torch.nn.functional.normalize(cb_attn[min_encoding_indices1]).view(z.shape)
            #z_q2 = cb_attn[min_encoding_indices1].view(z.shape)
            #zn = torch.nn.functional.normalize(cb_attn[min_encoding_indices1]).view(z.shape)
            z_q2 = self.logmap0(self.to_hyperboloid(cb_attnx[mei], 1),1).view(z.shape)
            zn =torch.nn.functional.normalize(self.logmap0(self.h1(self.to_hyperboloid(cb_attnx[mei], 1)),1)).view(z.shape)
            #cb_loss = F.mse_loss(z_q2, z_q2)
            #print(cb_loss)
            codebookvariance=total_min_distance=hsw = 0
            loss += cb_loss

        #  cb_loss = F.mse_loss(z_q, z_q2)
        #else:
         #   z_q2 = z_q
        """
        if not (prev_cb is None):
            cb_attn = torch.einsum('md,mn->nd',
                                    prev_cb.clone().detach(),
                                    attention_w)
            z_q2 = cb_attn[min_encoding_indices].view(z.shape)
            cb_loss = F.mse_loss(z_q, z_q2)
        """
        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw))


        perplexity = None
        min_encodings = None
        #self.r = self.clamp_class.apply(self.r)
        # compute loss for embedding
        if  (prev_cb is None):
            if not self.legacy:
                loss = self.beta * torch.mean((z_q2.detach() - z) ** 2) 
                if not self.ignorezq:
                    loss += torch.mean((z_q2 - z.detach()) ** 2) 
            else:
                loss = torch.mean((z_q2.detach() - z) ** 2)  
                if not self.ignorezq:
                    loss += self.beta * torch.mean((z_q2 - z.detach()) ** 2)

            #loss += cb_loss
            disentanglement_loss = codebookvariance - total_min_distance
            if self.disentangle:
                loss += hsw
                loss += disentanglement_loss

        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + hsw - total_min_distance
        else:
            loss = self.beta * torch.mean((z_q - z.detach()) ** 2) +hsw - total_min_distance
        """ 

        # preserve gradients
#        z_q = z + (z_q - z).detach()
        z_q2 = z + (z_q2 - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c-> b c h w').contiguous()

        # if self.remap is not None:
        #     min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
        #     min_encoding_indices = self.remap_to_used(min_encoding_indices)
        #     min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        # if self.sane_index_shape:
        #     min_encoding_indices = min_encoding_indices.reshape(
        #         z_q.shape[0], z_q.shape[2], z_q.shape[3])


        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
       # sampled_idx[min_encoding_indices] = 1
       # sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return (z_q2, loss,
                    sampled_idx, 
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

class VectorQuantizer2DHS1(nn.Module):
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
        #sphere = Hypersphere(dim=16-1)
        sphere = Hypersphere(self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.hsreg = lambda x: [ torch.norm(x[i]) for i in range(x.shape[0])]
        #self.r = torch.nn.Parameter(torch.ones(self.n_e)).to(device)
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

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def rbf(self, d):
        return (d ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

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

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b c h w ').contiguous()
        #z_flattened = z.view(-1, 16)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        z_flattened = torch.nn.functional.normalize(z_flattened)
        z = z_flattened.view(z.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #d =  torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        

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
        #distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1])
        distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1], z.shape[2])
        distances = self.rbf(distances)

        # get quantized vector and normalize
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(1 - hsw))


        perplexity = None
        min_encodings = None
        #self.r = self.clamp_class.apply(self.r)
        # compute loss for embedding
        z_flattened1 = z_q.view(z.shape[0], z_q.shape[1]*z_q.shape[2], self.e_dim) 
        
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2) + hsw
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2) + hsw 

        z_q = z + (z_q - z).detach()

        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_flattened1.detach() - z_flattened) ** 2) + \
                   torch.mean((z_flattened1 - z_flattened.detach()) ** 2) + hsw
        else:
            loss = torch.mean((z_flattened1.detach() - z_flattened) ** 2) + self.beta * \
                   torch.mean((z_flattened1 - z_flattened.detach()) ** 2) + hsw
        z_flattened1 = z_flattened + (z_flattened1 - z_flattened).detach()
        #disentanglement_loss = codebookvariance - total_min_distance
        """
        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + hsw - total_min_distance
        else:
            loss = self.beta * torch.mean((z_q - z.detach()) ** 2) +hsw - total_min_distance
        """ 

        # preserve gradients

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        #z_flattened1 = z_q.view(z.shape[0], z_q.shape[1]*z_q.shape[2], self.e_dim)
        #z_q = rearrange(z_q, 'b c h w -> b c h w').contiguous()
       # z_flattened1 = z_q.view(z.shape[0], z_q.shape[2]*z_q.shape[3], self.e_dim)

        #sampled_idx = torch.zeros(z.shape[0]*self.n_e)
        #sampled_idx[min_encoding_indices.detach()] = 1
        #sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), z_flattened1,codebookvariance, total_min_distance,  hsw, 1, z_flattened1

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

class HierarchyTFVQmodulatorCW(nn.Module):

    def __init__(self, *, 
                    features, 
                    dropout=0.0, 
                    z_channels, 
                    codebooksize, 
                    device,
                    nclasses=10,
                    num_heads = 4
                    ):
        super(HierarchyTFVQmodulatorCW, self).__init__()
        #self.norm1 = nn.BatchNorm2d(features, affine=True)
        self.nclasses = nclasses
        self.norm1 = Normalize(features)

        self.conv1 = torch.nn.Conv2d(features,
                                       z_channels,
                                       kernel_size=1,
                                       stride=1,
                                       )

        #self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        self.norm2 = Normalize(z_channels)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                      z_channels,
                                      kernel_size=1,
                                      stride=1,
                                      )

        #self.trim = trim
        
        self.quantize = VectorQuantizer2DHSCW(device, codebooksize[0], z_channels, beta=1.0, sigma=0.1)
       # self.q1w = torch.nn.Parameter(torch.ones(z_channels))
        self.t1 = TransformerBlock(16, 16*2, num_heads, dropout)     
        self.quantize1 = VectorQuantizer2DHSCW(device, codebooksize[1], z_channels, beta=1.0, sigma=0.1)
        #self.q2w = torch.nn.Parameter(torch.ones(z_channels))
        #self.t2 = TransformerBlock(16, 16*2, num_heads, dropout)
        #self.quantize2 = VectorQuantizer2DHSCW(device, codebooksize[2], z_channels, beta=1.0, sigma=0.1)
       # self.q3w = torch.nn.Parameter(torch.ones(z_channels))
        self.t3 = TransformerBlock(16, 16*2, num_heads, dropout)
        self.quantize3 = VectorQuantizer2DHSCW(device, nclasses, z_channels, beta=1.0, sigma=0.1)
        #self.q4w = torch.nn.Parameter(torch.ones(z_channels))
        self.clf = torch.nn.Linear(16, nclasses)
        """
        self.q1conv = None
        self.q2conv = None
        self.q3conv = None
        self.q4conv = None
        if self.trim:
            self.q1conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q2conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=2,
                                        )
            self.q3conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q4conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=2,
                                        )
        # self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)
        self.z_channels = z_channels
    def attention(self, input_,  w, x, conv=None):
        x = input_ + (w * x.view(-1, self.z_channels)).view(x.shape)
        if not (conv is None):
            x = conv(x)
        return x
    """
    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x1)
        
        z_q1, loss1, distances, info, zqf1, ce1, td1, hrc1, r1, i1 = self.quantize(x2)
        
        #z_q1 = self.attention(x2, self.q1w, z_q1, self.q1conv) 
        z_q1 = z_q1.view(zqf1.shape)
        z_q1 = self.t1(z_q1)
        z_q1 = z_q1.view(x2.shape)
        z_q2, loss2, distances, info, zqf2, ce2, td2, hrc2, r2, i2 = self.quantize1(z_q1)
        #z_q2 = self.attention(z_q1, self.q2w, z_q2, self.q2conv) 
        z_q2 = z_q2.view(zqf2.shape)
        z_q2 = self.t2(z_q2)
        z_q2 = z_q2.view(x2.shape)
        #z_q3, loss3, distances, info, zqf3, ce3, td3, hrc3, r3, i3 = self.quantize2(z_q2t)
        #z_q3 = self.attention(z_q2, self.q3w, z_q3, self.q3conv) 
        #zqf3 = self.t3(zqf3)
        #z_q3t = zqf3.view(z_q3.shape)

        z_q4, loss4, distances, info, zqf4, ce4, td4, hrc4, r4, i4 = self.quantize3(z_q2)
        #z_q4 = self.attention(z_q3, self.q4w, z_q4, self.q4conv)
        z_q4 = z_q4.view(zqf4.shape)
        z_q4t = self.clf(z_q4)
        #z_q4t = z_q4t.view(zqf3.shape[0],self.nclasses. zqf3.shape[2], zqf3.shape[3] )
        z_q4t = torch.mean(z_q4t, dim=1)
        loss = 0.25*(loss1 + loss2 + loss3 + loss4)
        # zqf = 0.25*(zqf1 + zqf2 + zqf3 + zqf4) 
        ce = 0.25*(ce1 + ce2 + ce3 + ce4) 
        td = 0.25*(td1 + td2 + td3 + td4) 
        hrc = 0.25*(hrc1 + hrc2 + hrc3 + hrc4) 
        r = torch.mean(r1)#0.25*(r1 + r2 + r3 + r4)

        return loss, z_q4, z_q1, ce, td, hrc, r, z_q4t, [i1, i2, i3, i4]


class HierarchyTFVQmodulator(nn.Module):

    def __init__(self, *, 
                    features, 
                    dropout=0.0, 
                    z_channels, 
                    codebooksize, 
                    device,
                    nclasses=10,
                    num_heads = 4
                    ):
        super(HierarchyTFVQmodulator, self).__init__()
        self.norm1 = nn.BatchNorm2d(features, affine=True)
        self.nclasses = nclasses
        #self.norm1 = Normalize(features)

        self.conv1 = torch.nn.Conv2d(features,
                                       z_channels,
                                       kernel_size=3,
                                       padding=1,
                                       stride=1,
                                       )

        self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        #self.norm2 = Normalize(z_channels)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                      z_channels,
                                      kernel_size=3,
                                      padding=1,
                                      stride=1,
                                      )
        self.num_layers = 5

        #self.trim = trim
        
        self.quantize = VectorQuantizer2DHS(device, codebooksize[0], z_channels, beta=1.0, sigma=0.1)
       # self.q1w = torch.nn.Parameter(torch.ones(z_channels))
        #self.t1 = TransformerBlock(z_channels, z_channels*2, num_heads, dropout)     
        #self.t11 = TransformerBlock(z_channels, z_channels*2, num_heads, dropout)
        #self.t1 = nn.ModuleList(
         #   [TransformerBlock(z_channels, z_channels*2, num_heads, dropout) for i in range(self.num_layers)]
        #)
        self.t1 = TransformerBlock(z_channels, z_channels*2, num_heads, dropout)
        self.quantize1 = VectorQuantizer2DHS(device, codebooksize[1], z_channels, beta=1.0, sigma=0.1)
        #self.q2w = torch.nn.Parameter(torch.ones(z_channels))
        self.t2 = TransformerBlock(z_channels, z_channels*2, num_heads, dropout)
        self.quantize2 = VectorQuantizer2DHS(device, codebooksize[2], z_channels, beta=1.0, sigma=0.1)
       # self.q3w = torch.nn.Parameter(torch.ones(z_channels))
        self.t3 = TransformerBlock(z_channels, z_channels*2, num_heads, dropout)
        self.quantize3 = VectorQuantizer2DHS(device, nclasses, z_channels, beta=1.0, sigma=0.1)
        #self.q4w = torch.nn.Parameter(torch.ones(z_channels))
        self.clf = torch.nn.Linear(z_channels, nclasses)
        """
        self.q1conv = None
        self.q2conv = None
        self.q3conv = None
        self.q4conv = None
        if self.trim:
            self.q1conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q2conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=2,
                                        )
            self.q3conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q4conv = torch.nn.Conv2d(z_channels,
                                        z_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=2,
                                        )
        # self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)
        self.z_channels = z_channels
    def attention(self, input_,  w, x, conv=None):
        x = input_ + (w * x.view(-1, self.z_channels)).view(x.shape)
        if not (conv is None):
            x = conv(x)
        return x
    """
    
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

    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x2)

        z_q1, loss1, distances, info, zqf1, ce1, td1, hrc1, r1, i1 = self.quantize(x2)
        #z_q1 = self.attention(x2, self.q1w, z_q1, self.q1conv) 
        z_q1 = rearrange(z_q1, 'b c h w-> b h w c').contiguous()
        z_q1 = z_q1.view(zqf1.shape)
        #z_q1 = self.t11(self.t1(z_q1))
        #for blk in self.t1:
        z_q1t = self.t1(z_q1)
        z_q1t = rearrange(z_q1t, 'b h c -> b c h').contiguous()
        z_q1t = z_q1t.view(x2.shape)
        z_q2, loss2, distances, info, zqf2, ce2, td2, hrc2, r2, i2 = self.quantize1(z_q1t)
        #z_q2 = self.attention(z_q1, self.q2w, z_q2, self.q2conv) 
        z_q2 = rearrange(z_q2, 'b c h w-> b h w c').contiguous()
        z_q2 = z_q2.view(zqf2.shape)
        z_q2t = self.t2(z_q2)
        z_q2t = rearrange(z_q2t, 'b h c -> b c h').contiguous()
        z_q2t = z_q2t.view(x2.shape)
        z_q3, loss3, distances, info, zqf3, ce3, td3, hrc3, r3, i3 = self.quantize2(z_q2t)
        #z_q3 = self.attention(z_q2, self.q3w, z_q3, self.q3conv) 
        z_q3 = rearrange(z_q3, 'b c h w-> b h w c').contiguous()
        z_q3 = z_q3.view(zqf3.shape)
        z_q3 = self.t3(z_q3)
        
        z_q3t = rearrange(z_q3, 'b h c -> b c h').contiguous()
        z_q3t = z_q3t.view(x2.shape)
        z_q4, loss4, distances, info, zqf4, ce4, td4, hrc4, r4, i4 = self.quantize3(z_q3t)
        #z_q4 = self.attention(z_q3, self.q4w, z_q4, self.q4conv)
        z_q4 = rearrange(z_q4, 'b c h w-> b h w c').contiguous()
        z_q4 = z_q4.view(zqf4.shape)
        
        #z_q4t = torch.squeeze(torch.mean(z_q3, dim=1))
        z_q4t = self.clf(z_q4)
        #z_q4t = z_q4t.view(zqf3.shape[0],self.nclasses. zqf3.shape[2], zqf3.shape[3] )
        z_q4t = torch.squeeze(torch.mean(z_q4t, dim=1))
        loss = 0.25*(loss1 + loss2 + loss3 + loss3)
        # zqf = 0.25*(zqf1 + zqf2 + zqf3 + zqf4) 
        ce = 0.25*(ce1 + ce2 + ce3 + ce3) 
        td = 0.25*(td1 + td2 + td3 + td3) 
        hrc = 0.25*(hrc1 + hrc2 + hrc3 + hrc3) 
        r = 1#torch.mean(r1)#0.25*(r1 + r2 + r3 + r4)

        return loss, z_q4t, z_q1t, ce, td, hrc, r, z_q4t, [i1, i2, i3, i3]

def tempsigmoid(x, k=3.0):
    nd = 1.0 
    temp = nd/torch.log(torch.tensor(k)) 
    return torch.sigmoid(x/temp)

class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w = torch.heaviside(w, torch.tensor([0.0]))
            x = torch.sum(w, 0)
            x = x.repeat(w.shape[0], 1)
            w = w/x
        
            module.weight.data=w

class weightConstraint1(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=tempsigmoid(w)
            module.weight.data=w

class weightConstraint2(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w = torch.heaviside(w, torch.tensor([0.0]))
            x = w.shape[0]
            module.weight.data=w

class weightConstraint3(object):
    def __init__(self):
        pass

    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            x = w.shape[1]
            module.weight.data=w/8



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

        # modulator layers
        self.norm1 = nn.BatchNorm2d(features, affine=True)
        #self.norm1 = Normalize(features)
        
        self.conv1 = torch.nn.Conv2d(features,
                                       z_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1
                                       )
        
        self.norm2 = nn.BatchNorm2d(z_channels, affine=True)
        #self.norm2 = Normalize(z_channels)
        #self.norm3 = nn.BatchNorm2d(z_channels, affine=True)
        #self.norm4 = nn.BatchNorm2d(z_channels, affine=True)
        self.conv2 = torch.nn.Conv2d(z_channels,
                                      z_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding =1,
                                      )
        """
        self.conv1 = ResnetBlock2D(in_channels=features,
                                           out_channels=z_channels,
                                           dropout=dropout)

        self.conv2 = ResnetBlock2D(in_channels=z_channels,
                                           out_channels=z_channels,
                                           dropout=dropout)
        
        """
        # ============

        self.trim = trim
        self.reasoning = reasoning
        self.combine = combine
        self.epsilon = 1e-4
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.min_norm = 1e-15
        self.hp1 =  HNNLayer(emb_dim, 3, 1, 0, True)
        self.hp2 =  HNNLayer(emb_dim, 3, 1, 0, True)
        self.hp3 =  HNNLayer(emb_dim, 3, 1, 0, True)
        if not gumble:
            self.quantize = VectorQuantizer2DHS(device, 
                                    codebooksize[0], 
                                    emb_dim, 
                                    beta=1.0, 
                                    disentangle=False,
                                    sigma=0.1)
            self.quantize1 = VectorQuantizer2DHS(device, 
                                    codebooksize[1], 
                                    emb_dim, 
                                    beta=1.0, 
                                    sigma=0.1, 
                                    disentangle=False,
                                    ignorezq = self.ignorezq)
            self.quantize2 = VectorQuantizer2DHS(device, 
                                    codebooksize[2], 
                                    emb_dim, 
                                    beta=1.0, 
                                    sigma=0.1,
                                    disentangle=False,
                                    ignorezq=self.ignorezq)
            self.quantize3 = VectorQuantizer2DHS(device, 
                                    codebooksize[3], 
                                    emb_dim, 
                                    beta=1.0, 
                                    sigma=0.1,
                                    disentangle=False,
                                    ignorezq=self.ignorezq)
        else:
            self.quantize = GumbelQuantize2DHS(device, 
                                    codebooksize[0], 
                                    emb_dim, 
                                    beta=1.0, 
                                    disentangle=False,
                                    sigma=0.1)
            self.quantize1 = GumbelQuantize2DHS(device, 
                                    codebooksize[1], 
                                    emb_dim, 
                                    beta=1.0, 
                                    disentangle=False,
                                    sigma=0.1)
            self.quantize2 = GumbelQuantize2DHS(device, 
                                    codebooksize[2], 
                                    emb_dim, 
                                    beta=1.0, 
                                    disentangle=False,
                                    sigma=0.1)
            self.quantize3 = GumbelQuantize2DHS(device, 
                                    codebooksize[3], 
                                    emb_dim, 
                                    beta=1.0, 
                                    disentangle=False,
                                    sigma=0.1)
        
        # self.q1w = torch.nn.Parameter(torch.ones(emb_dim))
        # self.q2w = torch.nn.Parameter(torch.ones(emb_dim))
        # self.q3w = torch.nn.Parameter(torch.ones(emb_dim))
        
        self.q1w = self.q2w = self.q3w = None
        """
        self.q1conv = torch.nn.Linear(z_channels,z_channels)
        self.q2conv = torch.nn.Linear(z_channels, z_channels)
        self.q3conv = torch.nn.Linear(z_channels, z_channels)
        """
        self.q1conv = nn.Linear(z_channels,z_channels)
        #self.q2conv = nn.Linear(z_channels, z_channels)
        self.q3conv = nn.Linear(z_channels, z_channels)
        
        self.h =  HNNLayer(emb_dim, emb_dim, 1, 0, True )
        self.h2 =  HNNLayer(emb_dim, emb_dim, 1, 0, True )
        #self.h3 =  HypLinear(emb_dim, emb_dim, 1, 0, True )
        wc = weightConstraint2()
        wc1 = weightConstraint3()
        #self.q1conv.apply(wc)
        #self.q2conv.apply(wc)
        #self.q3conv.apply(wc)
        if self.trim:
            self.q1conv = torch.nn.Conv2d(z_channels,
                                        z_channels//4,
                                        kernel_size=1,
                                        stride=1,
                                        )
            self.q2conv = torch.nn.Conv2d(z_channels//4,
                                        z_channels//8,
                                        kernel_size=1,
                                        stride=1,
                                        )
            self.q3conv = torch.nn.Conv2d(z_channels//8,
                                        1,
                                        kernel_size=1,
                                        stride=1,
                                        )

            if self.combine:
                f_channels = z_channels + z_channels//4 + z_channels//8 + 1
                self.combine_conv = torch.nn.Conv2d(f_channels,
                                            z_channels,
                                            kernel_size=1,
                                            stride=1,
                                            )
        """        
        if self.reasoning:
            self.rattn1 = HypLinear(codebooksize[0],
                                        codebooksize[1], 1, 0, True,
                                        )
            self.rattn2 = HypLinear(codebooksize[1],
                                        codebooksize[2], 1, 0, True,
                                        )
            self.rattn3 = HypLinear(codebooksize[2],
                                        codebooksize[3], 1, 0, True,
                                        )
            wc = weightConstraint1()
            self.rattn1.apply(wc)
            self.rattn2.apply(wc)
            self.rattn3.apply(wc)
        """
        
        if self.reasoning:
            #self.rattn1 = BinaryLinear(codebooksize[0],
             #                           codebooksize[1]
              #                          )
            self.rattn2 = BinaryLinear(codebooksize[0],
                                        codebooksize[2],
                                     )
            self.rattn3 = BinaryLinear(codebooksize[2],
                                        codebooksize[3],
                                        )
            self.rattn2.apply(wc)
            self.rattn3.apply(wc) 
        
        """
        if self.reasoning:
            self.rattn1 = torch.nn.Linear(codebooksize[0],
                                        codebooksize[1],
                                        )
            self.rattn2 = torch.nn.Linear(codebooksize[1],
                                        codebooksize[2],
                                        )
            self.rattn3 = torch.nn.Linear(codebooksize[2],
                                        codebooksize[3],
                                        )
            wc = weightConstraint()
            self.rattn1.apply(wc)
            self.rattn2.apply(wc)
            self.rattn3.apply(wc)
        """
        self.z_channels = z_channels
        self.emb_dim = emb_dim 
    def binary_parameters(self):
        for name, layer in self.named_parameters():
            if "binary" in name:
                yield layer

    def non_binary_parameters(self):
        for name, layer in self.named_parameters():
            if "bn" in name:
                yield layer
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
    def attention(self, input_,  w, y, conv=None, h = None):
        # x = input_ + (w * x.view(-1, self.emb_dim)).view(x.shape)

        if not (conv is None):
            x = y.view(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])
            x =self.expmap0(x, 1)
            x = h(x)
            x= self.logmap0(x, 1)
            x = x.view(y.shape[0], y.shape[1], y.shape[2] * y.shape[3])
            x = rearrange(x, 'b c h -> b h c').contiguous()
            x = torch.einsum('bhc, cc-> bhc', x, conv.weight)
            x = rearrange(x, 'b h c -> b c h').contiguous()
            #x = x.view(y.shape[0] * y.shape[1], y.shape[2] * y.shape[3])
            #x= self.logmap0(x, 1)
            x = x.view(y.shape[0], y.shape[1], y.shape[2], y.shape[3])
        else:
            x = y
        return x

    def other_parameters(self):
        for name, layer in self.named_parameters():
            if not ("rattn" in name.lower()):
                yield layer

    def reasoning_parameters(self):
        for name, layer in self.named_parameters():
            if "rattn" in name.lower():
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

        z_q1, loss1, sampled_idx1, ce1, td1, hrc1, r1, prevcb1, attention_w1, cb_attnp1, zn1 = self.quantize(x2)
        z_q1_attn = self.attention(x2, self.q1w, z_q1, self.q1conv, self.h)
        #z_q1_attn = self.attention(x2, self.q1w, z_q1, self.binaryc1, self.h)
        """
        z_q2, loss2, sampled_idx2, ce2, td2, hrc2, r2, prevcb2, attention_w2, cb_attnp2 = self.quantize1(z_q1_attn,
                                                                    self.quantize.embedding.weight,
                                                                    self.rattn1.weight.T)
                                                                    #self.binary1.weight.T)
        z_q2_attn = self.attention(z_q1_attn, self.q2w, z_q2, self.q2conv, self.h)
        """
        #z_q2_attn = self.attention(z_q1_attn, self.q2w, z_q2, self.binaryc2, self.h)
        z_q3, loss3, sampled_idx3, ce3, td3, hrc3, r3, prevcb3, attention_w3, cb_attnp3, zn2 = self.quantize2(z_q1_attn,
                                                                    self.quantize.embedding.weight,
                                                                    self.rattn2.weight.T)
                                                                    #self.binary2.weight.T)
        z_q3_attn = self.attention(z_q1_attn, self.q3w, z_q3, self.q3conv, self.h2)
        #z_q3_attn = self.attention(z_q2_attn, self.q3w, z_q3, self.binaryc3, self.h)


        z_q4, loss4, sampled_idx4, ce4, td4, hrc4, r4,  prevcb4, attention_w4, cb_attnp4, zn3= self.quantize3(z_q3_attn,
                                                                    cb_attnp3,
                                                                    self.rattn3.weight.T)
                                                                    #self.binary3.weight.T)

        # logs
        #attention2 = torch.where(attention_w2 > 0.5, 1,0)
        attention3 = torch.where(attention_w3 > 0.5, 1,0)
        attention4 = torch.where(attention_w4 > 0.5, 1,0)
        """
        prevcb3 = prevcb3[torch.abs(attention3).sum(dim=1) != 0]
        cb_attnp3p = cb_attnp3[torch.abs(attention3).sum(dim=0) != 0]
        #print(prevcb3.shape)
        #print(cb_attnp3p.shape)
        
        attention3 = attention3[torch.abs(attention3).sum(dim=1) != 0]
        attention3 = attention3[:,torch.abs(attention3).sum(dim=0) != 0]
        attention4 = torch.where(attention_w4 > 0.5, 1,0)
        e = (torch.abs(attention3).sum(dim=0) != 0).nonzero(as_tuple=False)
        f = (torch.abs(attention4).sum(dim=1) != 0).nonzero(as_tuple=False)
        g =torch.cat((e,f))
        g = torch.unique(g)
        cb_attnp3b = cb_attnp3[torch.abs(attention4).sum(dim=1) != 0]
        cb_attnp4 = cb_attnp4[torch.abs(attention4).sum(dim=0) != 0]
        attention4 = attention4[torch.abs(attention4).sum(dim=1) != 0]
        #attention4 = attention4[torch.abs(attention3).sum(dim=0) != 0]
        attention4 = attention4[:, torch.abs(attention4).sum(dim=0) != 0]
        """
        #attention2b = torch.where(attention_w2 > 0.5, 0,1)
        attention3b = torch.where(attention3 > 0.5, 0,1)
        #attention3b = attention3b[torch.abs(attention3b).sum(dim=1) != 0]
        #attention3b = attention3b[:,torch.abs(attention3b).sum(dim=0) != 0]
        attention4b = torch.where(attention4 > 0.5, 0,1)
        #attention4b = attention4b[torch.abs(attention4b).sum(dim=1) != 0]
        #attention4b = attention4b[:, torch.abs(attention4b).sum(dim=0) != 0]
        #pd1 = self.dist(prevcb2, cb_attnp2)
        #prevcb3 = self.hp1(self.to_poincare(self.expmap0(prevcb3,1),1))
        prevcb3 = self.to_poincare(self.hp1(self.proj(self.expmap0(prevcb3,1),1)),1)
        cb_attnp4 = self.to_poincare(self.hp3(self.proj(self.expmap0(cb_attnp4,1),1)),1)
        cb_attnp3 = self.to_poincare(self.hp2(self.proj(self.expmap0(cb_attnp3,1),1)),1)
        #cb_attnp3b = self.to_poincare(self.hp1(self.expmap0(cb_attnp3b,1)),1)
        #cb_attnp4 = self.hp3(self.to_poincare(self.expmap0(cb_attnp4,1),1))
        #cb_attnp3p = self.hp2(self.to_poincare(self.expmap0(cb_attnp3,1),1))
        #cb_attnp3b = self.hp2(self.to_poincare(self.expmap0(cb_attnp3b,1),1))


        pd2 = self.dist(prevcb3, cb_attnp3)
        pd3 = self.dist(cb_attnp3, cb_attnp4)
        #pd4 = self.dist(prevcb2, cb_attnp3)
        #pd5 = self.dist(prevcb2, cb_attnp4)
        pd6 = self.dist(prevcb3, cb_attnp4)
        pd1s = self.dist(prevcb3, prevcb3)
        pd2s = self.dist(cb_attnp3, cb_attnp3)
        #pd2sb = self.dist(cb_attnp3b, cb_attnp3b)
        pd3s = self.dist(cb_attnp4, cb_attnp4)
        #ploss = (torch.sum(attention2 * pd1) + torch.sum(attention3 * pd2) + torch.sum(attention4 * pd3))/ \
        #(torch.sum( (attention2b)*pd1) + torch.sum((attention3b)*pd2) + torch.sum((attention4b)*pd3) + torch.sum(pd4) + torch.sum(pd5) +torch.sum(pd6))
        ploss =  (torch.sum(attention3 * pd2) + torch.sum(attention4 * pd3))/ \
        (torch.sum((attention3b)*pd2) + torch.sum((attention4b)*pd3)  +torch.sum(pd6)+ torch.sum(pd1s) + torch.sum(pd2s)*4 + torch.sum(pd3s))
        #ploss = (torch.sum(attention3 * pd2)) / (torch.sum((attention3b)*pd2) + torch.sum(pd6)+ torch.sum(pd1s) + torch.sum(pd2s)) + \
         #       (torch.sum(attention4 * pd3))/ (torch.sum((attention4b)*pd3) + torch.sum(pd2s) + torch.sum(pd3s))

        #print(attention3)
        #(torch.sum( (1-attention2)*pd1) + torch.sum((1-attention3)*pd2) + torch.sum((1-attention4)*pd3) + torch.sum(pd4) + torch.sum(pd5) +torch.sum(pd6))
        #(torch.sum( (attention2b)*pd1) + torch.sum((attention3b)*pd2) + torch.sum((attention4b)*pd3) + torch.sum(pd4) + torch.sum(pd5) +torch.sum(pd6))
        #loss = 0.25*(loss1 + loss2 + loss3 + loss4) 
        #ce = 0.25*(ce1 + ce2 + ce3 + ce4) 
        #td = 0.25*(td1 + td2 + td3 + td4) 
        #hrc = 0.25*(hrc1 + hrc2 + hrc3 + hrc4) 
        #r = 0.25*(r1 + r2 + r3 + r4)
        loss = 0.33*(loss1 +  loss3 + loss4)
        ce = 0.33*(ce1 +  ce3 + ce4)
        td = 0.33*(td1 + td3 + td4)
        hrc = 0.33*(hrc1 +  hrc3 + hrc4)
        r = 0.33*(r1+ r3 + r4)
        # import pdb; pdb.set_trace()
        # print(list(self.reasoning_parameters()))

        # reasoning weights regularizations:
        # all_linear1_params = torch.cat([x.view(-1) for x in list(self.reasoning_parameters())])
        # l1_regularization = 1e-4*torch.norm(all_linear1_params, 1)
        # loss += l1_regularization

        if self.trim and self.combine:
            z_combine = torch.cat([z_q1, z_q3, z_q4], dim=1)
            z_q = self.combine_conv(z_combine)
        else:
            z_q = [z_q1,  z_q3, z_q4]
        return (loss, ploss,
                    [z_q1,  z_q3, z_q4],
                    z_q,  
                    [sampled_idx1,  sampled_idx3, sampled_idx4],
                    ce, td, hrc, r, [attention3, attention4], [prevcb3, cb_attnp3, cb_attnp4])

class VectorQuantizer2DHB(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, device, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, sigma = 0.1, init_weights=1e-3, epsilon=1e-5, node='parent', leaf = False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma
        self.node = node
        sphere = Hypersphere(dim=self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        #points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        #points_in_manifold = self.expmap0(points_in_manifold, 1)
        #self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        
        self.epsilon = epsilon
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.leaf = leaf
        self.min_norm = 1e-15
        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        points_in_manifold = self.expmap0(points_in_manifold, 1)
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True

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

    """
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1
    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p
    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    """
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

    def rbf(self, d):
        return (d ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

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
    """
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
        return torch.clamp(torch.log(x + z), max =2.5)
    
    def sdist(self, u, v):
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        y =  torch.clamp(x ** 2, min=1.0 + self.epsilon)
        z = torch.sqrt(y - 1)
        return torch.clamp(torch.log(x + z), max =2.5)
    """
    def dist(self, u, v):
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
    """
    def sdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(prod / K, min=1.0 + self.epsilon)
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)
    """

    def sdist(self, u, v):
        res = torch.sum(u * v, dim = 1)
        #sqdist = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        

        """
        u = torch.t(u[:,0].repeat(u.shape[1], 1))
        v = (v[:, 0].repeat(v.shape[1], 1))
        uv = torch.einsum('bd,dn->bn', u, v)
        theta = sqdist - 2* uv
        theta = torch.clamp(theta, min=1.0 + self.epsilon)
        return arcosh(theta) ** 2
        """
        theta = torch.clamp(res, min=1.0 + self.epsilon)
        return arcosh(theta) ** 2

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (torch.mm(x, torch.t(y))).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)
    """
    def sdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2
    """
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened =z.view(-1, self.e_dim)
        #z_flattened = torch.nn.functional.normalize(z_flattened)
        #if self.leaf == True:
        z_flattened = torch.nn.functional.normalize(z_flattened)

        z_flattened = self.proj(self.expmap0(z_flattened, 1), 1)
        
        #else:
         #  z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = self.dist(z_flattened, self.embedding.weight)
        if self.node =='parent':
            d1 = self.dist(self.embedding.weight, self.embedding.weight)
            min_distance = torch.kthvalue(d1, 2, 0)
            mean_min_distance = torch.mean(min_distance[0])
        else:
            mean_min_distance = 0

        min_encoding_indices = torch.argmin(d, dim=1)
        minenc = min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1], z.shape[2])
        #distances = self.rbf(distances)
        z_qf = self.embedding(min_encoding_indices)
        z_q = z_qf.view(z.shape)
        #if self.node == 'parent':
        z_qfl = self.logmap0(z_qf, 1)
        z_qfl =  z_qfl.view(z.shape[0],z.shape[1] * z.shape[2], z.shape[3] )
        z_ql = z_qfl.view(z.shape)
        #else:
           #z_qfl = z_qf
           #z_ql = z_qfl.view(z.shape)
         #  z_ql = None
        #print(z_ql)

        
        perplexity = None
        min_encodings = None
        # compute loss for embedding
                
        if self.leaf == True:
            if not self.legacy:
                loss = self.beta * torch.mean(self.sdist(z_qf.detach(), z_flattened)) + \
                     torch.mean(self.sdist(z_qf, z_flattened.detach()))# - mean_min_distance
            else:
                loss = torch.mean(self.sdist(z_qf.detach(), z_flattened)) + \
                     self.beta * torch.mean(self.sdist(z_qf, z_flattened.detach()))# - mean_min_distance
            z_qf = z_flattened + (z_qf - z_flattened).detach()

        else:
            loss = torch.mean(self.sdist(z_qf, z_flattened.detach()))
            #z_qf = z_flattened.detach() + z_qf - z_flattened.detach()
            z_qf = z_flattened + (z_qf - z_flattened).detach()
        
        """ 
        if not self.legacy:
                loss = self.beta * torch.mean(self.sdist(z_qf.detach(), z_flattened)) + \
                     torch.mean(self.sdist(z_qf, z_flattened.detach())) #- mean_min_distance
        else:
                loss = torch.mean(self.sdist(z_qf.detach(), z_flattened)) + \
                     self.beta * torch.mean(self.sdist(z_qf, z_flattened.detach())) #- mean_min_distance
        z_qf = z_flattened + (z_qf - z_flattened).detach()
        
        """
        """
        if self.leaf == True:
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                     torch.mean((z_qf - z.detach()) ** 2) 
            else:
                 loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                      torch.mean((z_q - z.detach()) ** 2)  
            z_q = z + (z_q - z).detach()
        else:
            if not self.legacy:
                loss = torch.mean((z_q - z.detach()) ** 2)
            else:
                loss = torch.mean((z_q - z.detach()) ** 2)
            z_q = z.detach() + z_q - z.detach()
        # preserve gradients
        #z_q = z + (z_q - z).detach()
        #z_qf = z_flattened + (z_qf - z_flattened).detach()
        # reshape back to match original input shape
        """
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        pred = z_ql.view(z_ql.shape[0], z_ql.shape[1]*z_ql.shape[2], z_ql.shape[3])  
        z_ql = rearrange(z_ql, 'b h w c -> b c h w').contiguous()
        #z_qf = z_qf.view(z.shape[0], z.shape[1] * z.shape[2], z.shape[3])
        #z_qf = rearrange(z_qf, 'b h c -> b c h').contiguous()



        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), mean_min_distance, z_ql, minenc, z_qfl, pred

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
class MLPBlock(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        #x = self.fn(self.linear1(x))
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class SABlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5


    def forward(self, x):
        q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
                raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
                raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x
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

    
    """
    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    def proj_tan0(self, u, c):
        return u
    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1
    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res
    """

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
    """
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p


    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u
    """


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

class HypLinearP(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_features, out_features, c, dropout, use_bias):
        super(HypLinearP, self).__init__()
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
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    def proj_tan0(self, u, c):
        return u
    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1
    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res
    """

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
    """
    
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
class HypActP(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self,  c_in, c_out):
        super(HypActP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act = nn.ReLU()
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
    """
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
    """
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1
    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p
    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    def proj_tan(self, u, p, c):
        return u
    def proj_tan0(self, u, c):
        return u
    


    def forward(self, x):
        xt = self.act(self.logmap0(x, c=self.c_in))
        xt = self.proj_tan0(xt, c=self.c_out)
        return self.proj(self.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HNNLayerP(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self,  in_features, out_features, c, dropout, use_bias):
        super(HNNLayerP, self).__init__()
        self.linear = HypLinear( in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct( c, c)

    def forward(self, x):
        h = self.linear(x)
        h = self.hyp_act(h)
        return h
class HierarchyVQhb(nn.Module):

    def __init__(self, *,
                 z_channels,
                 codebooksize,
                 device,
                 nclasses=10,
                 size = 16,
                 dropout = 0.0,
                 trim=False):
        super(HierarchyVQhb, self).__init__()
        self.device = device
        self.size = size
        self.min_norm = 1e-10

        self.trim = trim
        self.nclasses = nclasses

        self.quantize = VectorQuantizer2DHB(device, codebooksize, z_channels, beta=1.0, sigma=0.1, node='child', leaf = True)
        #self.t1 = HNNLayer(z_channels, z_channels, 1, 0, True)
        self.t1 = TransformerBlock(z_channels, z_channels*2, 1, dropout)
        #self.t1 = HNNLayer(self.size, self.size, 1, 0, True)

        self.quantize1 = VectorQuantizer2DHB(device, codebooksize // 4, z_channels, beta=1.0, sigma=0.1, node='child', leaf = False)
        #self.t2 = HNNLayer(z_channels, z_channels, 1, 0, True)
        self.t2 = TransformerBlock(z_channels, z_channels*2, 1, dropout)
        #self.t2 = HNNLayer(self.size, self.size, 1, 0, True)
        self.quantize2 = VectorQuantizer2DHB(device, codebooksize // 4, z_channels, beta=1.0, sigma=0.1, node='child', leaf = False)
        #self.t3 = HNNLayer(z_channels, z_channels, 1, 0, True)
        self.t3 = TransformerBlock(z_channels, z_channels*2, 1, dropout)
        #self.t3 = HNNLayer(self.size, self.size, 1, 0, True)
        self.quantize3 = VectorQuantizer2DHB(device, codebooksize // 16, z_channels, beta=1.0, sigma=0.1, node='parent', leaf = False)
        #self.t4 = HNNLayer(z_channels, z_channels, 1, 0, True)
        #self.t4 = HNNLayer(self.size, self.size, 1, 0, True)
        self.z_channels = z_channels

    def attention(self, input_, w, x, conv=None):
        x = input_ + (w * x.view(-1, self.z_channels)).view(x.shape)

        if not (conv is None):
            x = conv(x)
        return x

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p
    
    def forward(self, x):
        

        z_q1, loss1, distances1, info, td1, z_ql1, minenc1, z_qf1, p1 = self.quantize(x)
        #z_ql1 = self.attention(x, self.q1w, z_ql1, self.q1conv)
        z_qf1 = self.t1(z_qf1)
        z_q1r= z_qf1.view(z_q1.shape[0], z_q1.shape[2],z_q1.shape[3], z_q1.shape[1])
        z_q1r = rearrange(z_q1r, 'b h w c -> b c h w').contiguous()

        z_q2, loss2, distances2, info,  td2, z_ql2, minenc2, z_qf2, p2 = self.quantize1(z_q1r)
        #z_ql2 = self.attention(z_ql1, self.q2w, z_ql2, self.q2conv)
        z_qf2 = self.t2(z_qf2)
        z_q2r= z_qf2.view(z_q1.shape[0], z_q1.shape[2],z_q1.shape[3], z_q1.shape[1])
        z_q2r = rearrange(z_q2r, 'b h w c -> b c h w').contiguous()
        """
        z_q3, loss3, distances3, info,  td3, z_ql3, minenc3, z_qf3, p3 = self.quantize2(z_q2r)
        #z_ql3 = self.attention(z_ql2, self.q3w, z_ql3, self.q3conv)
        z_qf3 = self.t3(z_qf3)
        z_q3r= z_qf3.view(z_q1.shape[0], z_q1.shape[2],z_q1.shape[3], z_q1.shape[1])
        z_q3r = rearrange(z_q3r, 'b h w c -> b c h w').contiguous()
        """
        z_q4, loss4, distances4, info,  td4, z_ql4, minenc4, z_qf4, p4 = self.quantize3(z_q2r)
        #z_ql4 = self.attention(z_ql3, self.q4w, z_ql4, self.q4conv)
        z_qf4 = z_qf4.view(z_q4.shape[0], z_q4.shape[2] * z_q4.shape[3], z_q4.shape[1])
        loss = 0.33 * (loss1 + loss2 +  loss4)
        # zqf = 0.25*(zqf1 + zqf2 + zqf3 + zqf4)
        #td = 0.25 * (td1 + td2 + td3 + td4)
       # z_qf4 = self.clf(z_qf4)
        #z_qf4 = self.logmap0(z_qf4, 1)
        totaldistance = distances1+distances2+distances4
        """
        m = torch.flatten(minenc4, start_dim=1)
        pred = F.one_hot(torch.mode(m, dim = 1)[0], num_classes=self.nclasses)
        output = torch.zeros(self.nclasses, minenc4.shape[0], minenc4.shape[1],minenc4.shape[2]).to(self.device)
        l = torch.ones(1, minenc4.shape[0], minenc4.shape[1],minenc4.shape[2]).to(self.device)
        m = torch.unsqueeze(minenc4, dim=0)
        output = output.scatter_(0, m, l)
        output = rearrange(output, 'n d x y -> d n x y')
        """

        return loss, z_q4, z_ql4, td4, totaldistance, z_qf4 #pred, output
class VectorQuantizer2D(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, sigma = 0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

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

    def rbf(self, d):
        return (d ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

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

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        d1 = torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True) + \
             torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
             torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        min_distance = torch.kthvalue(d1, 2, 0)

        total_min_distance = torch.sum(min_distance[0])
        codebookvariance = torch.std(min_distance[0])
        min_encoding_indices = torch.argmin(d, dim=1)
        distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1], z.shape[2])
        distances = self.rbf(distances)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2) - total_min_distance
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2) - total_min_distance

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_flattened1 = z_q.view(z.shape[0],z.shape[1]*z.shape[2], self.e_dim)


        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), z_flattened1, codebookvariance, total_min_distance

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
