import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import geomstats.backend as gs
from geomstats.geometry.hypersphere import \
Hypersphere
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from mathutils import artanh, tanh
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)

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
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
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
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
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



class HierarchyVQmodulator(nn.Module):

    def __init__(self, *, 
                    features, 
                    z_channels, 
                    codebooksize, 
                    emb_dim,
                    device,
                    dropout=0.0, 
                    nclasses=10,
                    trim=True):
        super(HierarchyVQmodulator, self).__init__()
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

        self.trim = trim
        self.quantize = VectorQuantizer2DHS(device, codebooksize[0], emb_dim, beta=1.0, sigma=0.1)

        self.quantize1 = VectorQuantizer2DHS(device, codebooksize[1], emb_dim, beta=1.0, sigma=0.1)

        self.quantize2 = VectorQuantizer2DHS(device, codebooksize[2], emb_dim, beta=1.0, sigma=0.1)

        self.quantize3 = VectorQuantizer2DHS(device, codebooksize[3], emb_dim, beta=1.0, sigma=0.1)
        
        
        self.q1w = torch.nn.Parameter(torch.ones(emb_dim))
        self.q2w = torch.nn.Parameter(torch.ones(emb_dim))
        self.q3w = torch.nn.Parameter(torch.ones(emb_dim))
        

        self.q1conv = None
        self.q2conv = None
        self.q3conv = None
        if self.trim:
            self.q1conv = torch.nn.Conv2d(z_channels,
                                        z_channels//4,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q2conv = torch.nn.Conv2d(z_channels//4,
                                        z_channels//8,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )
            self.q3conv = torch.nn.Conv2d(z_channels//8,
                                        1,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        )

        # self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)

        self.z_channels = z_channels
        self.emb_dim = emb_dim 

    def attention(self, input_,  w, x, conv=None):
        print (input_.shape, x.shape, w.shape)
        x = input_ + (w * x.view(-1, self.emb_dim)).view(x.shape)

        if not (conv is None):
            x = conv(x)
        return x

    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x1)

        shape = x2.shape
        assert self.emb_dim == shape[2]*shape[3]

        z_q1, loss1, sampled_idx1, ce1, td1, hrc1, r1 = self.quantize(x2)
        z_q1_attn = self.attention(x2, self.q1w, z_q1, self.q1conv)

        z_q2, loss2, sampled_idx2, ce2, td2, hrc2, r2 = self.quantize1(z_q1_attn)
        z_q2_attn = self.attention(z_q1_attn, self.q2w, z_q2, self.q2conv)

        z_q3, loss3, sampled_idx3, ce3, td3, hrc3, r3 = self.quantize2(z_q2_attn)
        z_q3_attn = self.attention(z_q2_attn, self.q3w, z_q3, self.q3conv)

        z_q4, loss4, sampled_idx4, ce4, td4, hrc4, r4 = self.quantize3(z_q3_attn)

        loss = 0.25*(loss1 + loss2 + loss3 + loss4)
        ce = 0.25*(ce1 + ce2 + ce3 + ce4) 
        td = 0.25*(td1 + td2 + td3 + td4) 
        hrc = 0.25*(hrc1 + hrc2 + hrc3 + hrc4) 
        r = 0.25*(r1 + r2 + r3 + r4)

        return (loss, 
                    [z_q1, z_q2, z_q3, z_q4],  
                    [sampled_idx1, sampled_idx2, sampled_idx3, sampled_idx4],
                    ce, td, hrc, r)


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.9, max=1.1) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


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
<<<<<<< HEAD:src/layer2.py
        #sphere = Hypersphere(dim=16-1)
        #self.embedding = nn.Embedding(self.n_e, 16)
        #sphere = Hypersphere(self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        #points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        #self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
=======


        # uniformly sampled initialization
        sphere = Hypersphere(dim=self.e_dim - 1)
        self.embedding = nn.Embedding(self.n_e, 
                                            self.e_dim)


        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True

        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py
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
<<<<<<< HEAD:src/layer2.py
        #z = rearrange(z, 'b c h w -> b c h w ').contiguous()
        #z_flattened = z.view(-1, 16)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #d =  torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
=======
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py
        

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
        
<<<<<<< HEAD:src/layer2.py
=======

        # d1 = torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True) + \
        #      torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        #      torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py
        
        min_distance = torch.kthvalue(d1, 2, 0)
        total_min_distance = torch.mean(min_distance[0])
        codebookvariance = torch.mean(torch.var(d1, 1))
        # codebookvariance = torch.var(min_distance[0])


        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        
        min_encoding_indices = torch.argmin(d, dim=1)
<<<<<<< HEAD:src/layer2.py
        #distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1])
        distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1], z.shape[2])
        distances = self.rbf(distances)
=======
        # distances = d[range(d.shape[0]), min_encoding_indices].view(z.shape[0], z.shape[1])
        # distances = self.rbf(distances)

>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py

        # get quantized vector and normalize
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        hsw = torch.Tensor(self.hsreg(self.embedding.weight)).to(self.device)
        hsw = torch.mean(torch.square(self.r - hsw))


        perplexity = None
        min_encodings = None
        self.r = self.clamp_class.apply(self.r)
        # compute loss for embedding
         
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2) + hsw
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2) + hsw 


        disentanglement_loss = codebookvariance - total_min_distance

        """
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + hsw - total_min_distance
        else:
            loss = self.beta * torch.mean((z_q - z.detach()) ** 2) +hsw - total_min_distance
        """ 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
<<<<<<< HEAD:src/layer2.py
        #z_q = rearrange(z_q, 'b c h w-> b c h w').contiguous()
        #z_flattened1 = z_q.view(z.shape[0],self.e_dim, z_q.shape[2]*z_q.shape[3])

        z_q = rearrange(z_q, 'b h w c-> b c h w').contiguous()
        z_flattened1 = z_q.view(z.shape[0], z_q.shape[2]*z_q.shape[3], self.e_dim)


        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), z_flattened1,codebookvariance, total_min_distance,  hsw, self.r

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


class HierarchyVQmodulator(nn.Module):

    def __init__(self, *, 
                    features, 
                    dropout=0.0, 
                    z_channels, 
                    codebooksize, 
                    device,
                    nclasses=10,
                    trim=True):
        super(HierarchyVQmodulator, self).__init__()
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

        self.trim = trim
        
        self.quantize = VectorQuantizer2DHS(device, codebooksize, z_channels, beta=1.0, sigma=0.1)
        self.q1w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize1 = VectorQuantizer2DHS(device, codebooksize//2, z_channels, beta=1.0, sigma=0.1)
        self.q2w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize2 = VectorQuantizer2DHS(device, codebooksize//4, z_channels, beta=1.0, sigma=0.1)
        self.q3w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize3 = VectorQuantizer2DHS(device, nclasses, z_channels, beta=1.0, sigma=0.1)
        self.q4w = torch.nn.Parameter(torch.ones(z_channels))
        

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

    def forward(self, x):
        x1 = self.norm1(x)
        #x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x1)

        z_q1, loss1, distances, info, zqf1, ce1, td1, hrc1, r1 = self.quantize(x2)
        z_q1 = self.attention(x2, self.q1w, z_q1, self.q1conv) 

        z_q2, loss2, distances, info, zqf2, ce2, td2, hrc2, r2 = self.quantize1(z_q1)
        z_q2 = self.attention(z_q1, self.q2w, z_q2, self.q2conv) 
        
        z_q3, loss3, distances, info, zqf3, ce3, td3, hrc3, r3 = self.quantize2(z_q2)
        z_q3 = self.attention(z_q2, self.q3w, z_q3, self.q3conv) 
        
        z_q4, loss4, distances, info, zqf4, ce4, td4, hrc4, r4 = self.quantize3(z_q3)
        z_q4 = self.attention(z_q3, self.q4w, z_q4, self.q4conv)

        loss = 0.25*(loss1 + loss2 + loss3 + loss4)
        # zqf = 0.25*(zqf1 + zqf2 + zqf3 + zqf4) 
        ce = 0.25*(ce1 + ce2 + ce3 + ce4) 
        td = 0.25*(td1 + td2 + td3 + td4) 
        hrc = 0.25*(hrc1 + hrc2 + hrc3 + hrc4) 
        r = torch.mean(r1)#0.25*(r1 + r2 + r3 + r4)

        return loss, z_q4, z_q1, ce, td, hrc, r

class VectorQuantizer2DHB(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, device, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, sigma = 0.1, init_weights=1e-3, epsilon=1e-3, node='parent', leaf = False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma
        self.node = node
        #sphere = Hypersphere(dim=self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        #points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        #self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        self.epsilon = epsilon
        self.leaf = leaf
        self.min_norm = 1e-15

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

    def dist(self, u, v):
        sqdist = torch.sum(u ** 2, dim=1,  keepdim=True) + \
                 torch.sum(v ** 2, dim=1) - 2 * \
                 torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        squnorm = torch.sum(v ** 2, dim=1)
        sqvnorm = torch.sum(u ** 2, dim=1,  keepdim=True)
        #x = 1 + 2 * sqdist / torch.einsum('bd,dn->bn', 1 - squnorm, rearrange(1 - sqvnorm, 'n d -> d n')) + self.epsilon
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2)
        return x #torch.log(x + z)


    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        #if self.leaf == True:
        z_flattened = self.expmap0(z.view(-1, self.e_dim), 1)
        #else:
        #z_flattened = z.view(-1, self.e_dim)
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
        #if self.leaf == True:
        z_qf = self.logmap0(z_qf, 1, 2)
        z_ql = z_qf.view(z.shape)
        #else:

         #  z_ql = None



        perplexity = None
        min_encodings = None

        # compute loss for embedding
        """
        if self.leaf == True:
            if not self.legacy:
                loss = self.beta * torch.mean(self.dist(z_qf.detach(), z_flattened)) + \
                     torch.mean(self.dist(z_qf, z_flattened.detach())) - mean_min_distance
            else:
                loss = torch.mean(self.dist(z_qf.detach(), z_flattened)) + \
                     self.beta * torch.mean(self.dist(z_qf, z_flattened.detach())) - mean_min_distance
            z_qf = z_flattened + (z_qf - z_flattened).detach()

        else:
            loss = torch.mean(self.dist(z_qf, z_flattened.detach()))
            z_qf = z_flattened.detach() + z_qf - z_flattened.detach()
        """

        if not self.legacy:
                loss = self.beta * torch.mean(self.dist(z_qf.detach(), z_flattened)) + \
                     torch.mean(self.dist(z_qf, z_flattened.detach())) - mean_min_distance
        else:
                loss = torch.mean(self.dist(z_qf.detach(), z_flattened)) + \
                     self.beta * torch.mean(self.dist(z_qf, z_flattened.detach())) - mean_min_distance
        z_qf = z_flattened + (z_qf - z_flattened).detach()
        """
        
        if self.leaf == True:
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                     torch.mean((z_q - z.detach()) ** 2) 
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
        z_ql = rearrange(z_ql, 'b h w c -> b c h w').contiguous()        


=======
        # z_q = rearrange(z_q, 'b h w c-> b c h w').contiguous()
>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py


        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

<<<<<<< HEAD:src/layer2.py
        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), mean_min_distance, z_ql, minenc
=======


        sampled_idx = torch.zeros(z.shape[0]*self.n_e)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return (z_q, loss + disentanglement_loss,
                    sampled_idx, 
                    codebookvariance, 
                    total_min_distance,  
                    hsw, 
                    torch.mean(self.r))
                    
>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py

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

<<<<<<< HEAD:src/layer2.py
class modulator(nn.Module):

    def __init__(self, *,
                 features,
                 z_channels,
                 ):
        super(modulator, self).__init__()
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
    def forward(self, x):
        x1 = self.norm1(x)
        # x1 = nonlinearity(x1)
        x1 = self.conv1(x1)
        x2 = self.norm2(x1)
        x2 = self.conv2(x2)

        return x2
class HierarchyVQhb(nn.Module):

    def __init__(self, *,
                 z_channels,
                 codebooksize,
                 device,
                 nclasses=10,
                 trim=True):
        super(HierarchyVQhb, self).__init__()
        self.device = device

        self.trim = trim
        self.nclasses = nclasses

        self.quantize = VectorQuantizer2DHB(device, codebooksize, z_channels, beta=1.0, sigma=0.1, node='child', leaf = True)
        self.q1w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize1 = VectorQuantizer2DHB(device, codebooksize // 2, z_channels, beta=1.0, sigma=0.1, node='child', leaf = False)
        self.q2w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize2 = VectorQuantizer2DHB(device, codebooksize // 4, z_channels, beta=1.0, sigma=0.1, node='child', leaf = False)
        self.q3w = torch.nn.Parameter(torch.ones(z_channels))

        self.quantize3 = VectorQuantizer2DHB(device, nclasses, z_channels, beta=1.0, sigma=0.1, node='parent', leaf = False)
        self.q4w = torch.nn.Parameter(torch.ones(z_channels))
        
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
                                          stride=1,
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
                                          stride=1,
                                          )

        # self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)

        self.z_channels = z_channels

    def attention(self, input_, w, x, conv=None):
        x = input_ + (w * x.view(-1, self.z_channels)).view(x.shape)

        if not (conv is None):
            x = conv(x)
        return x
    
    def forward(self, x):
   

        z_q1, loss1, distances1, info, td1, z_ql1, minenc1 = self.quantize(x)
        z_ql1 = self.attention(x, self.q1w, z_ql1, self.q1conv)

        z_q2, loss2, distances2, info,  td2, z_ql2, minenc2 = self.quantize1(z_ql1)
        z_ql2 = self.attention(z_ql1, self.q2w, z_ql2, self.q2conv)

        z_q3, loss3, distances3, info,  td3, z_ql3, minenc3 = self.quantize2(z_ql2)
        z_ql3 = self.attention(z_ql2, self.q3w, z_ql3, self.q3conv)

        z_q4, loss4, distances4, info,  td4, z_ql4, minenc4 = self.quantize3(z_ql3)
        z_ql4 = self.attention(z_ql3, self.q4w, z_ql4, self.q4conv)

        loss = 0.25 * (loss1 + loss2 + loss3 + loss4)
        # zqf = 0.25*(zqf1 + zqf2 + zqf3 + zqf4)
        #td = 0.25 * (td1 + td2 + td3 + td4)
        totaldistance = distances1+distances2+distances3+distances4
        m = torch.flatten(minenc4, start_dim=1)
        pred = F.one_hot(torch.mode(m, dim = 1)[0], num_classes=self.nclasses)
        output = torch.zeros(self.nclasses, minenc4.shape[0], minenc4.shape[1],minenc4.shape[2]).to(self.device)

        l = torch.ones(1, minenc4.shape[0], minenc4.shape[1],minenc4.shape[2]).to(self.device)
        m = torch.unsqueeze(minenc4, dim=0)
        output = output.scatter_(0, m, l)
        output = rearrange(output, 'n d x y -> d n x y')


        return loss, z_q4, z_ql4, td4, totaldistance, pred, output
=======


>>>>>>> bec013bcdba6b505127ee2ae50f899cfcad6950b:src/layers.py
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