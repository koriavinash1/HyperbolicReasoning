import torch
import math
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import geomstats.backend as gs
from geomstats.geometry.hypersphere import \
Hypersphere
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

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
        def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                     dropout=0.0, resamp_with_conv=True, in_channels,
                      z_channels, input, hiddendim):
            super(Decoder, self).__init__()
            self.ch = ch
            self.temb_ch = 0
            self.num_resolutions = len(ch_mult)
            block_in = ch * ch_mult[self.num_resolutions - 1]
            self.input = input
            self.inchannel = math.prod(self.input)
            self.hiddendim = hiddendim
            self.fc1 = nn.Linear(self.inchannel, self.hiddendim)
            self.fc2 = nn.Linear(self.hiddendim, self.inchannel)

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
            xf = torch.flatten(x, start_dim=1)
            xf1 = self.fc1(xf)
            xf2 = nonlinearity(xf1)
            xf3 = self.fc2(xf2)
            xf4 = nonlinearity(xf3)
            xrs = xf4.view(x.shape[0],self.input[0],  self.input[1], self.input[2])
            x1 = self.conv_in(xrs)
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
        self.quantize = VectorQuantizer2DHS(device, codebooksize, z_channels, beta=0.25, sigma = 0.1)
        #self.BottleneckMLP = BottleneckMLP(input = input, hiddendim = hiddendim)



    def forward (self, x):
        x1 = self.enc(x)
        z_q, loss, distances, info, zqf, ce, td,hrc = self.quantize(x1)



        return loss, z_q, zqf, ce, td, hrc

class VectorQuantizer2DHS(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self,device,  n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, sigma = 0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sigma = sigma
        self.device = device
        sphere = Hypersphere(dim=self.e_dim-1)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.n_e))
        self.embedding.weight.data.copy_(points_in_manifold).requires_grad=True
        #self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.hsreg = lambda x: [torch.square(1 - torch.norm(x[i])) for i in range(x.shape[0])]
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
        #d =  torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        #d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        #d = torch.acos(d)
        #d1 = torch.acos(d1)
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
        #hsreg = lambda x: [1 - torch.norm(x[i]) for i in range(x.shape[0])]
        hsw = torch.sum(torch.Tensor(self.hsreg(self.embedding.weight))).to(self.device)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2) + torch.sum(torch.Tensor(self.hsreg(self.embedding.weight))).to(self.device) - total_min_distance
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2) +torch.sum(torch.Tensor(self.hsreg(self.embedding.weight))).to(self.device)- total_min_distance

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

        return z_q, loss, distances, (perplexity, min_encodings, min_encoding_indices), z_flattened1,codebookvariance, total_min_distance,  hsw

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

class CodeBook(nn.Module):
    def __init__(self, nfeatures, nsampling, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super(CodeBook).__init__()
        self.e_dim = nsampling 
        self.n_e = nfeatures
        self.beta = beta
        self.legacy = legacy

        self.entropy = HLoss
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
            self.re_embed = self.n_e

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

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)


        # entropy regularization
        loss += self.entropy(z_q)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

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
