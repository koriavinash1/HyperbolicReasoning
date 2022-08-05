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

from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
import torch.nn.init as init

from src.reasoning import BinaryLinear
from torch.nn.modules.module import Module

#  The following functions are adapted from https://github.com/HazyResearch/hgcn
# https://arxiv.org/abs/1910.12933 (Hyperbolic Convolutional Neural Networks)

class HypLinear(nn.Module):
    """
    Hyperboloid linear layer.
    """

    def __init__(self, in_features, out_features, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

    def proj(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x):
        
        normu = self.minkowski_norm(u)
        theta = torch.clamp(normu, max=self.max_norm)
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result)

    def logmap(self, x, y):
        xy = torch.clamp(self.minkowski_dot(x, y), max=-self.eps[x.dtype]) 
        u = y + xy * x
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x)
    # Parallel Transport from origin of hyperboloid
    def ptransp0(self, x, u):
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] =  x0 * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) 
        res = u - alpha * v
        return self.proj_tan(res, x)

    def expmap0(self, u):
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm
        res = torch.ones_like(u)
        res[:, 0:1] = cosh(theta)
        res[:, 1:] =  sinh(theta) * x / x_norm
        return self.proj(res)

    def logmap0(self, x):
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1], min=1.0 + self.eps[x.dtype])
        res[:, 1:] = arcosh(theta) * y / y_norm
        return res

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def mobius_add(self, x, y):
        u = self.logmap0(y)
        v = self.ptransp0(x, u)
        return self.expmap(v, x)

    def mobius_matvec(self, m, x):
        u = self.logmap0(x)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu)


    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.mobius_matvec(m = drop_weight,x=x)
        res = self.proj(mv)
        if self.use_bias:
            bias = self.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.expmap0(bias)
            hyp_bias = self.proj(hyp_bias)
            res = self.mobius_add(res, hyp_bias)
            res = self.proj(res)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features
        )



class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self,  in_features, out_features, dropout, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear( in_features, out_features, dropout, use_bias)

    def forward(self, x):
        h = self.linear(x)
        return h

