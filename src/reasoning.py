from typing import TypeVar, Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from collections import OrderedDict


################################################################################

# taken from https://github.com/pytorch/pytorch/blob/bfeff1eb8f90aa1ff7e4f6bafe9945ad409e2d97/torch/nn/common_types.pyi

T = TypeVar("T")
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

################################################################################
# Quantizers


class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        output = inp.sign()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        clipped = inp.abs() <= Binarize.clip_value

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output


binarize = Binarize.apply

################################################################################
# Optimizers for binary networks

class MomentumWithThresholdBinaryOptimizer(Optimizer):
    def __init__(
        self,
        binary_params,
        bn_params,
        ar: float = 0.0001,
        threshold: float = 0,
        adam_lr=0.001,
    ):
        if not 0 < ar < 1:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )
        if threshold < 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        self.total_weights = {}
        self._adam = Adam(bn_params, lr=adam_lr)

        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(MomentumWithThresholdBinaryOptimizer, self).__init__(
            binary_params, defaults
        )

    def step(self, closure: Optional[Callable[[], float]] = ..., ar=None):
        self._adam.step()

        flips = {None}

        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]
            flips = {}

            if ar is not None:
                y = ar

            for param_idx, p in enumerate(params):
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_((1 - y))
                    m.add_(grad.mul(y))

                mask = (m.abs() >= t) * (m.sign() == p.sign())
                mask = mask.double() * -1
                mask[mask == 0] = 1

                flips[param_idx] = (mask == -1).sum().item()

                p.data.mul_(mask)

        return flips

    def zero_grad(self) -> None:
        super().zero_grad()
        self._adam.zero_grad()


################################################################################
# binary torch layers


class BinaryLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)

        return f.linear(inp, weight, bias)


class BinaryConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride=1,
        padding=1,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)

        return f.conv2d(
            inp, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

class Reasoning(nn.Module):
    def __init__(self, 
            layers= [],
            ar = 1e-4,
            lr = 1e-3,
            threshold = 1e-5):
        super(Reasoning, self).__init__()

        self.ar = ar
        self.lr = lr 
        self.threshold = threshold
        self.layers = nn.ModuleList([]) 
        for il in range(len(layers) - 1):
            block1 = []
            binary = BinaryLinear(layers[il], 
                                            layers[il +1], 
                                            bias=True, 
                                            binarize_input=True)
            
            block1.append(("binary{}".format(il), binary))
            if il < len(layers) - 2:
                bn = nn.BatchNorm1d(layers[il +1])
                block1.append(("bn{}".format(il), bn))

            self.layers.append(nn.Sequential(OrderedDict(block1)))

        
        self.opt = MomentumWithThresholdBinaryOptimizer(
                    self.binary_parameters(),
                    self.non_binary_parameters(),
                    ar=self.ar,
                    threshold=self.threshold,
                    adam_lr=self.lr,
                )



    def binary_parameters(self):
        for name, layer in self.named_parameters():
            if "binary" in name.lower():
                yield layer

    def non_binary_parameters(self):
        for name, layer in self.named_parameters():
            if "bn" in name.lower():
                yield layer

    def train_step(self, x, y):
        self.opt.zero_grad()
        be_loss = 0
        ce_loss = 0

        x = [xi.to(y.device) for xi in x]
        input_ = x[0]
        for i, block in enumerate(self.layers[:-1]):
            input_ = block(input_)
            be_loss += self.bce_loss(input_, x[i +1])

        y_ = self.layers[-1](input_)
        ce_loss = self.ce_loss(y_, y)


        loss = ce_loss + be_loss

        loss.backward()
        self.opt.step() 
        return loss.detach()


    def ce_loss(self, p, y):
        return f.cross_entropy(p, y)

    def bce_loss(self, p, y):
        return f.binary_cross_entropy_with_logits(p, y)

    @torch.no_grad()
    def val_step(self, x, y):
        be_loss = 0
        ce_loss = 0
        
        x = [xi.to(y.device) for xi in x]

        input_ = x[0]
        for i, block in enumerate(self.layers[::-1]):
            input_ = block(input_)
            be_loss += self.bce_loss(input_, x[i +1])

        y_ = self.layers[-1](input_)
        ce_loss = self.ce_loss(y_, y)


        loss = ce_loss + be_loss
        return 0.2*loss
 
