import torch
import random
from random import random
from torch import nn
import numpy as np
from contextlib import contextmanager, ExitStack
import torch.nn.functional as F

try:
    from torch.cuda import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model