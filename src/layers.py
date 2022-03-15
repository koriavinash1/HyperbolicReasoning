import torch
import math
from torch import nn
import torch.nn.functional as F


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


class CodeBook(nn.Module):
    def __init__(self, nfeatures, nsampling):
        super(CodeBook).__init__()
        self.sampling = nsampling 
        self.nfeatures = nfeatures



    def forward(self, x):
        pass


    def sample(self, x):
        pass