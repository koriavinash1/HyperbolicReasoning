import os
import fire
import random
import json
from tqdm import tqdm
from src.train import Trainer

from clsmodel import mnist, stl10, afhq
import torch
import torch.distributed as dist

import numpy as np


def train_from_folder(data_root = './data',
    logs_root ='./logs/',
    name = 'default',
    image_size = 128,
    style_depth = 16,
    batch_size = 5,
    nepochs = 150000,
    learning_rate = 2e-4,
    num_workers =  None,
    save_every = 1000,
    aug_prob = 0.,
    recon_loss_weightage = 1.0,
    disentangle_weightage = 1.0,
    quantization_weightage = 1.0,
    hessian_weightage = 1.0,
    pl_weightage = 1.0,
    seed = 42,
    nclasses=10,
    latent_dim=512,
    featurelen=1024,
    encoder=False,
    log = False,
):
    model_args = dict(
        data_root =data_root,
        logs_root =logs_root,
        name = name,
        image_size = image_size,
        style_depth = style_depth,
        batch_size = batch_size,
        nepochs = nepochs,
        learning_rate = learning_rate,
        num_workers =  num_workers,
        save_every = save_every,
        aug_prob = aug_prob,
        recon_loss_weightage = recon_loss_weightage,
        disentangle_weightage = disentangle_weightage,
        quantization_weightage = quantization_weightage,
        hessian_weightage = hessian_weightage,
        pl_weightage = pl_weightage,
        seed = seed,
        nclasses = nclasses,
        latent_dim = latent_dim,
        featurelen = featurelen,
        encoder = encoder,
        log = log,
    )

    os.makedirs(os.path.join(logs_root, name), exist_ok=True)
    with open(os.path.join(logs_root, name, 'exp-config.json'), 'w') as f:
        json.dump(model_args, f, indent = 4)
    

    if data_root.lower().__contains__('mnist'):
        if data_root.__contains__('TSWIv2'):
            net = mnist(32, 'tswiv2')
        elif data_root.__contains__('TSWI'):
            net = mnist(32, 'tswi')
        elif data_root.__contains__('TS'):
            net = mnist(32, 'ts')
        elif data_root.__contains__('TI'):
            net = mnist(32, 'ti')
        elif data_root.__contains__('IT'):
            net = mnist(32, 'it')
        else:
            raise ValueError()
    elif data_root.lower().__contains__('stl'):
        net = stl10(3, True)
    elif data_root.lower().__contains__('afhq'):
        net = afhq(3, True)
    else:
        raise ValueError("explainee model not found")
        
    model_args['classifier'] = net

    trainer = Trainer(**model_args)
    trainer.train()



if __name__ == '__main__':
    fire.Fire(train_from_folder)
