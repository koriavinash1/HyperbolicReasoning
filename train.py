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


def train_from_folder(\
                      data_root='/vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv0/TI/data',
                      logs_root='logs',
                      name='MNIST2',
                      image_size=(32,32),
                      codebook_size = [64, 32, 10],
                      latent_dim=32,
                      in_channels_codebook = 64,
                      batch_size=50,
                      ch_mult=(1, 2, 4, 8),
                      nclasses=10,
                      avg_pool=False,
                      ch=32,
                      num_res_blocks = 1,

                    #   codebook_size = [128, 32, 3],
                    #   name='AFHQ',
                    #   data_root='/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq',
                    #   logs_root='../logs',
                    #   image_size=(128,128),
                    #   batch_size=15,
                    #   latent_dim=512,
                    #   in_channels_codebook = 1024,
                    #   ch_mult=(1, 2, 4, 8, 16, 32),
                    #   nclasses=3,
                    #   ch=32,
                    #   num_res_blocks = 1,
                    #   avg_pool=True,


                      nepochs=100,
                      learning_rate=2e-4,
                      decoder_learning_rate=2e-4,
                      num_workers=10,
                      sigma = 0.1,
                      seed=42,
                      log=True,
                      resamp_with_conv=True,
                      in_channels =3,
                      trim=False,
                      combine=False,
                      reasoning=True
                      ):

    model_args = dict(
        data_root=data_root,
        logs_root=logs_root,
        image_size=image_size,
        codebook_size = codebook_size,
        inchannels_codebook = in_channels_codebook,
        batch_size=batch_size,
        nepochs=nepochs,
        learning_rate=learning_rate,
        decoder_learning_rate = decoder_learning_rate,
        num_workers=num_workers,
        sigma= sigma,
        seed=seed,
        nclasses=nclasses,
        latent_dim=latent_dim,
        log=log,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        resamp_with_conv=resamp_with_conv,
        in_channels=in_channels,
        trim = trim,
        avg_pool=avg_pool,
        combine=combine,
        reasoning=reasoning
    )

    os.makedirs(os.path.join(logs_root, name), exist_ok=True)
    model_args['logs_root'] = os.path.join(logs_root, name)
    with open(os.path.join(logs_root, name, 'exp-config.json'), 'w') as f:
        json.dump(model_args, f, indent=4)

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
