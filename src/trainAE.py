from click import style
import torch
import os
from torch import nn
import fire
import json
from layers import DiscAE, DiscClassifier, Decoder
from clsmodel import mnist, stl10, afhq
from torch.optim import Adam
import numpy as np
from dataset import get
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable
import math
from loss import hpenalty, calc_pl_lengths, recon_loss


class Trainer():
    def __init__(self,
                    codebook_length = 128,
                    sampling_size = 128,
                    name = 'default',
                    data_root = './data',
                    logs_root ='./logs/',
                    image_size = (64,64),
                    style_depth = 16,
                    batch_size = 10,
                    nepochs = 20,
                    sigma=0.1,
                    learning_rate = 2e-4,
                    num_workers =  None,
                    save_every = 'best',
                    aug_prob = 0.,
                    recon_loss_weightage = 1.0,
                    disentangle_weightage = 1.0,
                    quantization_weightage = 1.0,
                    hessian_weightage = 1.0,
                    pl_weightage = 1.0,
                    seed = 42,
                    nclasses=10,
                    latent_dim=256,
                    ch=32,
                    out_ch=3,
                    ch_mult=(1, 2, 4, 8),
                    num_res_blocks = 1,
                    dropout=0.0,
                    resamp_with_conv=True,
                    in_channels =3,

                    hiddendim = 256,
                    log = False):

        self.codebook_length = codebook_length
        self.sampling_size = sampling_size
        self.latent_size = latent_dim
        self.cuda = True
        self.log = log
        self.style_depth = style_depth
        self.seed = seed
        self.nclasses = nclasses
        map = lambda x, y: [x[i]//y[-1] for i in range(len(x))]
        self.latentdim = [self.latent_size]+map(image_size, ch_mult)
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.data_root = data_root
        self.logs_root = logs_root
        self.input_size = image_size
        self.num_workers = num_workers
        self.__init__dl()
        print(torch.cuda.is_available())

        self.lr = learning_rate
        self.wd = 0.00001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =  DiscAE(ch = ch, out_ch = out_ch, ch_mult = ch_mult, num_res_blocks = num_res_blocks,
                      dropout = dropout, resamp_with_conv = resamp_with_conv, in_channels = in_channels,
                      z_channels = self.latent_size,sigma = sigma, input = self.latentdim, hiddendim = hiddendim, codebooksize= self.codebook_length).to(self.device)
        self.__init__opt(self.model)



    def __init__dl(self):
        train_loader, test_loader = get(data_root = self.data_root,
                                                    batch_size = self.batch_size,
                                                    num_workers=1,
                                                    input_size = self.input_size)

        return train_loader, test_loader

    def __init__opt(self, model):
        self.opt = Adam(model.parameters(),
                                lr=self.lr,
                                weight_decay=self.wd)
        self.LR_sch = ReduceLROnPlateau(self.opt)



    def training_step(self, train_loader, epoch):

        for batch_idx, (data, _) in enumerate(train_loader):
            #if self.cuda:
               # data = data.cuda()
            
            data = Variable(data.to(self.device))

            self.opt.zero_grad()

            # feature extraction

            recon, quant_loss, hidden_dim, dec, zqf,ce,td= self.model(data)

            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(logits = recon, target = data)

            # disentanglement loss
            disentanglement_loss = torch.mean(calc_pl_lengths(hidden_dim, recon)) + \
                                   hpenalty(dec, zqf, G_z=recon)

            # total loss

            loss = torch.exp(recon_loss_ + 0.2 * disentanglement_loss + quant_loss +ce)

            print(
                f"train epoch:%.1f" % epoch,
                f"train reconloss:%.4f" % recon_loss_,
                f" train disentanglement_loss:%.4f" % disentanglement_loss,
                f"total train cb avg distance:%.4f" % td,
                f"total train cb avg variance:%.4f" % ce,
                f"total train loss:%.4f" % loss
            )

            loss.backward()
            self.opt.step()
        pass


    @torch.no_grad()
    def validation_step(self, val_loader, epoch):
        mean_loss = []
        for batch_idx, (data, _) in enumerate(val_loader):
            #if self.cuda:
             #   data = data.cuda()
            data = Variable(data.to(self.device))


            # feature extraction

            recon, quant_loss, hidden_dim, dec, zqf, ce, td = self.model(data)


            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(recon, data)

            # disentanglement loss
            disentanglement_loss = hpenalty(dec, zqf, G_z=recon)



            # total loss
            loss = torch.exp(recon_loss_ + 0.2 * disentanglement_loss + quant_loss+ce)
            mean_loss.append(loss.cpu().numpy())

            print(
                f"val epoch:%.1f" % epoch,
                f"val reconloss:%.4f" % recon_loss_,
                f" val disentanglement_loss:%.4f" % disentanglement_loss,
                f"total val cb avg distance:%.4f" % td,
                f"total val cb avg variance:%.4f" % ce,
                f"total val loss:%.4f" % loss
            )
        return np.mean(mean_loss)


    def train(self):
        train_loader, valid_loader = self.__init__dl()

        min_loss = np.inf
        for iepoch in range(self.nepochs):

            self.training_step(train_loader, iepoch)
            loss = self.validation_step(valid_loader, iepoch)

            if loss < min_loss:
                self.save_model(iepoch, loss)
            else:
                self.LR_sch.step(loss)

    def save_model(self, iepoch, loss):
        model = {
                'model': self.model.state_dict(),
                'epoch': iepoch,
                'loss': loss
        }

        os.makedirs(os.path.join(self.logs_root, 'models'), exist_ok=True)
        path = os.path.join(self.logs_root, 'models')
        torch.save(model, os.path.join(path, 'best.pth'))


    def load_model(self, path):
        model = torch.load(path)
        loaded_epoch = model['epoch']
        loss = model['loss']

        self.model.load_state_dict(model['codebook'])


    @torch.no_grad()
    def explainations(self):
        pass


def train_from_folder(data_root='/vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv0/TI/data',
                      logs_root='/vol/biomedic3/as217/symbolicAI/SymbolicInterpretability/logs',
                      name='default',
                      image_size=(32,32),
                      style_depth=16,
                      batch_size=50,
                      nepochs=20,
                      learning_rate=2e-4,
                      num_workers=None,
                      save_every=1000,
                      aug_prob=0.,
                      sigma = 0.1,
                      recon_loss_weightage=1.0,
                      disentangle_weightage=1.0,
                      quantization_weightage=1.0,
                      hessian_weightage=1.0,
                      pl_weightage=1.0,
                      seed=42,
                      nclasses=10,
                      latent_dim=256,
                      log=True,
                      ch=32,
                      out_ch=3,
                      ch_mult=(1, 2, 4),
                      num_res_blocks = 1,
                      dropout=0.0,
                      resamp_with_conv=True,
                      in_channels =3,
                      hiddendim = 256,
                      ):
    model_args = dict(
        data_root=data_root,
        logs_root=logs_root,
        name=name,
        image_size=image_size,
        style_depth=style_depth,
        batch_size=batch_size,
        nepochs=nepochs,
        learning_rate=learning_rate,
        num_workers=num_workers,
        save_every=save_every,
        sigma= sigma,
        aug_prob=aug_prob,
        recon_loss_weightage=recon_loss_weightage,
        disentangle_weightage=disentangle_weightage,
        quantization_weightage=quantization_weightage,
        hessian_weightage=hessian_weightage,
        pl_weightage=pl_weightage,
        seed=seed,
        nclasses=nclasses,
        latent_dim=latent_dim,
        log=log,
        ch=ch,
        out_ch=out_ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        resamp_with_conv=resamp_with_conv,
        in_channels=in_channels,
        hiddendim=hiddendim,
    )

    os.makedirs(os.path.join(logs_root, name), exist_ok=True)
    with open(os.path.join(logs_root, name, 'exp-config.json'), 'w') as f:
        json.dump(model_args, f, indent=4)



    trainer = Trainer(**model_args)
    trainer.train()


if __name__ == '__main__':
    fire.Fire(train_from_folder)
