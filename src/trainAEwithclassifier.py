from click import style
import torch
import os
from torch import nn
import fire
import json
from layers import DiscAE, DiscClassifier, Decoder
from clsmodel import mnist
from torch.optim import Adam
import numpy as np
from dataset import get
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable
import math
from loss import hpenalty, calc_pl_lengths, recon_loss


class Trainer():
    def __init__(self,
                    classifier,
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
        with torch.no_grad():
            self.feature_extractor = classifier.features.to(self.device)
            self.feature_extractor.eval()

            self.classifier_baseline = classifier.classifier.to(self.device)
            self.classifier_baseline.eval()
        self.modelclass =  DiscClassifier(ch = ch, out_ch = out_ch, ch_mult = ch_mult, num_res_blocks = num_res_blocks,
                      dropout = dropout, resamp_with_conv = resamp_with_conv, in_channels = in_channels,
                      z_channels = self.latent_size,sigma = sigma, input = self.latentdim, hiddendim = hiddendim, device = self.device, codebooksize= self.codebook_length).to(self.device)
        self.inchannel = math.prod(self.latentdim)        
        clfq = []
        clfq.append(nn.Linear(self.inchannel, hiddendim))
        clfq.append(nn.Linear(hiddendim, self.nclasses ))
        self.classifier_quantized = nn.Sequential(*clfq).to(self.device)


        self.opt = Adam(list(self.modelclass.parameters()) +list(self.classifier_quantized.parameters()),
                        lr=self.lr,
                        weight_decay=self.wd)
        self.LR_sch = ReduceLROnPlateau(self.opt)
        self.dec = Decoder(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, input=self.latentdim, hiddendim=hiddendim,
                           dropout=dropout, out_ch=out_ch, resamp_with_conv=resamp_with_conv, in_channels=in_channels,
                           z_channels=self.latent_size).to(self.device)
        self.opt2 = Adam(self.dec.parameters(),
                        lr=self.lr,
                        weight_decay=self.wd)
        self.LR_sch2 = ReduceLROnPlateau(self.opt2)
    def cq(self, hd):
        h = hd.view(hd.size(0), -1)
        dis_target = self.classifier_quantized(h)
        return dis_target

    def __init__dl(self):
        train_loader, test_loader = get(data_root = self.data_root,
                                                    batch_size = self.batch_size,
                                                    num_workers=1,
                                                    input_size = self.input_size)

        return train_loader, test_loader




    def training_step(self, train_loader, epoch):

        for batch_idx, (data, _) in enumerate(train_loader):
            #if self.cuda:
               # data = data.cuda()
            
            data = Variable(data.to(self.device))

            self.opt.zero_grad()
            self.opt2.zero_grad()
            with torch.no_grad():
                features = self.feature_extractor(data)
                features = features.view(features.size(0), -1)
                conti_target = self.classifier_baseline(features)
            # feature extraction

            quant_loss, hidden_dim, zqf, ce , td, hrc = self.modelclass(data)
            h = hidden_dim.view(hidden_dim.size(0), -1)
            dis_target = self.cq(hidden_dim)
            class_loss_ = recon_loss(logits = dis_target, target =conti_target)
            recon = self.dec(hidden_dim)
            # disentanglement loss
            disentanglement_loss = torch.mean(calc_pl_lengths(hidden_dim, recon)) + \
                                   hpenalty(self.dec, zqf, G_z=recon)
            disentanglement_classloss = hpenalty(self.cq, hidden_dim, G_z=dis_target) 
            #loss = class_loss_ + 0.2 * disentanglement_loss + disentanglement_classloss + quant_loss +ce
            loss = class_loss_ +  quant_loss +ce
           # total loss
           # if loss > 0:
            #    loss = loss
           # else:
            #    loss = torch.exp(class_loss_ + 0.2 * disentanglement_loss + quant_loss +ce)
            
            for p in self.dec.parameters(): p.requires_grad = False
            loss.backward(retain_graph = True)
            recon_loss_ = recon_loss(logits = recon, target = data)
            for p in self.dec.parameters(): p.requires_grad = True
            for p  in self.modelclass.parameters(): p.requires_grad = False
            for p in self.classifier_quantized.parameters(): p.requires_grad = False

            recon_loss_.backward()
            self.opt.step()

            # recon_loss between continious and discrete features

            self.opt2.step()
            for p  in self.modelclass.parameters(): p.requires_grad = True
            for p in self.classifier_quantized.parameters(): p.requires_grad = True 
            print(
                f"train epoch:%.1f" % epoch,
                f"train reconloss:%.4f" % recon_loss_,
                f"hypersphereloss train:%.4f" % hrc,
                f"train disentanglement_loss:%.4f" % disentanglement_loss,
                f"train disentanglement_classloss:%.4f" % disentanglement_classloss,
                f"train classloss:%.4f" % class_loss_,
                f"total train cb avg distance:%.4f" % td,
                f"total train cb avg variance:%.4f" % ce,
                f"total train loss:%.4f" % loss
            )


        pass


    @torch.no_grad()
    def validation_step(self, val_loader, epoch):
        mean_loss = []
        mean_recon_loss_ = []
        for batch_idx, (data, _) in enumerate(val_loader):
            #if self.cuda:
             #   data = data.cuda()
            data = Variable(data.to(self.device))


            # feature extraction
            with torch.no_grad():
                features = self.feature_extractor(data)
                features = features.view(features.size(0), -1)
                conti_target = self.classifier_baseline(features)
            # feature extraction

            quant_loss, hidden_dim, zqf, ce , td, hrc = self.modelclass(data)
            h = hidden_dim.view(hidden_dim.size(0), -1)
            dis_target = self.classifier_quantized(h)
            recon = self.dec(hidden_dim)


            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(logits = recon, target = data)
            class_loss_ = recon_loss(logits = dis_target, target =conti_target)

            # disentanglement loss
            disentanglement_loss =         hpenalty(self.dec, zqf, G_z=recon)



            # total loss
            loss = torch.exp(class_loss_ + 0.2 * disentanglement_loss + quant_loss+ce)
            mean_loss.append(loss.cpu().numpy())
            mean_recon_loss_.append(recon_loss_.cpu().numpy())

            print(
                f"val epoch:%.1f" % epoch,
                f"val reconloss:%.4f" % recon_loss_,
                f"hypersphereloss val:%.4f" % hrc,
                f" val disentanglement_loss:%.4f" % disentanglement_loss,
                f"val classloss:%.4f" % class_loss_,
                f"total val td avg distance:%.4f" % td,
                f"total val cb avg variance:%.4f" % ce,
                f"total val loss:%.4f" % loss
            )
        return np.mean(mean_loss), np.mean(mean_recon_loss_)


    def train(self):
        train_loader, valid_loader = self.__init__dl()

        min_loss = np.inf
        min_recon = np.inf
        for iepoch in range(self.nepochs):

            self.training_step(train_loader, iepoch)
            loss, rloss = self.validation_step(valid_loader, iepoch)

            if loss < min_loss:
                self.save_classmodel(iepoch, loss)
                min_loss = loss
            else:
                self.LR_sch.step(loss)

            if rloss < min_recon:
                self.save_decmodel(iepoch, loss)
                min_recon = rloss
            else:
                self.LR_sch2.step(loss)

    def save_classmodel(self, iepoch, loss):
        model = {
                'modelclass': self.modelclass.state_dict(),
                'discreteclassifier':self.classifier_quantized.state_dict(),
                'epoch': iepoch,
                'loss': loss
        }

        os.makedirs(os.path.join(self.logs_root, 'models'), exist_ok=True)
        path = os.path.join(self.logs_root, 'models')
        torch.save(model, os.path.join(path, 'best.pth'))

    def save_decmodel(self, iepoch, loss):
        model = {
                'decmodel': self.dec.state_dict(),
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
                      nepochs=50,
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
