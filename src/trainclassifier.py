from click import style
import torch
import os
from torch import nn
import fire
import json
from layers import DiscAE, DiscClassifier, Decoder, VQmodulator, HierarchyVQmodulator
from clsmodel import mnist #, afhq, stl10
from torch.optim import Adam
import numpy as np
from dataset import get
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable
import math
import torchvision

from loss import hpenalty, calc_pl_lengths, recon_loss, ce_loss

from sklearn.metrics import accuracy_score, f1_score
import progressbar



class Trainer():
    def __init__(self,
                    classifier,
                    codebook_length = 256,
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
            self.feature_extractor = classifier.features.to(self.device).eval()
            self.classifier_baseline = classifier.classifier.to(self.device).eval()
            for p in self.classifier_baseline.parameters(): p.requires_grad = False 
            for p in self.feature_extractor.parameters(): p.requires_grad = False 
        

        # Code book defn
        # self.modelclass = VQmodulator(features = 64,  z_channels = self.latent_size, codebooksize = self.codebook_length, device = self.device).to(self.device)
        self.modelclass = HierarchyVQmodulator(features = 64,  z_channels = self.latent_size, codebooksize = self.codebook_length, device = self.device).to(self.device)
        
        # Quantized classifier
        self.inchannel = np.prod(self.latentdim)        
        clfq = []
        clfq.append(nn.Linear(self.inchannel, hiddendim))
        clfq.append(nn.Linear(hiddendim, self.nclasses ))
        
        self.classifier_quantized = nn.Sequential(*clfq).to(self.device)



        # Optimizers
        self.opt = Adam(list(self.modelclass.parameters()) + list(self.classifier_quantized.parameters()),
                        lr=self.lr,
                        weight_decay=self.wd)
        self.LR_sch = ReduceLROnPlateau(self.opt)



        # Decoder and optimizer
        self.dec = Decoder(ch=ch, 
                            ch_mult=ch_mult, 
                            num_res_blocks=num_res_blocks, 
                            input=self.latentdim, 
                            hiddendim=hiddendim,
                            dropout=dropout, 
                            out_ch=out_ch, 
                            resamp_with_conv=resamp_with_conv, 
                            in_channels=in_channels,
                            z_channels=self.latent_size).to(self.device)
        self.opt2 = Adam(self.dec.parameters(),
                        lr=self.lr,
                        weight_decay=self.wd)
        self.LR_sch2 = ReduceLROnPlateau(self.opt2)


        # number of parameters
        print ('FeatureExtractor: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad), sum(p.numel() for p in self.feature_extractor.parameters())))
        print ('ContiClassifier: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.classifier_baseline.parameters() if p.requires_grad), sum(p.numel() for p in self.classifier_baseline.parameters())))
        print ('DisClassifier: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.modelclass.parameters() if p.requires_grad), sum(p.numel() for p in self.modelclass.parameters())))
        print ('CodeBook: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.classifier_quantized.parameters() if p.requires_grad), sum(p.numel() for p in self.classifier_quantized.parameters())))
 

        self.training_widgets = [progressbar.FormatLabel(''),
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]


        self.validation_widgets = [progressbar.FormatLabel(''),
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]

    def cq(self, hd):
        h = hd.view(hd.size(0), -1)
        dis_target = self.classifier_quantized(h)
        return dis_target




    def __init__dl(self):
        train_loader, test_loader = get(data_root = self.data_root,
                                                    batch_size = self.batch_size,
                                                    num_workers=16,
                                                    input_size = self.input_size)

        return train_loader, test_loader




    def training_step(self, train_loader, epoch):
        
        for batch_idx, (data, _) in enumerate(train_loader):

            data = Variable(data.to(self.device))
            m = nn.Softmax(dim=1)

            self.opt.zero_grad()
            self.opt2.zero_grad()


            # feature extraction
            with torch.no_grad():
                f = self.feature_extractor(data)
                features1 = f.view(f.size(0), -1)
                conti_target = m(self.classifier_baseline(features1))
                conti_target = torch.argmax(conti_target, 1)
                

            # code book sampling
            quant_loss, latent_features, zqf, ce , td, hrc, r = self.modelclass(f)

            dis_target = m(self.cq(latent_features))

            # class_loss_ = recon_loss(logits = dis_target, target = conti_target)
            class_loss_ = ce_loss(logits = dis_target, target = conti_target)


            loss = class_loss_ +  quant_loss #quant_loss = quant_loss + cb_disentanglement_loss
            loss.backward()
            self.opt.step()


            # disentanglement loss
            # disentanglement_loss = torch.mean(calc_pl_lengths(hidden_dim, recon)) + \
            #                      hpenalty(self.dec, zqf, G_z=recon)
            # disentanglement_classloss = hpenalty(self.cq, hidden_dim, G_z=dis_target) 
            # loss = class_loss_ + 0.2 * disentanglement_loss + disentanglement_classloss + quant_loss +ce
            # loss = class_loss_ +  quant_loss +ce
            # total loss
            # if loss > 0:
            #    loss = loss
            # else:
            #    loss = torch.exp(class_loss_ + 0.2 * disentanglement_loss + quant_loss +ce)

            #for p in self.dec.parameters(): p.requires_grad = False
            #loss.backward(retain_graph = True)


            recon = self.dec(latent_features.detach())
            recon_loss_ = recon_loss(logits = recon, target = data)
            recon_loss_.backward()
            self.opt2.step()


            # for p in self.dec.parameters(): p.requires_grad = True
            # for p  in self.modelclass.parameters(): p.requires_grad = False
            # for p in self.classifier_quantized.parameters(): p.requires_grad = False
            # recon_loss between continious and discrete features
            # for p  in self.modelclass.parameters(): p.requires_grad = True


            with torch.no_grad():
                 self.modelclass.quantize.r.clamp_(0.9, 1.1)

            self.training_widgets[0] = progressbar.FormatLabel(
                                f" train epoch:%.1f" % epoch +
                                f" train classloss:%.4f" % class_loss_ +
                                f" train reconloss:%.4f" % recon_loss_ +
                                f" train qloss:%.4f" % quant_loss +
                                f" radius:%.4f" % r +
                                f" hypersphereloss train:%.4f" % hrc +
                                # f" train disentanglement_loss:%.4f" % disentanglement_loss +
                                # f" train disentanglement_classloss:%.4f" % disentanglement_classloss +
                                f" total train cb avg distance:%.4f" % td +
                                f" total train cb avg variance:%.4f" % ce +
                                f" total train loss:%.4f" % loss
                            )
            self.training_pbar.update(batch_idx)
            break
        pass


    @torch.no_grad()
    def validation_step(self, val_loader, epoch):
        mean_loss = []; mean_recon_loss_ = []
        mean_f1_score = []; mean_acc_score = []


        for batch_idx, (data, _) in enumerate(val_loader):

            data = Variable(data.to(self.device))
            m = nn.Softmax(dim=1)


            # feature extraction
            with torch.no_grad():
                features = self.feature_extractor(data)
                features1 = features.view(features.size(0), -1)
                conti_target = m(self.classifier_baseline(features1))
                conti_target = torch.argmax(conti_target, 1)



            # code book sampling
            quant_loss, latent_features, zqf, ce , td, hrc, r = self.modelclass(features)

            dis_target = m(self.cq(latent_features))
            recon = self.dec(latent_features)

            # save sample reconstructions
            results_dir = os.path.join(self.logs_root, 'recon_imgs')
            os.makedirs(results_dir, exist_ok=True)

            if batch_idx == 1:
                torchvision.utils.save_image(recon, 
                                        str(results_dir + f'/{str(epoch)}.png'), 
                                        nrow=int(self.batch_size**0.5))
                torchvision.utils.save_image(data, 
                                        str(results_dir + f'/{str(epoch)}.png'), 
                                        nrow=int(self.batch_size**0.5))


            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(logits = recon, target = data)
            class_loss_ = ce_loss(logits = dis_target, target = conti_target)


            # disentanglement loss
            # disentanglement_loss =         hpenalty(self.dec, zqf, G_z=recon)



            # total loss
            loss = class_loss_ + quant_loss + ce
            mean_loss.append(loss.cpu().numpy())
            mean_recon_loss_.append(recon_loss_.cpu().numpy())

            # acc metrics
            acc = accuracy_score(torch.argmax(dis_target, 1).cpu().numpy(),
                                            conti_target.cpu().numpy())
            f1_ = f1_score(torch.argmax(dis_target, 1).cpu().numpy(),
                                            conti_target.cpu().numpy(), average='micro')


            mean_f1_score.append(f1_)
            mean_acc_score.append(acc)

            self.validation_widgets[0] = progressbar.FormatLabel(
                                f" val epoch:%.1f" % epoch +
                                f" val reconloss:%.4f" % recon_loss_ +
                                f" hypersphereloss val:%.4f" % hrc + 
                                # f" val disentanglement_loss:%.4f" % disentanglement_loss +
                                f" val classloss:%.4f" % class_loss_ +
                                f" total val td avg distance:%.4f" % td +
                                f" total val cb avg variance:%.4f" % ce +
                                f" total val loss:%.4f" % loss +
                                f" F1:%.4f" % loss +
                                f" Accuracy:%.4f" % loss
                            )
            self.validation_pbar.update(batch_idx)

        return (np.mean(mean_loss), 
                    np.mean(mean_recon_loss_), 
                    np.mean(mean_f1_score), 
                    np.mean(mean_acc_score))


    def train(self):
        train_loader, valid_loader = self.__init__dl()

        self.training_pbar = progressbar.ProgressBar(widgets=self.training_widgets, 
                                    maxval=len(train_loader))
        self.validation_pbar = progressbar.ProgressBar(widgets=self.validation_widgets, 
                                    maxval=len(valid_loader))
        self.training_pbar.start()
        self.validation_pbar.start()
        min_loss = np.inf
        min_recon = np.inf



        for iepoch in range(self.nepochs):

            self.training_step(train_loader, iepoch)
            loss, rloss, f1, acc  = self.validation_step(valid_loader, iepoch)

            stats = {'loss': loss, 'f1': f1, 'acc': acc, 'rloss': rloss}
            print ('Epoch: {}. Stats: {}'.format(iepoch, stats))


            if loss < min_loss:
                self.save_classmodel(iepoch, stats)
                min_loss = loss
            else:
                self.LR_sch.step(loss)

            if rloss < min_recon:
                min_recon = rloss
            else:
                self.LR_sch2.step(loss)



    def save_classmodel(self, iepoch, stats):
        model = {
                'modelclass': self.modelclass.state_dict(),
                'discreteclassifier':self.classifier_quantized.state_dict(),
                'decmodel': self.dec.state_dict(),
                'epoch': iepoch,
                'stats': stats
        }

        os.makedirs(os.path.join(self.logs_root, 'models'), exist_ok=True)
        path = os.path.join(self.logs_root, 'models')
        torch.save(model, os.path.join(path, 'best.pth'))



    def load_model(self, path):
        model = torch.load(path)
        loaded_epoch = model['epoch']
        stats = model['stats']

        print ("Model loaded from {}, loaded epoch:{} with stats: {}".format(path, loaded_epoch, stats))

        self.modelclass.load_state_dict(model['modelclass'])
        self.classifier_quantized.load_state_dict(model['discreteclassifier'])
        self.dec.load_state_dict(model['decmodel'])


    @torch.no_grad()
    def explainations(self):
        pass


def train_from_folder(data_root='/vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv0/TI/data',
                      logs_root='../logs',
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
                      latent_dim=64,
                      log=True,
                      ch=32,
                      out_ch=3,
                      ch_mult=(1, 2, 4, 8),
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
