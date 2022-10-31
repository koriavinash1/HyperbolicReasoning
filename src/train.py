from base64 import decode
from click import style
import torch
import os
from torch import nn
import fire
import json

from src.layer import Decoder,  HierarchyVQmodulator
from src.loss import recon_loss, ce_loss
from src.dataset import get
from src.radam import RiemannianAdam
from src.reasoning import MomentumWithThresholdBinaryOptimizer, BinaryLinear

from torch.optim import Adam, SGD
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

import math
import torchvision
from sklearn.metrics import accuracy_score, f1_score
import progressbar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trainer class
class Trainer():
    def __init__(self,
                    classifier,
                    codebook_size = [128, 64, 32],
                    data_root = './data',
                    logs_root ='./logs/',
                    image_size = (64,64),
                    batch_size = 10,
                    latent_dim=256,
                    inchannels_codebook = 64,
                    ch_mult=(1, 2, 4, 8),
                    nclasses=10,
                    ch=32,
                    num_res_blocks = 1,
                    avg_pool=False,
                    nepochs = 20,
                    learning_rate = 1e-3,
                    decoder_learning_rate = 1e-2,
                    num_workers =  None,
                    sigma=0.1,
                    seed = 42,
                    resamp_with_conv=True,
                    in_channels =3,
                    log = False,
                    trim=False,
                    quantise = 'feature',
                    reasoning=True):

        self.latent_size = latent_dim
        self.quantise = quantise
        self.cuda = True
        self.log = log
        self.seed = seed
        self.nclasses = nclasses

        map = lambda x, y: [x[i]//y[-1] for i in range(len(x))]
        self.emb_dim = inchannels_codebook if self.quantise == 'spatial' else int(np.prod(map(image_size, ch_mult))) 
        self.latentdim = [self.latent_size]+map(image_size, ch_mult)

        self.nepochs = nepochs
        self.batch_size = batch_size
        self.data_root = data_root
        self.logs_root = logs_root
        self.input_size = image_size
        self.num_workers = num_workers

        self.given_channels = inchannels_codebook
        self.required_channels = latent_dim
        self.trim = trim 
        self.reasoning = reasoning
        

        self.__init__dl()
        print(torch.cuda.is_available())

        self.avg_pool = avg_pool
        self.lr = learning_rate
        self.dlr = decoder_learning_rate
        self.wd = 0.00001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Extract features from pretrained classifier
        with torch.no_grad():
            self.feature_extractor = classifier.features.to(self.device).eval()
            self.classifier_baseline_ = classifier.classifier.to(self.device).eval()
            if avg_pool:
                self.classifier_baseline = lambda x: self.classifier_baseline_(nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)) 
            else:
                self.classifier_baseline = self.classifier_baseline_


            for p in self.classifier_baseline.parameters(): p.requires_grad = False 
            for p in self.feature_extractor.parameters(): p.requires_grad = False 
        

        # Code book defn
        self.codebook_size = codebook_size 
        self.modelclass = HierarchyVQmodulator(features = self.given_channels,  
                                                z_channels = self.required_channels, 
                                                emb_dim = self.emb_dim,
                                                codebooksize = codebook_size, 
                                                device = self.device,
                                                trim = self.trim,
                                                quantise = self.quantise,
                                                reasoning=self.reasoning).to(self.device)
        
        # Quantized classifier
        clfq = []
        #clfq.append(torch.nn.Linear(self.emb_dim, self.emb_dim//2))
        #clfq.append(torch.nn.Linear(self.emb_dim//2, self.nclasses))
        clfq.append(torch.nn.Linear(self.given_channels, self.given_channels//2))
        clfq.append(torch.nn.Linear(self.given_channels//2, self.nclasses))
        self.classifier_quantized = nn.Sequential(*clfq).to(self.device)
        for p in self.classifier_quantized.parameters(): p.requires_grad = True

            

        # Optimiser for binary weights
        self.opt = MomentumWithThresholdBinaryOptimizer(
                         list(self.modelclass.reasoning_parameters()),
                         list(self.modelclass.other_parameters())+ list(self.classifier_quantized.parameters()),
                         ar=0.0001, 
                         threshold=1e-7,
                         adam_lr=self.lr,
                     )
        #self.LR_sch = ReduceLROnPlateau(self.opt._adam, patience=2)



        # Decoder and optimizer
        self.dec = Decoder(ch = ch, 
                            ch_mult = ch_mult, 
                            num_res_blocks = num_res_blocks, 
                            input = self.latentdim, 
                            out_ch = 3, 
                            resamp_with_conv = resamp_with_conv, 
                            in_channels = in_channels,
                            z_channels = self.required_channels).to(self.device)
        self.opt2 = Adam(self.dec.parameters(),
                        lr=self.dlr,
                        weight_decay=self.wd)
        self.LR_sch2 = ReduceLROnPlateau(self.opt2, patience=2)



        # number of parameters
        print ('FeatureExtractor: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad), sum(p.numel() for p in self.feature_extractor.parameters())))
        print ('ContiClassifier: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.classifier_baseline_.parameters() if p.requires_grad), sum(p.numel() for p in self.classifier_baseline_.parameters())))
        print ('codebook: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.modelclass.parameters() if p.requires_grad), sum(p.numel() for p in self.modelclass.parameters())))
        print ('DisClassifier: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.classifier_quantized.parameters() if p.requires_grad), sum(p.numel() for p in self.classifier_quantized.parameters())))
        print ('Decoder: Total number of trainable params: {}/{}'.format(sum(p.numel() for p in self.dec.parameters() if p.requires_grad), sum(p.numel() for p in self.dec.parameters())))
 

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


    def eval(self):
        self.modelclass.eval()
        self.dec.eval()
        self.classifier_quantized.eval()
        self.classifier_baseline_.eval()
        self.feature_extractor.eval()


    def training_step(self, train_loader, epoch):
        
        for batch_idx, (data, _) in enumerate(train_loader):

            data = Variable(data.to(self.device))
            m = nn.Softmax(dim=1)

            self.opt.zero_grad()
            self.opt2.zero_grad()


            # feature extraction
            with torch.no_grad():
                f = self.feature_extractor(data)
                f_ = f if self.avg_pool else f.view(f.size(0), -1)
                conti_target = m(self.classifier_baseline(f_))
                conti_target = torch.argmax(conti_target, 1)
                    

            # code book sampling
            quant_loss, ploss, features, _, cvE , cdE, attnblocks, codebooks  = self.modelclass(f)

            if isinstance(features, list):
                classifier_features = features[-1]
                decoder_features = features[0]
            else:
                decoder_features = classifier_features = features

            classifier_features = torch.mean(classifier_features.view(classifier_features.shape[0], classifier_features.shape[1], classifier_features.shape[2]*classifier_features.shape[3]), 2)
            
            """
            classifier_features = classifier_features.view(classifier_features.shape[0], classifier_features.shape[1], classifier_features.shape[2]*classifier_features.shape[3])
            cf = []
            for i in range(classifier_features.shape[0]):
                x = torch.round(torch.mean(classifier_features[i], dim =1)* 10**5)/ (10**5)
                xx = torch.unique(x)
                w = []
                for j in range(xx.shape[0]):
                    y = torch.nonzero(xx[j] == x)
                    z = classifier_features[i][y[0]]
                    w.append(z)
                w = torch.mean(torch.stack(w), dim = 0)
                cf.append(w)
            
            classifier_features = torch.stack(cf)
            """
<<<<<<< HEAD
=======
            
            #classifier_features = torch.stack(cf)
>>>>>>> b818b5c28a852b7d90ad90437b45c8bcdfb93865
            #classifier_features = torch.mean(classifier_features, dim =1)    
            
            classifier_features = torch.mean(classifier_features, dim =2)
            #print(classifier_features.shape)
            dis_target = m(self.cq(classifier_features))
            class_loss_ = ce_loss(logits = dis_target, target = conti_target)


            recon = self.dec(decoder_features.clone().detach())
            recon_loss_ = recon_loss(logits = recon, target = data)
            recon_loss_.backward()
            self.opt2.step()

            loss = class_loss_ +  quant_loss + ploss #+ recon_loss_  # quant_loss = quant_loss + cb_disentanglement_loss
            loss.backward()
            self.opt.step()
            """             
            # Normalising weights of edges
            for l in self.modelclass.Aggregation:
                #l.weight=torch.nn.Parameter((l.weight - torch.min(l.weight))/(torch.max(l.weight) - torch.min(l.weight)))
                print(l.weight.grad)
            # Normalise weights of feature attention function
            for l in self.modelclass.featureattns:
                #l.weight=torch.nn.Parameter((l.weight - torch.min(l.weight))/(torch.max(l.weight) - torch.min(l.weight)))
                print(l.weight.grad)

            """
            self.training_pbar.update(batch_idx)
            #self.training_widgets[0] = progressbar.FormatLabel(
            print(
                                f" tepoch:%.1f" % epoch +
                                f" tcloss:%.4f" % class_loss_ +
                                f" poincareloss:%.4f" % ploss +
                                f" trcnloss:%.4f" % recon_loss_ +
                                f" tqloss:%.4f" % quant_loss +
                                f" t<mean Euclidian codebook distance>:%.4f" % cdE +
                                f" t<mean Euclidian codebook variance>:%.4f" % cvE +
                                f" ttloss:%.4f" % loss
                            )
        pass


    @torch.no_grad()
    def validation_step(self, val_loader, epoch):
        mean_loss = []; mean_recon_loss_ = []
        mean_f1_score = []; mean_acc_score = []

        plot = False 
        for batch_idx, (data, _) in enumerate(val_loader):

            data = Variable(data.to(self.device))
            m = nn.Softmax(dim=1)


            # feature extraction
            with torch.no_grad():
                f = self.feature_extractor(data)
                f_ = f if self.avg_pool else f.view(f.size(0), -1)
                conti_target = m(self.classifier_baseline(f_))
                conti_target = torch.argmax(conti_target, 1)


            # code book sampling
            quant_loss, ploss, features, _, cdE, cvE,  attnblocks, codebooks = self.modelclass(f)

            if isinstance(features, list):
                classifier_features = features[-1]
                decoder_features = features[0]
            else:
                decoder_features = classifier_features = features


            classifier_features = torch.mean(classifier_features.view(classifier_features.shape[0], classifier_features.shape[1], classifier_features.shape[2]*classifier_features.shape[3]), 2)
            """
            classifier_features = classifier_features.view(classifier_features.shape[0], classifier_features.shape[1], classifier_features.shape[2]*classifier_features.shape[3])
            cf = []
            for i in range(classifier_features.shape[0]):
                x = torch.round(torch.mean(classifier_features[i], dim =1)* 10**5)/ (10**5)
                xx = torch.unique(x)
                w = []
                for j in range(xx.shape[0]):
                    y = torch.nonzero(xx[j] == x)
                    z = classifier_features[i][y[0]]
                    w.append(z)
                w = torch.mean(torch.stack(w), dim = 0)
                cf.append(w)

            classifier_features = torch.stack(cf)
            """
            dis_target = m(self.cq(classifier_features))
            recon = self.dec(decoder_features)

            # save sample reconstructions
            results_dir = os.path.join(self.logs_root, 'recon_imgs')
            os.makedirs(results_dir, exist_ok=True)

            if batch_idx == 1:
                torchvision.utils.save_image(recon, 
                                        str(results_dir + f'/{str(epoch)}-recon.png'), 
                                        nrow=int(self.batch_size**0.5))
                torchvision.utils.save_image(data, 
                                        str(results_dir + f'/{str(epoch)}-orig.png'), 
                                        nrow=int(self.batch_size**0.5))


            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(logits = recon, target = data)
            class_loss_ = ce_loss(logits = dis_target, target = conti_target)


            # total loss
            loss = class_loss_ + quant_loss + ploss
            mean_loss.append(loss.cpu().numpy())
            mean_recon_loss_.append(recon_loss_.cpu().numpy())

            # acc metrics
            acc = accuracy_score(torch.argmax(dis_target, 1).cpu().numpy(),
                                            conti_target.cpu().numpy())
            f1_ = f1_score(torch.argmax(dis_target, 1).cpu().numpy(),
                                            conti_target.cpu().numpy(), average='micro')


            mean_f1_score.append(f1_)
            mean_acc_score.append(acc)

            self.validation_pbar.update(batch_idx)
            #self.validation_widgets[0] = progressbar.FormatLabels(
            print(                    
                                f" vepoch:%.1f" % epoch +
                                f" vrcnloss:%.4f" % recon_loss_ +
                                f" vcloss:%.4f" % class_loss_ +
                                f" poincareloss:%.4f" % ploss +
                                f" v<mean Euclidian codebook distance>:%.4f" % cdE +
                                f" v<mean Eucdlian codebook variance>:%.4f" % cvE +
                                f" tv loss:%.4f" % loss +
                                f" vF1:%.4f" % f1_ +
                                f" vAcc:%.4f" % acc 
                            )

        return (np.mean(mean_loss), 
                    np.mean(mean_recon_loss_), 
                    np.mean(mean_f1_score), 
                    np.mean(mean_acc_score), attnblocks, codebooks)


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

        plot = True
        # Plot Poincare graph for 2D embeddinggs
        for iepoch in range(self.nepochs):

            self.training_step(train_loader, iepoch)
            loss, rloss, f1, acc, attnblocks, codebooks= self.validation_step(valid_loader, iepoch)
          
            if plot and ((iepoch % 1) ==0):
                plt.figure(figsize=(10, 10))
                for i in range(len(attnblocks)):
                    att = attnblocks[i].cpu().numpy()
                    cb1 = codebooks[i].cpu().numpy()
                    cb2 = codebooks[i+1].cpu().numpy()
                    for j in range(cb1.shape[0]):
                        for k in range(cb2.shape[0]):
                            point1 = cb1[j]
                            point2 = cb2[k]


                            x1 = [point1[0]]
                            x2 = [point2[0]]
                            y1 = [point1[1]]
                            y2 = [point2[1]]
                            ptarrow = False
                            if att[j,k] == 1:
                                if i == 0:
                                    ptarrow = np.sum(att[:, k]) >= 10 #0.*att.shape[0]
                                    plt.plot(x1, y1, 'ro')
                                    plt.plot(x2, y2, 'bo')
                                elif i == 1:
                                    ptarrow = np.sum(att[:, k]) >= 5 #0.*att.shape[0]
                                    plt.plot(x2, y2, 'go')

                            if ptarrow:
                                plt.arrow(x2[0], y2[0], x1[0]-x2[0], y1[0]-y2[0])

                r_patch = mpatches.Patch(color='r', label="$\zeta^0$ symbols")
                b_patch = mpatches.Patch(color='b', label="$\zeta^1$ symbols")
                g_patch = mpatches.Patch(color='g', label="$\zeta^2$ symbols")

                plt.legend(handles=[r_patch, b_patch, g_patch])
                plt.savefig(f'poincaresphere-{iepoch}.png', )
                  

            stats = {'loss': loss, 'f1': f1, 'acc': acc, 'rloss': rloss}
            print ('Epoch: {}. Stats: {}'.format(iepoch, stats))

            if loss < min_loss:
                self.save_classmodel(iepoch, stats)
                min_loss = loss
            #else:
            #    self.LR_sch.step(loss)

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
