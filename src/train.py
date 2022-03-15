from calendar import c
import torch
from torch import nn
from src.layers import StyleVectorizer, CodeBook
from torch.optim import Adam
import numpy as np
from src.dataset import get
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable

from src.loss import hpenalty, calc_pl_lengths, recon_loss


class Trainer():
    def __init__(self, 
                    classifier, 
                    nfeatures = 1024,
                    latent_size = 64,
                    codebook_length = 1024,
                    sampling_size = 128,
                    num_classes = 10):

        self.codebook_length = codebook_length
        self.sampling_size = sampling_size
        self.num_classes = num_classes
        self.nfeatures = nfeatures
        self.latent_size = latent_size
        self.style_depth = 2

        with torch.no_grad():
            self.feature_extractor = classifier.features
            self.feature_extractor.eval()

            self.classifier_baseline = classifier.classifier
            self.classifier_baseline.eval()

        self.classifier_quantized = classifier.classifier

        self.DiscreteModel()
        self.__init__dl()

        self.lr = 0.001
        self.wd = 0.00001
        self.__init__opt()


    def DiscreteModel(self):
        self.encoder = nn.Linear(self.nfeatures, self.latent_size)
        self.stylevectorizer = StyleVectorizer(self.latent_size, depth = self.style_depth)
        self.codebook = CodeBook(self.codebook_length, self.sampling_size)        


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
        self.LR_sch = ReduceLROnPlateau(self.opt, 
                                step_size=5000, 
                                gamma=0.9)



    def training_step(self, train_loader):

        for batch_idx, (data, _) in enumerate(train_loader):
            if self.cuda:
                data = data.cuda()
            data = Variable(data)

            self.opt.zero_grad()

            # feature extraction
            with torch.no_grad():
                features = self.feature_extractor(data)
                conti_target = self.classifier_baseline(features)

            features = self.encoder(features)
            x_mod = self.stylevectorizer(features)
            x_mod, quant_loss = self.codebook(x_mod)
            dis_target = self.classifier_quantized(x_mod)

            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(dis_target, conti_target)

            # disentanglement loss
            disentanglement_loss = calc_pl_lengths(x_mod, dis_target) + \
                                        hpenalty(self.classifier_quantized, dis_target)

            # total loss
            loss = recon_loss_ + disentanglement_loss + quant_loss
            loss.backward()
            self.opt.step()
            self.LR_sch.step(loss)
        pass


    @torch.no_grad()
    def validation_step(self, val_loader):
        mean_loss = []
        for batch_idx, (data, _) in enumerate(val_loader):
            if self.cuda:
                data = data.cuda()
            data = Variable(data)


            # feature extraction
            with torch.no_grad():
                features = self.feature_extractor(data)
                conti_target = self.classifier_baseline(features)

            features = self.encoder(features)
            x_mod = self.stylevectorizer(features)
            x_mod, quant_loss = self.codebook(x_mod)
            dis_target = self.classifier_quantized(x_mod)

            # recon_loss between continious and discrete features
            recon_loss_ = recon_loss(dis_target, conti_target)

            # disentanglement loss
            disentanglement_loss = calc_pl_lengths(x_mod, dis_target) + \
                                        hpenalty(self.classifier_quantized, dis_target)

            # total loss
            loss = recon_loss_ + disentanglement_loss + quant_loss
            mean_loss.append(loss.cpu().numpy())
        return np.mean(mean_loss)


    def train(self):
        train_loader, valid_loader = self.__init__dl()

        self.training_step(train_loader)
        loss = self.validation_step(valid_loader)
        


    @torch.no_grad()
    def explainations(self):
        pass