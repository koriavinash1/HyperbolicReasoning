from click import style
import torch
import os
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
                    codebook_length = 1024,
                    sampling_size = 128,
                    data_root = './data',
                    logs_root ='./logs/',
                    image_size = 128,
                    style_depth = 16,
                    batch_size = 5,
                    nepochs = 150,
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
                    latent_dim=512,
                    featurelen=1024,
                    encoder=False,
                    log = False,):

        self.codebook_length = codebook_length
        self.sampling_size = sampling_size
        self.nfeatures = featurelen
        self.latent_size = latent_dim
        self.encoder = encoder
        self.log = log
        self.style_depth = style_depth
        self.seed = seed 
        self.nclasses = nclasses

        with torch.no_grad():
            self.feature_extractor = classifier.features
            self.feature_extractor.eval()

            self.classifier_baseline = classifier.classifier
            self.classifier_baseline.eval()

        self.classifier_quantized = classifier.classifier

        self.DiscreteModel()

        self.nepochs = nepochs
        self.batch_size = batch_size
        self.data_root = data_root
        self.logs_root = logs_root
        self.input_size = image_size
        self.num_workers = num_workers
        self.__init__dl()

        self.lr = learning_rate
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
                                        hpenalty(self.classifier_quantized, 
                                                    x_mod, G_z=dis_target)

            # total loss
            loss = recon_loss_ + disentanglement_loss + quant_loss
            loss.backward()
            self.opt.step()
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
                                        hpenalty(self.classifier_quantized, 
                                                        x_mod, G_z=dis_target)

            # total loss
            loss = recon_loss_ + disentanglement_loss + quant_loss
            mean_loss.append(loss.cpu().numpy())
        return np.mean(mean_loss)


    def train(self):
        train_loader, valid_loader = self.__init__dl()

        min_loss = np.inf
        for iepoch in range(self.nepochs):
            self.training_step(train_loader)
            loss = self.validation_step(valid_loader)

            if loss < min_loss:
                self.save_model(iepoch, loss)
            else:
                self.LR_sch.step(loss)

    def save_model(self, iepoch, loss):
        model = {
                'encoder': self.encoder.state_dict(),
                'codebook': self.codebook.state_dict(),
                'classifier': self.classifier_quantized.state_dict(),
                'modulator': self.stylevectorizer.state_dict(),
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

        self.encoder.load_state_dict(model['encoder'])
        self.codebook.load_state_dict(model['codebook'])
        self.classifier_quantized.load_state_dict(model['classifier'])
        self.stylevectorizer.load_state_dict(model['modulator'])


    @torch.no_grad()
    def explainations(self):
        pass