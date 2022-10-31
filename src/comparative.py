from ast import Pass
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from captum.attr import Lime, LimeBase
from captum.attr import  LayerGradCam
from captum.attr import DeepLift
from captum.attr import GradientShap


class Baseline(object):
    def __init__(self, classifier, niter = 100):
        self.classifier = classifier
        self.niter = niter

    def get_noisy(self, image, 
                        type = 'gauss', 
                        weightage=0.3):
        if type == "gauss":
            mean = 0; var = 1; sigma = var**0.5
            gauss = mean + var*torch.rand(image.shape)
            gauss = gauss.view(*image.shape)
            noisy = image + weightage*gauss.to(image.device)
            return noisy

        elif type == "s&p":
            s_vs_p = 0.5

            out = image.clone()
            
            # Salt mode
            num_salt = np.ceil(weightage * image.size * s_vs_p)
            coords = [torch.randint(0, i - 1, int(num_salt)) for i in image.shape[-3:]]
            for x, y, z in zip(*coords):
                out[0, 0, x, y, z] = 1

            # Pepper mode
            num_pepper = np.ceil(weightage* image.size * (1. - s_vs_p))
            coords = [torch.randint(0, i - 1, int(num_pepper)) for i in image.shape[-3:]]
            for x, y, z in zip(*coords):
                out[0, 0, x, y, z] = 0
                
            return out

        elif type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = torch.poisson(weightage * image * vals) / float(vals)
            return noisy

        elif type =="speckle":
            gauss = torch.rand(*image.shape)
            gauss = gauss.view(*image.shape).to(image.device)        
            noisy = image + weightage*image * gauss
            return noisy


    def compute_robustness(self, images, 
                                labels,
                                type = 'gaussian', 
                                percentage=0.3):

        perturbations = []

        for _ in range(self.niter):
            noisy_images = self.get_noisy(images, type, percentage) 
            exp = self.get_explainations(noisy_images, labels)
            perturbations.append(exp.detach().cpu().numpy())
        
        perturbations = np.concatenate(perturbations, 0)
        return np.mean(np.var(perturbations, 0))


class LIME(Baseline):
    def __init__(self, 
                    classifier, 
                    npertub = 25):
        super().__init__(classifier)

        self.lime = Lime(classifier)
        self.npertub = npertub

    def get_explanations(self, images, labels):
        lime_attrs = self.lime.attribute(
                    images,
                    target=labels,
                    perturbations_per_eval=self.npertub).cpu().numpy()
        return lime_attrs


class SHAP(Baseline):
    def __init__(self, classifier, 
                        nsamples=25, 
                        std=1e-4):
        self.gs = GradientShap(classifier)
        self.nsamples = nsamples
        self.std = std


    def get_explanations(self, images, labels):
        gs_attrs = np.concatenate([self.gs.attribute(images[i].unsqueeze(0),
                                                        n_samples=self.nsamples,
                                                        stdevs=self.std,
                                                        baselines=images[i].unsqueeze(0) * 0,
                                                        target=l).cpu().numpy()  for i, l in enumerate(labels)], 0)

        return gs_attrs



class GradCAM(Baseline):
    def __init__(self, classifier, layeridx=10):
        self.saliency = LayerGradCam(classifier, classifier.features[layeridx])
    
    def get_explanations(self, images, labels):
        image_size = images[0].shape
        gradCAM = lambda x, i: F.upsample(self.saliency.attribute(x, target=i), size=(image_size[1], image_size[2]), mode='bilinear')
        saliency_attrs = np.concatenate([gradCAM(images[i].unsqueeze(0), l).detach().cpu().numpy().repeat(3, 1) for i, l in enumerate(labels)], 0)

        return saliency_attrs


class DeepLIFT(Baseline):
    def __init__(self, classifier):
        self.dl = DeepLift(classifier)

    def get_explanations(self, images, labels):
        dl_attrs = np.concatenate([self.dl.attribute(images[i].unsqueeze(0), 
                                                    target=l, 
                                                    baselines=images[i].unsqueeze(0) * 0).detach().cpu().numpy() for i, l in enumerate(labels)], 0)
        return dl_attrs



class HyperbolicReasoning(Baseline):
    def __init__(self, inferer):
        # inferer is as instance of InductiveReasoningDT class in src.inference
        self.inferer = inferer 

    def get_explanations(self, images, labels, combine=True):
        attrs = [self.inferer.query(l, visual = img, local=True, return_plots = True) for l, img in zip(labels, images)]
        if not combine:
            return attrs
        else:
            return np.mean(attrs, 1)