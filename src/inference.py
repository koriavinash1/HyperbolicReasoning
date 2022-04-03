from importlib_metadata import distribution
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

import networkx as nx


class SymbolExtractor(object):
    def __init__(self,
                    data_loader, 
                    ncodebook_features,
                    nclasses,
                    feature_extractor = None,
                    modulator = None,
                    codebook = None,
                    classifier = None,
                    logs_root = '.'):
        self.dl = data_loader
        self.feature_extractor = feature_extractor
        self.modulator = modulator 
        self.codebook = codebook 
        self.classifier = classifier
        self.nclasses = nclasses
        self.ncodebook_features = ncodebook_features 
        self.logs_root = logs_root
        
        os.makedirs(self.logs_root, exist_ok=True)


    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.modulator(features)
        symbols, features = self.codebook(features)
        y = self.classifier(features)
        y = F.one_hot(torch.argmax(y, 1), num_classes=y.shape[1])
        return symbols.cpu().numpy(), y.cpu().numpy() 


    def marginal(self, data):
        # data dim: nsamples x (ncodebookfeatures + nclasses)
        # returns marignal probabilities for each feature and class
        __marginals__ = np.mean(data, 0)
        return __marginals__


    def single_conditional(self, data, nodes, parents):
        # nodes: list
        # parents: dict with variables and corresponding values
        # return <|nodes|>

        parents_idxs = parents.keys()
        parent_values = parents.values()
        x, y = np.where(data[:, parents_idxs] == parent_values)
        _data_ = data[x[0], :]
        return np.mean(_data_[:, nodes], 0)
        

    def conditional(self, data, nodes, parents):
        # nodes, parents:-> list
        # returns: probability distribution table 
        #   dim: 2^|parents| x |nodes|

        parent_values = [1, 0]*len(parents)
        parent_values = list(itertools.product(*parent_values))

        distribution = []
        for parent_value in parent_values:
            distribution.append(self.single_conditional(data, 
                                            nodes, 
                                            {k:v for k, v in zip(parents, parent_value)}))
        
        distribution = np.array(distribution)
        return distribution
        

    def generateBoolean(self):

        _path_ = os.path.join(self.logs_root, 'boolean_data.pickle')
        if os.path.exists(_path_):
            with open(_path_, 'rb') as _file_:
                data = pickle.load(_file_)
            
            assert data.shape[1] == (self.ncodebook_features + self.nclasses), \
                "Loaded data and defined number of features and classes doesn't match"
            return data

        features = []; labels = []
        for x, _, _ in self.dl:
            f, y = self.forward(x)
            features.extend(f)
            labels.extend(y)
        
        data = np.concatenate([features, labels], 1)
        
        with open(_path_, 'wb') as _file_:
            pickle.dump(data, _file_)

        return data 


