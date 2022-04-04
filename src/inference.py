from importlib_metadata import distribution
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.DNF_Ex import DNF_Ex

from Lib.mylibw import read_by_tokens
from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import (accuracy_score, 
                                precision_recall_curve,
                                auc,
                                precision_recall_fscore_support,
                                average_precision_score,
                                log_loss,
                                roc_auc_score,
                                confusion_matrix)
import pandas as pd
import operator
import scipy.signal

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



class ILPExplainer(object):
    def __init__(self, boolean_data,
                        nhatoms,
                        nbatoms,
                        max_clause = 10,
                        lr = 1e-3,
                        nchain=7,
                        batch_size = 16,
                        nepochs = 200,
                        tpredicate = 1,
                        classmapping = None,
                        featuremapping = None
                        ):
        # boolean_data: background facts for ILP learning
        # nhatoms: total number of head atoms
        # nbatoms: total number of body atoms
        # max_clause: max. number of clause in ILP formulae
        # lr: learning rate
        # nchain: total number of forward chaining steps
        # batch_size: batch size for training
        # nepochs: total number of training epochs
        # tpredicate : target predicate to construct explanation rules
        # classmapping: idx to str mapping for head atoms
        # featuremapping: idx to str mapping for body atoms

        self.data = boolean_data
        self.nhatoms = nhatoms
        self.nbatoms = nbatoms
        self.max_clause = max_clause
        self.lr = lr 
        self.nchain = nchain
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.target_predicate = tpredicate
        self.classmapping = classmapping
        self.featuremapping = featuremapping

        if self.classmapping is None:
            self.classmapping = {k: 'class-{}'.format(k) for k in range(self.nhatoms)}

        if self.featuremapping is None:
            self.featuremapping = {k: 'class-{}'.format(k) for k in range(self.nbatoms)}



    def formatting(self, data):
        constants = {}
        
        for i in range(self.nbatoms):
            constants[self.featuremapping[i]] = [f'f{i}_{j}' for j in range(len(data))]
        
        for i in range(self.nhatoms):
            constants[self.classmapping[i]] = [f'c{i}_{j}' for j in range(len(data))]

        constants['X'] = [f'x_{j}' for j in range(len(data))]
        return constants

    def addBackground(self):
        pass

    def initialize(self):
        constants = self.formatting(self.data)
        predColl = PredCollection (constants)

        for i in range(self.nbatoms):
            predColl.add_pred(dname = f'fc_{i}', arguments =['X'], variables =['X'] )

        for i in range(self.nhatoms):
            predColl.add_pred(dname=f'fclass_{i}', 
                                arguments=['X'], 
                                variables=['X'],
                                pFunc = DNF(f'fclass_{i}',
                                            terms=self.max_clause, 
                                            init=[1,.1,0,.5],
                                            sig=1), 
                                use_neg=True, 
                                exc_conds=[('*','rep1') ], 
                                exc_terms=[],  
                                Fam='or',)
        predColl.initialize_predicates()  
        return predColl

    def represent(self, ):
        pass     

    
    def learn(self,):
        pass 


    def training_metrics(self, 
                            preds, 
                            targets):

        pass

