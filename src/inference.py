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

from visual_semantics import visualSemantics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

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





class InductiveReasoningDT(object):
    def __init__(self,  
                    data,
                    ncodebook_features,
                    nclasses,
                    feature_extractor = None,
                    codebook = None,
                    classifier = None,
                    decoder = None,
                    logs_root = '.',
                    device = 0,
                    mask_threshold=0.2,
                    class_mapping=None):

        self.ncodebook_features = ncodebook_features
        self.nclasses = nclasses
        self.Xdata = data[0]
        self.Ydata = data[1]

        if isinstance(ncodebook_features, list):
            assert self.ncodebook_features[-1] == self.nclasses, "reasoning block count mismatch"

        self.logs_root = logs_root

        with torch.no_grad():
            self.decoder = decoder.to(self.device).eval() 
            self.feature_extractor = feature_extractor.to(self.device).eval()
            self.codebook_sampler = codebook.to(self.device).eval()
            self.qclassifier = classifier.to(self.device).eval()
            
            for p in self.decoder.parameters(): p.requires_grad = False 
            for p in self.qclassifier.parameters(): p.requires_grad = False 
            for p in self.codebook_sampler.parameters(): p.requires_grad = False 
            for p in self.feature_extractor.parameters(): p.requires_grad = False 


        os.makedirs(os.path.join(self.logs_root, 'Logs'), exist_ok=True)

        self.nx_graph = self.define_nx_graph()
        self.vsemantics = visualSemantics(decoder=decoder,
                                        reasoning_graph=self.nx_graph,
                                        feature_extractor=self.feature_extractor,
                                        codebook_sampler=self.codebook_sampler,
                                        device=device,
                                        threshold=mask_threshold)

        self.class_mapping = class_mapping
        if not (self.class_mapping is None):
            self.class_mapping = {i: f'class-{i}' for i in range(self.nclasses)}


    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        loss, sampled_features, _, sampled_symbols, *_ = self.codebook(features)
        y = self.qclassifier(sampled_features)
        y = torch.argmax(y, 1)
        return [ss.cpu().numpy() for ss in sampled_symbols], \
                    y.cpu().numpy(), \
                    [sf.cpu().numpy() for sf in sampled_features]


    def define_nx_graph(self):
        G = nx.DiGraph()
        
        # create nodes 
        total_nodes = 0
        layers = []
        for i in range(len(self.ncodebook_features)):
            if hasattr(self.codebook_sampler, f'rattn{i}'):
                layers.append(getattr(self.codebook_sampler, f'rattn{i}'))


        for i in range(len(layers)):
            for j in range(layers[i][0].layer.weight.shape[0]):
                G.add_node(f'Cb^{i}_{j}', layer= i+1)
                total_nodes +=1
                
        # add edges
        for i in range(len(layers) - 1):
            for j in range(layers[i][0].layer.weight.shape[0]):
                for k in range(layers[i+1][0].layer.weight.shape[0]):
                    if layers[i+1].layer.weight[k][j].item() > 0:
                        G.add_edge(f'Cb^{i}_{j}', 
                                    f'Cb^{i+1}_{k}', 
                                    weight=1)


        # filter graph 
        for node in G.nodes:
            if (G.out_degree(node) == 0) and (G.in_degree(node) == 0):
                G.remove_node(node)

        pos_ = nx.multipartite_layout(G, scale=2000)
        # pos_ = nx.multipartite_layout(G, subset_key='layer', scale=2000)
        nx.draw(G, pos_, edge_color='b', width=1, with_labels = True, arrows=True, arrowsize=20, node_size=1000) 
        return G
    

    def get_class_symbol(self, class_idx):
        xbatch = self.Xdata[self.Ydata == class_idx][:10]
        sampled_symbols, y, sampled_features = self.forward(xbatch)
        return torch.argmax(torch.sum(sampled_symbols[-1], 1)[y == class_idx], 1)


    def query(self, class_idx, visual=None):
        class_symbol = self.get_class_symbol(class_idx)
        node_name = f'Cb^{len(self.ncodebook_features)-1}_{class_symbol}'

        heirarchical_rules = [[self.class_mapping[class_symbol] + ' <- ' + node_name]]

        nodes = [node_name]
        layer_depths = np.arange(1, len(self.ncodebook_features) - 1)
        for i in layer_depths[::-1]:
            rules = []; new_nodes = []
            for node in nodes:
                parents = self.nx_graph.predecessors(node)
                rule = node + " <- "
                for pnode in parents:
                    rule += pnode + ','

                rules.append(rule)
                new_nodes.extend(parents)

            nodes = new_nodes
            heirarchical_rules.append(rules)


        # print rules
        for i, hrules in enumerate(heirarchical_rules):
            print ("*************** Level-{} heirarchy **************".format(i))
            for rule in hrules:
                print(rule)
                if not (visual is None):
                    target, body = self.vsemantics.visual_rule(rule)
                    self.show(target, body)

        return heirarchical_rules
    
    
    def show(self, target, body, save_path=None):

        def check(image, normalize=True):
            if image.dtype == np.float32:
                if normalize:
                    image = np.uint8(255*((image - image.min())/(image.max() - image.min())))# * 127.5 + 127.5)
                else:
                    image = image 
                    
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = image [:, :, 0]
            return image

        target = check(target.squeeze().cpu().numpy())
        body = [check(b.squeeze().cpu().numpy()) for b in body]
        
        nimgs = len(body) + 1
        img_size = 3
        nrows = 1
        ncols = nimgs

        plt.clf()
        fig = plt.figure(num = nimgs, figsize=(img_size*ncols, img_size*nrows))
        fig.tight_layout()
        
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=0.2, hspace=0.02)

        ax = plt.subplot(gs[0, 0])
        ax.imshow(target)

        for xidx in range(len(body)):
            ax = plt.subplot(gs[xidx + 1, 0])
            ax.imshow(body[xidx])

        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_title("visual rule")
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    
        plt.tight_layout()
        if not (save_path is None):
            plt.savefig(save_path)
        plt.show()
     