from pyexpat import features
from cv2 import threshold
import torch
import math
from torch import nn
import numpy as np
import os 
import pickle
import networkx as nx
import torch.nn.functional as F
import plotly.graph_objects as go
import networkx as nx


class visualSemantics(object):
    def __init__(self, 
                    decoder,
                    reasoning_graph = None,
                    feature_extractor = None,
                    codebook_sampler = None,
                    device=0,
                    threshold=0.2
                    ):

        self.device = device
        with torch.no_grad():
            self.decoder = decoder.to(self.device).eval() 
            self.feature_extractor = feature_extractor.to(self.device).eval()
            self.codebook_sampler = codebook_sampler.to(self.device).eval()
            for p in self.decoder.parameters(): p.requires_grad = False 
            for p in self.codebook_sampler.parameters(): p.requires_grad = False 
            for p in self.feature_extractor.parameters(): p.requires_grad = False 

        self.reasoning_nx = reasoning_graph
        self.threshold = threshold


    @torch.no_grad()
    def forward(self, x):
        f = self.feature_extractor(x)
        quant_loss, features, feature_idxs, ce , td, hrc, r = self.codebook_sampler(f)
        return features, feature_idxs


    @torch.no_grad()
    def recon(self, features):
        return self.decoder(features)

    def effect(self, features, symbols):
        xorig = self.recon(features)
        feature_clone = features.clone() 
        feature_clone[:, symbols, :, :] = 0
        x = self.recon(feature_clone)
        return (xorig - x)

    def mask(self, effect):
        return 1.*(effect > self.threshold)

    def semantics(self, x, node_name):

        if node_name not in self.reasoning_nx:
            print ("Insignificant Symbol, symbol ignored in reasoning")
            return

        def find_root(node):

            if isinstance(node, list):
                nodes = node 
            else:
                nodes = [node]
            
            parents = []
            for node in nodes:
                if self.G.predecessors(node):  #True if there is a predecessor, False otherwise
                    parents.extend(find_root(self.G.predecessors(node)))
                else:
                    parents.extend(node)
            return set(parents)


        root_symbols = [int(s.split('_')[-1]) for s in find_root(node_name)]
        
        features, feature_idxs = self.forward(x)
        filter_batch = [(root_symbols.all() in fidx) for fidx in feature_idxs]

        effect = self.effect(features[filter_batch, ...], root_symbols)
        return x[filter_batch, ...]*self.mask(effect)


    def visual_rule(self, x, rule):
        target_node, body_nodes = rule.split(' <- ')
        body_nodes = rule.split(',')

        if target_node not in self.reasoning_nx:
            target_visual = x 
        else:
            target_visual = self.semantics(x, target_node)

        body_visuals = [self.semantics(x, bnode) for bnode in body_nodes] 
        return target_visual, body_visuals