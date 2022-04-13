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
                    reasoning_graph,
                    feature_extractor = None,
                    codebook_sampler = None,
                    device=0,
                    threshold=0.2
                    ):

        self.device = device
        with torch.no_grad():
            self.decoder = decoder.to(self.device).eval() 
            self.reasoning_graph = reasoning_graph.to(self.device).eval()
            self.feature_extractor = feature_extractor.to(self.device).eval()
            self.codebook_sampler = codebook_sampler.to(self.device).eval()
            for p in self.decoder.parameters(): p.requires_grad = False 
            for p in self.reasoning_graph.parameters(): p.requires_grad = False 
            for p in self.codebook_sampler.parameters(): p.requires_grad = False 
            for p in self.feature_extractor.parameters(): p.requires_grad = False 

        self.reasoning_nx = self.define_nx_graph()
        self.threshold = threshold

    def define_nx_graph(self):
        G = nx.DiGraph()
        
        # create nodes 
        total_nodes = 0
        layers = self.reasoning_graph.layers
        for i in range(len(layers)):
            for j in range(layers[i][0].layer.weight.shape[0]):
                G.add_node(f'L{i}.C{j}', layer= i+1)
                total_nodes +=1
                
        # add edges
        for i in range(len(layers) - 1):
            for j in range(layers[i][0].layer.weight.shape[0]):
                for k in range(layers[i+1][0].layer.weight.shape[0]):
                    if layers[i+1].layer.weight[k][j].item() > 0:
                        G.add_edge(f'L{i}.C{j}', 
                                    f'L{i+1}.C{k}', 
                                    weight=1)


        # filter graph 
        for node in G.nodes:
            if (G.out_degree(node) == 0) and (G.in_degree(node) == 0):
                G.remove_node(node)

        pos_ = nx.multipartite_layout(G, scale=2000)
        # pos_ = nx.multipartite_layout(G, subset_key='layer', scale=2000)
        nx.draw(G, pos_, edge_color='b', width=1, with_labels = True, arrows=True, arrowsize=20, node_size=1000) 
        return G
    

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

    def semantics(self, x, symbol=0, layer=1):
        node_name = f'L{layer}.C{symbol}'

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


        root_symbols = [int(s.split('C')[-1]) for s in find_root(node_name)]
        
        features, feature_idxs = self.forward(x)
        filter_batch = [(root_symbols.all() in fidx) for fidx in feature_idxs]

        effect = self.effect(features[filter_batch, ...], root_symbols)
        return x[filter_batch, ...]*self.mask(effect)