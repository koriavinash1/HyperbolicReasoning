from lib2to3.pygram import Symbols
from importlib_metadata import distribution
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy import stats
from tqdm import tqdm


from src.visual_semantics import visualSemantics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import pandas as pd
import operator
import scipy.signal

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from matplotlib.colors import LinearSegmentedColormap

def get_transparent_cmap(cmap):
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=cmap, colors=color_array)


    return map_object



class InductiveReasoningDT(object):
    def __init__(self,  
                    data,
                    ncodebook_features,
                    nclasses,
                    feature_extractor = None,
                    train_loader = None,
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
        self.device = device
        self.threshold = 0.3

        # if isinstance(ncodebook_features, list):
        #     assert self.ncodebook_features[-1] == self.nclasses, "reasoning block count mismatch"

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
        print (self.nx_graph.number_of_edges())
        if not (train_loader is None):
            self.filter_grpah(train_loader)
            print (self.nx_graph.number_of_edges())

        self.vsemantics = visualSemantics(decoder=decoder,
                                        reasoning_graph=self.nx_graph,
                                        feature_extractor=self.feature_extractor,
                                        codebook_sampler=self.codebook_sampler,
                                        device=device,
                                        threshold=mask_threshold)

        self.class_mapping = class_mapping
        if (self.class_mapping is None):
            self.class_mapping = {i: f'class-{i}' for i in range(self.nclasses)}


    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        _, _, sampled_features, sampled_symbols, *_ = self.codebook_sampler(features)

        cls_features = sampled_features[-1]
        cls_features = torch.mean(cls_features.view(cls_features.shape[0], 
                                                    cls_features.shape[1], 
                                                    cls_features.shape[2]*\
                                                    cls_features.shape[3]), 2)
        y = self.qclassifier(cls_features.view(cls_features.shape[0], -1))
        y = torch.argmax(y, 1)
        return [ss[1].cpu().numpy() for ss in sampled_symbols], \
                    y.cpu().numpy(), \
                    [sf.cpu().numpy() for sf in sampled_features]


    def define_nx_graph(self):
        G = nx.DiGraph()
        
        # create nodes 
        total_nodes = 0
        layers = self.codebook_sampler.reasoningLayers

        for i in range(len(layers)):
            weight = layers[i].weight.T
           
            for j in range(weight.shape[0]):
                G.add_node(f'$\zeta^{{{i}}}_{{{j}}}$', layer= i+1)
                total_nodes +=1

        for k in range(weight.shape[1]):
            G.add_node(f'$\zeta^{{{i+1}}}_{{{k}}}$', layer= i+2)
            total_nodes +=1



        # add edges
        for i in range(len(layers)):
            weight = layers[i].weight.T
            for j in range(weight.shape[0]):
                for k in range(weight.shape[1]):
                    if weight[j][k].item() > 0:
                        G.add_edge(f'$\zeta^{{{i}}}_{{{j}}}$', 
                                    f'$\zeta^{{{i+1}}}_{{{k}}}$', 
                                    weight=1)


        # filter graph 
        list_ = list(G.nodes)
        for node in list_:
            if (G.out_degree(node) == 0) and (G.in_degree(node) == 0):
                G.remove_node(node)

        pos_ = nx.multipartite_layout(G, subset_key='layer', scale=2)
        # pos_ = graphviz_layout(G, prog="twopi")
        plt.figure(figsize=(20, 20))
        nx.draw(G, pos_, edge_color='b', with_labels = True, node_size=1000) 
        plt.savefig('global-tree.png')
        return G
    

    def filter_grpah(self, train_loader):
        sampled_symbols = {i:[] for i in range(len(self.ncodebook_features))}

        for i, (xdata, ydata, *_) in tqdm(enumerate(train_loader)):
            if i > 1000: break
            symbols,  _, _ = self.forward(xdata.cuda(0))
            for i in range(len(symbols)):
                sampled_symbols[i].extend(symbols[i])
        

        for k, v in sampled_symbols.items():
            s = np.array(v).flatten()
            mode = stats.mode(s)
            mode_count = np.sum(s == mode)
            unique = np.unique(s)
            for us in unique:
                us_count = np.sum(s == us)
                if us_count < self.threshold * mode_count:
                    node = f'$\zeta^{{{k}}}_{{{us}}}$'
                    if node in list(self.nx_graph.nodes):
                        self.nx_graph.remove_node(node)


    def get_class_symbol(self, class_idx):
        xbatch = self.Xdata[self.Ydata == class_idx]
        sampled_symbols, y, sampled_features = self.forward(xbatch)
        filtered_symbols = (sampled_symbols[-1]).flatten() #TODO: filtering
        mode = stats.mode(filtered_symbols)
        mode_count = np.sum(filtered_symbols == mode)
        return_symbols = []
        for symbol in np.unique(filtered_symbols):
            symbol_count = np.sum(filtered_symbols == symbol)
            if symbol_count > 0.5*mode_count:
                return_symbols.append(symbol)
        return return_symbols


    def get_class_tree(self, class_idx, savepath='.'):
        G = nx.DiGraph()
        max_layers = len(self.ncodebook_features)+1
        os.makedirs(savepath, exist_ok=True)

        G.add_node(self.class_mapping[class_idx], layer= max_layers)

        class_symbols = self.get_class_symbol(class_idx)
        node_names = [f'$\zeta^{{{len(self.ncodebook_features)-1}}}_{{{class_symbol}}}$' for class_symbol in class_symbols]

        
        for node in node_names:
            G.add_node(node, layer=max_layers - 1)
            G.add_edge(node, self.class_mapping[class_idx], weight=1)
       
        for i in range(len(self.ncodebook_features)):
            new_nodes = []
            for node in node_names:
                parents = list(self.nx_graph.predecessors(node))
                new_nodes.extend(parents)

                for pnode in parents:
                    G.add_node(pnode, layer= max_layers  - 2 - i)
                    G.add_edge(pnode, node, weight=1)

            node_names = new_nodes
        

        # filter graph 
        list_ = list(G.nodes)
        for node in list_:
            if (G.out_degree(node) == 0) and (G.in_degree(node) == 0):
                G.remove_node(node)

        pos_ = nx.multipartite_layout(G, subset_key='layer', scale=2)
        # pos_ = graphviz_layout(G, prog="twopi")
        plt.figure(figsize=(4, 4))
        nx.draw(G, pos_, edge_color='b', with_labels = True, node_size=1000) 
        plt.savefig(os.path.join(savepath, f'class-{class_idx}-tree.png'))
        return G


    def get_local_tree(self, class_idx, x, savepath='.'):
        G = nx.DiGraph()
        sampled_symbols, _, _ = self.forward(x)
        os.makedirs(savepath, exist_ok=True)

        max_layers = len(self.ncodebook_features)+1
        G.add_node(self.class_mapping[class_idx], layer= max_layers)


        class_symbols = self.get_class_symbol(class_idx)
        class_symbols = [symbol for symbol in class_symbols if symbol in sampled_symbols[-1]]
        node_names = [f'$\zeta^{{{len(self.ncodebook_features)-1}}}_{{{class_symbol}}}$' for class_symbol in class_symbols]        
        
        for node in node_names:
            G.add_node(node, layer=max_layers - 1)
            G.add_edge(node, self.class_mapping[class_idx], weight=1)


        sampled_symbols = sampled_symbols[:-1]
        for i, layer_symbols in enumerate(sampled_symbols[::-1]):
            layer_symbols = np.unique(layer_symbols[0])
            new_nodes = []
            for node in node_names:
                parents = list(self.nx_graph.predecessors(node))
                print (node, parents, layer_symbols)
                for pnode in parents:
                    if int(pnode.split('_')[-1][1:-2]) in layer_symbols:
                        G.add_node(pnode, layer= max_layers  - 2 - i)
                        G.add_edge(pnode, node, weight=1)
                        new_nodes.append(pnode)

            node_names = new_nodes
        

        # filter graph 
        list_ = list(G.nodes)
        for node in list_:
            if (G.out_degree(node) == 0) and (G.in_degree(node) == 0):
                G.remove_node(node)


        pos_ = nx.multipartite_layout(G, subset_key='layer', scale=2)
        # pos_ = graphviz_layout(G, prog="twopi")
        plt.figure(figsize=(4, 4))
        nx.draw(G, pos_, edge_color='b', with_labels = True, node_size=1000) 
        plt.savefig(os.path.join(savepath, 'local-tree.png'))
        return G


    def query(self, class_idx, visual=None, local = False, overlay=False, save_path=None):
        if local:
            graph = self.get_local_tree(class_idx, visual)
        else:
            graph = self.get_class_tree(class_idx)

        os.makedirs(save_path, exist_ok=True)

        node_names = [self.class_mapping[class_idx]]
        heirarchical_rules = []

        for i in range(len(self.ncodebook_features)):
            rules = []; new_nodes = []
            for node in node_names:
                parents = list(graph.predecessors(node))
                rule = node + " <- "
                for pnode in parents:
                    rule += pnode + ','

                if len(parents) > 0:
                    rules.append(rule[:-1])
                    new_nodes.extend(parents)

            node_names = new_nodes
            heirarchical_rules.append(rules)


        # print rules
        for i, hrules in enumerate(heirarchical_rules):
            print ("*************** Level-{} heirarchy **************".format(i))
            for rule in hrules:
                print(rule)
                if not (visual is None):
                    target, body = self.vsemantics.visual_rule(visual, rule)
                    cmap = None if (i == 0) else get_transparent_cmap('bwr')
                    if overlay:
                        self.show(rule, target, body, cmap=cmap, overlay=visual.squeeze(0).cpu().numpy(), save_path=save_path)
                    else:
                        self.show(rule, target, body, cmap=cmap, save_path=save_path)

        return heirarchical_rules
    
    
    def show(self, rule, target, body, save_path=None, cmap='coolwarm', overlay=None):
        
        def check(image, normalize=True):
            if image.dtype == np.float32:
                if normalize:
                    image = np.uint8(255*((image - image.min())/(image.max() - image.min())))# * 127.5 + 127.5)
                else:
                    image = image 
                    
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                # image = image [:, :, 0]
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

        plt.suptitle(rule)
        ax = plt.subplot(gs[0, 0])
        if (target.shape[-1] != 3) and (not (overlay is None)) and (not (cmap is None)):
            ax.imshow(check(overlay))
        ax.imshow(target, cmap=cmap, alpha=1.0 if ((overlay is None) or (cmap is None)) else 0.8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
    


        for xidx in range(len(body)):
            ax = plt.subplot(gs[0, xidx + 1])
            if not (overlay is None):
                ax.imshow(check(overlay))
            ax.imshow(body[xidx], cmap=get_transparent_cmap('bwr'), alpha=1.0 if overlay is None else 0.8)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        
        plt.tight_layout()
        if not (save_path is None):
            plt.savefig(os.path.join(save_path, f'{rule}.png'))
        plt.show()
     