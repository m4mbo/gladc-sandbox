# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import data.load_data as load_data
import networkx as nx
from models.graph_autoencoder import *
import torch
import torch.nn as nn
import time
import models.graph_autoencoder as graph_autoencoder
from loss import *
from util import *
from torch.autograd import Variable
from data.graph_build import GraphBuild
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import cm
from models.model import *
from random import shuffle
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from mb.logger import LossTracker

def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='Tox21_HSE', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=2000, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=128, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=2, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=2, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def gen_ran_output(h0, adj, model, vice_model):

    # Adding noise to every parameter of vice_model except proj_head

    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + 1.0 * torch.normal(0,torch.ones_like(param.data)*param.data.std()).cuda()     
    x1_r,Feat_0= vice_model(h0, adj)
    return x1_r,Feat_0

def train(dataset, data_test_loader, NetG, noise_NetG, args, loss_tracker):    
    
    optimizerG = torch.optim.Adam(NetG.parameters(), lr=args.lr)
    epochs=[]
    auroc_final = 0
    node_Feat=[]
    graph_Feat=[]
    max_AUC=0

    # Logging losses
    reconstruction_loss = []
    contrastive_loss = []
    node_graph_loss = []
    total_loss = []
    test_loss = []

    for epoch in range(args.num_epochs):
        total_time = 0
        total_lossG = 0.0
        total_reconstruction_loss = 0.0
        total_contrastive_loss = 0.0
        total_node_graph_loss = 0.0
        total_test_loss = 0.0

        NetG.train()
        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            adj_label = Variable(data['adj_label'].float(), requires_grad=False).cuda()
            
            # x1_r -> node-level latent feature representation (array where each row corresponds to a node feature)
            # Feat_0 -> graph-level latent representation 
            # x1_r_1 -> randomized node-level latent representation
            # Feat_0_1 -> randomized graph-level latent representation
            # x_fake -> reconstructed node features
            # s_fake -> reconstructed adjacency matrix
            # x2 -> node-level latent feature representation of reconstructed feature array
            # Feat_1 -> graph-level latent representation of reconstructed adjacency matrix

            x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
            x1_r_1 ,Feat_0_1= gen_ran_output(h0, adj, NetG.shared_encoder, noise_NetG)
            x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)

            # 'err_g_con_s' and 'err_g_con_x' -> loss to measure how well the reconstruction matches the original 
            # 'node_loss' -> mse of latent node-level feature representations
            # 'graph_loss' -> mse of latent adjacency matrix representations
            # err_g_enc -> contrastive loss (ensures that model can distinguish between different views of the graph)

            err_g_con_s, err_g_con_x = loss_func(adj_label, s_fake, h0, x_fake)

            node_loss=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1).mean(dim=0)

            err_g_enc=loss_cal(Feat_0_1, Feat_0)

            # lossG = err_g_con_s + err_g_con_x + node_loss + graph_loss + err_g_enc
            
            lossG = err_g_con_s + err_g_con_x + graph_loss + node_loss

            optimizerG.zero_grad()
            lossG.backward()
          
            optimizerG.step()
          
            total_lossG += lossG.item()
            total_reconstruction_loss += (err_g_con_s + err_g_con_x).item()
            total_contrastive_loss += err_g_enc.item()
            total_node_graph_loss += (node_loss + graph_loss).item()
                        
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        # Appending the losses
        reconstruction_loss.append(total_reconstruction_loss / len(dataset))
        contrastive_loss.append(total_contrastive_loss / len(dataset))
        node_graph_loss.append(total_node_graph_loss / len(dataset))
        total_loss.append(total_lossG / len(dataset))

        if (epoch+1)%10 == 0 and epoch > 0:
            epochs.append(epoch)
            NetG.eval()   
            loss = []
            y=[]

            for batch_idx, data in enumerate(data_test_loader):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()

                x1_r,Feat_0 = NetG.shared_encoder(h0, adj)
            
                x_fake,s_fake,x2,Feat_1=NetG(x1_r,adj)
                
                loss_node=torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)

                loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
                
                loss_=loss_node+loss_graph

                total_test_loss += loss_.item()    # Logging

                loss_ = np.array(loss_.cpu().detach())
                
                loss.append(loss_)
                if data['label'] == 0:
                    y.append(1)
                else:
                    y.append(0)             

            test_loss.append(total_test_loss / len(data_test_loader))

            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            if test_roc_ab > max_AUC:
                max_AUC=test_roc_ab

        auroc_final = max_AUC

    loss_tracker.add_train_losses(reconstruction_loss, contrastive_loss, node_graph_loss, total_loss)
    loss_tracker.add_test_loss(test_loss)

    return auroc_final
    
if __name__ == '__main__':

    args = arg_parse()
    DS = args.DS
    setup_seed(args.seed)

    # Taking data as it is (with its previously defined split)
    # No fold procedure as in 'main_py'
    graphs_train_ = load_data.read_graphfile(args.datadir, args.DS+'_training', max_nodes=args.max_nodes)  
    graphs_test = load_data.read_graphfile(args.datadir, args.DS+'_testing', max_nodes=args.max_nodes)  
    datanum = len(graphs_train_) + len(graphs_test)    
    
    if args.max_nodes == 0:
        max_nodes_num_train = max([G.number_of_nodes() for G in graphs_train_])
        max_nodes_num_test = max([G.number_of_nodes() for G in graphs_test])
        max_nodes_num = max([max_nodes_num_train, max_nodes_num_test])
    else:
        max_nodes_num = args.max_nodes
        
    print("Number of graphs:",datanum)
    print("")
    
    num_trials = 5
    max_aurocs = []

    loss_tracker = LossTracker(DS)

    # 5 trials according to paper
    for i in range(num_trials):

        print("Trial {}:".format(i+1))   

        train_num=len(graphs_train_)
        all_idx = [idx for idx in range(train_num)]
        shuffle(all_idx)
        num_train=math.ceil(1*train_num)
        train_index = all_idx[:num_train]
        graphs_train_1 = [graphs_train_[i] for i in train_index]
        graphs_train = []

        # Taking abnormals out of training set

        for graph in graphs_train_1:
            if graph.graph['label'] == 0:
                graphs_train.append(graph)
        for graph in graphs_train:
            graph.graph['label'] = 0    # Not sure why
                
        graphs_test_nor = []
        graphs_test_ab = []

        # Not sure why

        for graph in graphs_test:
            if graph.graph['label'] == 0:
                graphs_test_nor.append(graph)
            else:
                graphs_test_ab.append(graph)
        for graph in graphs_test_nor:
            graph.graph['label'] = 0
        for graph in graphs_test_ab:
            graph.graph['label'] = 1
            graphs_test_nor.append(graph)
        graphs_test = graphs_test_nor
                    
        num_train = len(graphs_train)
        num_test = len(graphs_test)
        print("Train-test split:", num_train, num_test)
        print("")

            
        dataset_sampler_train = GraphBuild(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)

        NetG= NetGe1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()

   
        noise_NetG= Encoder1(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).cuda()
            
        #NetG= NetGe(dataset_sampler_train.feat_dim,args.hidden_dim, args.output_dim,args.dropout,args.batch_size).cuda()
        #noise_NetG= Encoder(dataset_sampler_train.feat_dim,args.hidden_dim, args.output_dim,args.dropout,args.batch_size).cuda() 
        
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                        shuffle=True,
                                                        batch_size=args.batch_size)

        
        dataset_sampler_test = GraphBuild(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                            shuffle=False,
                                                            batch_size=1)
        #train(data_train_loader, data_test_loader, model_teacher, model_student, args) 

        result = train(data_train_loader, data_test_loader, NetG, noise_NetG, args, loss_tracker)

        print("Max AUC:", result)
        print("")

        max_aurocs.append(result)
    
    print('Average: {}, Std: {}'.format(np.mean(max_aurocs), np.std(max_aurocs)))
    loss_tracker.plot_losses()
    loss_tracker.plot_final_losses()

    
    
    
