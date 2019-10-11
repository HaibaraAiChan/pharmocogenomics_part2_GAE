from __future__ import division
from __future__ import print_function

import argparse
import os
from data_generator import DataGenerator
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
#
from model import GCNModelVAE
from optimizer import loss_function
from utils import  preprocess_graph, get_roc_score
    # ,mask_test_edges, preprocess_graph, get_roc_score


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--K', type=int, default=10, help='the nearest K neighbors')
    parser.add_argument('--f_length', type=int, default=100, help='the length of feature vector')
    parser.add_argument('--dataset', type=str, default='../../../data/index_ENSP/GAE_INPUT/', help='index ENSP folder.')
    parser.add_argument('--outfolder', type=str, default='../../../data/index_ENSP/GAE_OUTPUT/', help='model output folder.')
    args = parser.parse_args()
    return args

# def gae_for(args):
#     print("Using {} dataset".format(args.dataset))
#     unweight = True
#     adj, features = load_data(args.dataset, unweight)
#     n_nodes, feat_dim = features.shape
#
#     # Store original adjacency matrix (without diagonal entries) for later
#     adj_orig = adj
#     tmp_debug1 = adj_orig.diagonal()
#     tmp_debug2 = tmp_debug1[np.newaxis, :]
#     tmp_debug3 = sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#     adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#     adj_orig.eliminate_zeros()
#
#     # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#     # adj = adj_train
#
#     # # Some preprocessing
#     # adj_norm = preprocess_graph(adj)
#     # adj_label = adj_train + sp.eye(adj_train.shape[0])
#     # # adj_label = sparse_to_tuple(adj_label)
#     # adj_label = torch.FloatTensor(adj_label.toarray())
#     #
#     # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
#     #
#     # model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
#     # optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     #
#     # hidden_emb = None
#     # for epoch in range(args.epochs):
#     #     t = time.time()
#     #     model.train()
#     #     optimizer.zero_grad()
#     #     recovered, mu, logvar = model(features, adj_norm)
#     #     loss = loss_function(preds=recovered,
#     #                          labels=adj_label,
#     #                          mu=mu,
#     #                          logvar=logvar,
#     #                          n_nodes=n_nodes,
#     #                          norm=norm,
#     #                          pos_weight=pos_weight)
#     #     loss.backward()
#     #     cur_loss = loss.item()
#     #     optimizer.step()
#     #
#     #     hidden_emb = mu.data.numpy()
#     #     roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
#     #
#     #     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
#     #           "val_ap=", "{:.5f}".format(ap_curr),
#     #           "time=", "{:.5f}".format(time.time() - t)
#     #           )
#     #
#     # print("Optimization Finished!")
#     #
#     # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
#     # print('Test ROC score: ' + str(roc_score))
#     # print('Test AP score: ' + str(ap_score))

def load_list(data_folder, K):
    sub_adj_path_list = []
    sub_feat_path_list =[]
    folder_list = os.listdir(data_folder)
    folder_list.sort()
    print(folder_list)
    K_folder = [f for f in folder_list if str(K) in f][0]
    print(K_folder)
    K_path = data_folder + K_folder

    for folder_name in os.listdir(data_folder + K_folder):
        adj_path = data_folder + K_folder+"/" + folder_name+'/'
        feature_path = K_folder+"/" + folder_name+'/'
        K_adj_file_path = K_folder+"/" + folder_name+'/'+folder_name+'.pkl'

        n_f = len(os.listdir(adj_path))-1
        Adj_file_list = [K_adj_file_path]*n_f

        num_file = 0
        feature_list = []
        for filename in os.listdir(adj_path):
            if '.pkl' in filename:
                continue

            feature_list.append(feature_path+filename)
            # Adj_file_list.append(K_adj_file_path)

            num_file = num_file + 1
            print(num_file, end=' ')
        if num_file == n_f:
            print(' ' + K_adj_file_path + 'is done')
            sub_adj_path_list = sub_adj_path_list + Adj_file_list
            sub_feat_path_list = sub_feat_path_list + feature_list
    print('adj and feature list are done')
    list_t = [list(a) for a in zip(sub_adj_path_list, sub_feat_path_list)]
    return list_t


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True
    args = args()

    path = args.dataset
    output = args.outfolder
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    dim = args.K + 1
    feature_length = args.f_length

    cnt = 0

    adj_feat_list = load_list(args.dataset, args.K)

    partition = {"train": adj_feat_list}

    # Parameters
    train_params = {'dim': dim,
                    'adj_dim':  (dim, dim),
                    'feature_dim':  (dim, 1),
                    'batch_size': batch_size,
                    'shuffle': True,
                    'path': path}
    training_generator = DataGenerator(partition['train'],  **train_params)
    print("the training data is ready")
    for epoch in range(epochs):
    # Training
        for adj_bs, features_bs in training_generator:
        # Transfer to GPU
        #     local_batch = local_batch.cuda()

            n_nodes = np.array(features_bs).shape[0]
            feat_dim = batch_size
            adj_norm = preprocess_graph(adj_bs)
            adj_label = adj_bs + sp.eye(adj_bs.shape[0])
            # adj_label = sparse_to_tuple(adj_label)
            adj_label = torch.FloatTensor(adj_label.toarray())
            # adj_label = torch.LongTensor(adj_label.toarray())

            pos_weight = float(adj_bs.shape[0] * adj_bs.shape[0] - adj_bs.sum()) / adj_bs.sum()
            norm = adj_bs.shape[0] * adj_bs.shape[0] / float((adj_bs.shape[0] * adj_bs.shape[0] - adj_bs.sum()) * 2)

            model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            hidden_emb = None
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features_bs, adj_norm)
            loss = loss_function(preds=recovered,
                             labels=adj_label,
                             mu=mu,
                             logvar=logvar,
                             n_nodes=n_nodes,
                             norm=norm,
                             pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()
            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_bs)


