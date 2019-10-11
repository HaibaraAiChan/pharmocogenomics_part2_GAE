import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import math


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    # return torch.FloatTensor(indices, values, shape)

def cosine_similarity(T1, T2):  # input T1, T2 should be tensorFloat dim*dim here

    similarities = F.cosine_similarity(T1, T2)
    # print(similarities)
    return np.mean(similarities.numpy(), 0)


def correlation_coefficient_tensor(T1, T2):
    T1 = T1.cpu().detach().numpy()
    T2 = T2.cpu().detach().numpy()
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result


def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result


def get_roc_score(emb, adj_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_reconstruct = np.dot(emb, emb.T)
    adj_orig = sp.coo_matrix(adj_orig)
    adj_orig = adj_orig.toarray()
    r = correlation_coefficient(adj_orig, adj_reconstruct)
    print(r)
    r_list = []
    r_list.append(r)

    roc_score = roc_auc_score([1*len(r_list)], r_list)
    ap_score = average_precision_score([1*len(r_list)], r_list)
    #
    return roc_score, ap_score


def draw_loss_curve(loss_list):

    plt.plot(loss_list)
    plt.show()
    return



if __name__ == "__main__":
    x = np.array([[0.1, 0.2, 0.5], [0.2, 1.0, 0.2]])
    y = np.array([[0.1, 0.2, 0.4], [0.2, 0.9, 0.2]])

    r = correlation_coefficient(x, y)
    print(r)
    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)
    # cos = cosine_similarity(X,Y)
    # print(cos)
    logsoftmax = torch.nn.LogSoftmax()
    loss = torch.mean(torch.sum(- logsoftmax(X) * r, 1))
    print(loss)
    entropy = torch.mean(torch.sum(- Y * logsoftmax(X), 1))
    print(entropy)