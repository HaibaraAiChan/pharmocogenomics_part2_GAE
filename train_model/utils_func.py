import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import pandas as pd
import time
import math


def make_one_graph(path_file_name):
    G = {}
    with open(path_file_name) as file:
        for row in file:
            r = row.strip().split('\t')
            label = r.pop(0)
            neighbors = {v: int(length) for v, length in [e.split(',') for e in r]}
            G[label] = neighbors
    return G


def prepare_graph_files(dataset, ofolder):
    ofolder = '../data/index_ENSP/graphs/'
    names = ['node', 'graph', 'gene', 'cellline']

    folder_files = os.listdir(dataset)
    node_file = [f for f in folder_files if ((names[0] in f) and (f.endswith('.csv')))][0]
    node_data = pd.read_csv(dataset + node_file, delimiter=',')
    node_length = node_data['Protein'].count()
    graph_file = [f for f in folder_files if ((names[1] in f) and (f.endswith('.csv')))][0]
    graph_data = pd.read_csv(dataset + graph_file, delimiter=',')
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    total_str = ''

    for i in range(node_length):

        tmp_df = graph_data.loc[graph_data['protein1'] == i]
        # tmp_df['distance'] = tmp_df.iloc[:, 2].apply(lambda x: 1 / x)
        # tt = 1/tmp_df['combined_score']
        tmp_df['distance'] = 1 / tmp_df['combined_score']
        tmp_df = tmp_df.sort_values(by=['distance'], ascending=True)
        tmp_df['out'] = tmp_df['protein2'].astype(str) + ',' + tmp_df['combined_score'].astype(str)

        out_str = str(i) + '\t' + '\t'.join(tmp_df['out'].tolist())
        # total_str = total_str + out_str
        # file = open(ofolder + str(i) + '.txt', "w")
        # file.write(out_str)
        # file.close()
        # print(out_str)
        file = open('../data/index_ENSP/total_graph.txt', "a")
        file.write(out_str + '\n')
        file.close()
        if (i % 10 == 0):
            print(i, '-' * 40)
        # print(out_str)

    return


def prepare_target_cell_line_files(dataset, ofolder):
    ofolder = '../data/index_ENSP/target_cell_line/'
    names = ['node', 'graph', 'gene', 'cellline']

    folder_files = os.listdir(dataset)
    gene_file = [f for f in folder_files if ((names[2] in f) and (f.endswith('.csv')))][0]
    cellline_file = [f for f in folder_files if ((names[3] in f) and (f.endswith('.csv')))][0]

    gene_data = pd.read_csv(dataset + gene_file, delimiter=',')
    cellline_data = pd.read_csv(dataset + cellline_file, delimiter=',')

    if not os.path.exists(ofolder):
        os.makedirs(ofolder)

    def target_gene(idx1, idx2, idx3):
        tmp = gene_data[(gene_data['Protein'] == idx1) & (gene_data['Drug'] == idx2) & (gene_data['Cell_line'] == idx3)]
        tmp.to_csv(ofolder + str(idx1) + '_' + idx2 + '_' + idx3 + '.csv', index=False)
        return ofolder + str(idx1) + '_' + idx2 + '_' + idx3 + '.csv'

    ttt = cellline_data
    start = time.time()
    ttt['features'] = ttt.apply(
        lambda row: target_gene(row['Protein'], row['Drug'], row['Cell_lines']), axis=1)
    end = time.time()
    print('elapse: ', str(end - start))
    return cellline_data.count


def load_data(dataset, unweight):
    # load the data: 'interactions', 'gene', 'cell-line'
    names = ['node', 'graph', 'gene', 'cellline']
    objects = []
    folder_files = os.listdir(dataset)
    index_file = [f for f in folder_files if names[0] in f][0]
    graph_file = [f for f in folder_files if names[1] in f][0]
    gene_file = [f for f in folder_files if names[2] in f][0]
    cellline_file = [f for f in folder_files if names[3] in f][0]

    print("loading data ", "*" * 20)
    print(graph_file)
    print(gene_file)
    print(cellline_file)
    graph_data = pd.read_csv(dataset + graph_file, delimiter=',')
    gene_data = pd.read_csv(dataset + gene_file, delimiter=',')
    cellline_data = pd.read_csv(dataset + cellline_file, delimiter=',')

    arr = graph_data.values
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sp.coo_matrix((arr[:, 2], (arr[:, 0], arr[:, 1])), shape=shape, dtype=arr.dtype)
    adj = coo.tocsr()

    # cellline_data['features'] = cellline_data.apply(lambda row: target_gene(row['Protein'], row['Drug'], row['Cell_lines']),axis=1)

    # features =
    return adj
    # return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


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


# def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])
#
#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])
#
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#
#     return roc_score, ap_score


if __name__ == "__main__":
    # x = np.array([[0.1, .32, .2, 0.4, 0.8], [.23, .18, .56, .61, .12]])
    # y = np.array([[2, 4, 0.1, .32, .2], [1, 3, .23, .18, .56]])
    x = np.array([[0.1, 0.2], [0.2, 1.0]])
    y = np.array([[0.1, 0.2], [0.2, 0.9]])


    r = correlation_coefficient(x, y)
    print(r)


    # input_path = '../../data/index_ENSP/GAE_INPUT/'
    # K = 10
    # input_folder = input_path + str(K) + '_neighbors/'
    # load_data(input_folder)

    # prepare_graph_files('../data/index_ENSP/', None)
    # graph_path = '../data/index_ENSP/graphs/0.txt'
    #
    # graph = make_one_graph(graph_path)
    # print(graph)
