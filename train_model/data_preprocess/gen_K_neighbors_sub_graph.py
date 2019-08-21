"""
each cell-line target protein will generate one sub graph. (total num is the file num of target_cell_line folder)
each sub-graph has fixed number K neighbors
input: cell-line gene expression,
        nearest K neighbors
output: sub graph Adjacency Matrix,
        gene expression features

"""
import pandas as pd
import time
import argparse
import shutil
import os
import torch
import numpy as np
import pickle
from scipy.sparse import *
from scipy import *
import matplotlib.pyplot as plt

import networkx as nx

import pickle


def save_pickle(matrix, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder+filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix

"""
draw each Target protein-Drug-Cellline combination  a graph
"""
def draw_cell_graph(G,out,cell):
    common_nodes = [node for (node, dict) in G.nodes(data=True) if dict['expression'] == 0.0]
    p_nodes = [node for (node, dict) in G.nodes(data=True) if dict['expression'] == 1.0]
    n_nodes = [node for (node, dict) in G.nodes(data=True) if dict['expression'] == -1.0]
    # pos = nx.spring_layout(G)  # positions for all nodes
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # pos = nx.random_layout(G)
    pos = nx.spring_layout(G)
    # nx.draw(G, pos,
    #         # node_color='b',
    #         node_size=5,
    #         edgelist=edges,
    #         edge_width=1,
    #         edge_color=weights,
    #         # edge_cmap=plt.cm.gist_ncar)
    #         edge_cmap=plt.cm.hsv)
    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist=common_nodes, node_color='grey', alpha=0.4, node_size=10)
    nx.draw_networkx_nodes(G, pos, nodelist=p_nodes, node_color='r', alpha=1, node_size=15)
    nx.draw_networkx_nodes(G, pos, nodelist=n_nodes, node_color='b', alpha=1, node_size=15)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color=weights, edge_cmap=plt.cm.Pastel2)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                    width=6, alpha=0.5, edge_color='b', style='dashed')

    # labels
    # nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    #
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, font_size=2, font_color='r', edge_labels=labels)

    plt.axis('off')
    plt.show()

    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos)
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(out+cell[0:-4]+'.png')

"""
target node shortest path to all other nodes: generated Adj matrix  graph
"""
def draw_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.0]
    # pos = nx.spring_layout(G)  # positions for all nodes
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # pos = nx.random_layout(G)
    pos = nx.spring_layout(G)
    nx.draw(G, pos,
            node_color='b',
            node_size=5,
            edgelist=edges,
            edge_width=1,
            edge_color=weights,
            # edge_cmap=plt.cm.gist_ncar)
            edge_cmap=plt.cm.hsv)
    # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=10)

    # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                    width=6, alpha=0.5, edge_color='b', style='dashed')

    # labels
    # nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    #
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, font_size=2, font_color='r', edge_labels=labels)

    plt.axis('off')
    plt.show()
    # plt.savefig()

    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw(G, pos)
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.savefig( < wherever >)


def file_sort(str_list):
    res=[]
    for str in str_list:
        idx = int(str.split('_')[0])
        res.append((idx, str))

    def getKey(item):
        return item[0]
    res = sorted(res, key=getKey)
    result = [item[1] for item in res]
    return result


def path_file_sort(str_list):
    res=[]
    for str in str_list:
        idx = int(str.split('_')[-1][0:-4])
        res.append((idx, str))

    def getKey(item):
        return item[0]
    res = sorted(res, key=getKey)
    result = [item[1] for item in res]
    return result


def get_adj_matrix(spath_folder, path_file, K):
    df_data = pd.read_csv(spath_folder + path_file, delimiter=',')
    df_K = df_data[0:K]
    # print(df_K)
    """
    # features tensor list of list : K*1
    features = torch.FloatTensor(np.array(features.todense()))
    # adj is csr_matrix
    >>> row = array([0,0,1,2,2,2])
    >>> col = array([0,2,2,0,1,2])
    >>> data = array([1,2,3,4,5,6])
    >>> adj = csr_matrix( (data,(row,col)), shape=(K,K) ).todense()
    matrix([[1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]])
    38, 376, 0.0031847133757961785, "[38, 376]"
    38, 9475, 0.0032679738562091504, "[38, 9475]"
    38, 4456, 0.003289473684210526, "[38, 4456]"
    38, 11819, 0.003289473684210526, "[38, 11819]"
    38, 2585, 0.0033003300330033004, "[38, 2585]"
    38, 8139, 0.0033003300330033004, "[38, 8139]"
    38, 15717, 0.0033003300330033004, "[38, 15717]"
    38, 10176, 0.0033003300330033004, "[38, 10176]"
    38, 336, 0.0033112582781456954, "[38, 336]"
    """
    # row = array()
    # node_list = []
    # edge_list =[]
    # r, c = df_K.shape
    # assert r == K
    # for i in range(K):
    #     cur_r = df_K.iloc[i]
    #     print(cur_r)
    #     path = cur_r['shortest_path_seq'][1:-1].split(',')
    #     path_list = [int(i) for i in path]
    #     for node in path_list:
    #         if node not in node_list:
    #             node_list.append(node)
    #             if len(node_list) < 2:
    #                 continue
    #             edge_list.append((node_list[-2], node, ))

    G = nx.Graph()
    r, c = df_K.shape
    assert r == K
    for i in range(K):
        cur_r = df_K.iloc[i]
        # print(cur_r)
        path = cur_r['shortest_path_seq'][1:-1].split(',')
        path_list = [int(i) for i in path]
        weights = cur_r['weight_list'][1:-1].split(',')
        weights_list = [float(i) for i in weights]
        # if len(path_list) >= 3:
        #     print('path list')
        #     print(path_list)
        #     print('weight list')
        #     print(weights_list)
        for i in range(1, len(path)):
            G.add_edge(path_list[i - 1], path_list[i], weight=weights_list[i - 1])
    # draw_graph(G)
    adj = nx.adjacency_matrix(G)
    return adj, G


def data_prepare(K, cell_folder, spath_folder, output_folder):
    target_cell_file_list = [name for name in os.listdir(cell_folder) if os.path.isfile(cell_folder+name)]
    target_cell_file_list = file_sort(target_cell_file_list)
    path_file_list = [name for name in os.listdir(spath_folder) if os.path.isfile(spath_folder + name)]
    path_file_list = path_file_sort(path_file_list)

    for path_file in path_file_list:
        adj, G_K = get_adj_matrix(spath_folder, path_file, K)
        target_node_idx = path_file.split('_')[-1][0:-4]

        cur_folder = output_folder + str(target_node_idx) + '/'
        save_pickle(adj, cur_folder, str(target_node_idx)+'.pkl')

        cell_file_list = [cell_file for cell_file in target_cell_file_list if cell_file.split('_')[0] == target_node_idx]
        K_nodes = list(G_K.nodes)

        for cell_file in cell_file_list:
            G_C = G_K
            df = pd.read_csv(cell_folder + cell_file, delimiter=',')
            tup_list = []
            exp_dict = {}
            for node in K_nodes:
                exp = df[df['Gene'] == node]['Expression'].tolist()

                if len(exp) == 0:
                    exp.append(0)
                else:
                    print(exp)

                tup_list.append((node, exp[0]))
                exp_dict[node] = exp[0]
            df_o = pd.DataFrame(tup_list, columns=['node', 'expression'])
            df_o.to_csv(cur_folder+cell_file, sep=',', index=False)
            # nx.set_node_attributes(G_C, exp_dict, 'expression')
            # draw_cell_graph(G_C, cur_folder, cell_file)
            # print('drawing end')

    return


def process_data(data_file, uni_file, folder, label_flag):
    df_data = pd.read_csv(data_file, delimiter=',')
    df_uni = pd.read_csv(uni_file, delimiter=',')
    df_uni = df_uni.set_index('Protein')
    key_dict = df_uni.to_dict()
    return


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-K',
                        required=False,
                        type=int,
                        default=10,
                        help='the fixed number neighbors of target protein.')
    parser.add_argument('-cellFolder',
                        required=False,
                        default='../../../data/index_ENSP/target_cell_line/',
                        help='input target cell-line folder.')
    parser.add_argument('-spathFolder',
                        required=False,
                        default='../../../data/index_ENSP/target_node_shortest_path/',
                        help='all target protein shortest path folder.')
    parser.add_argument('-outFolder',
                        required=False,
                        default='../../../data/index_ENSP/GAE_INPUT/',
                        help='the fixed number neighbors of target protein.')
    return parser.parse_args()


if __name__ == "__main__":
    parse = getArgs()
    K = parse.K
    cell_folder = parse.cellFolder
    spath_folder = parse.spathFolder
    outFolder = parse.outFolder
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    output_folder = outFolder + str(K) + '_neighbors/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    cell_size = len([name for name in os.listdir(cell_folder) if name])
    spath_size = len([i for i in os.listdir(spath_folder) if os.path.isfile(spath_folder+i)])

    if cell_size == 0:
        print('error: cell line folder is empty')

    if spath_size == 0:
        print('error: shortest path folder is empty')

    start = time.time()
    data_prepare(K, cell_folder, spath_folder, output_folder)
    end = time.time()
    print('vector time elapsed :' + str(end - start))
