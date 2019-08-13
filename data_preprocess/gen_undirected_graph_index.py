"""
from original directed graph to undirected graph
A<-->B  to  A--B

"""
import pandas as pd
from igraph import *
import numpy as np


def read2tuple(filename):
    data_set = pd.read_csv(filename, delimiter=',')
    tuples = list(data_set.itertuples(index=False, name=None))
    y = np.unique(tuples, axis=0)
    z = []
    for i in y:
        z.append(tuple(i))
    # y = np.sort(tuples, axis=0)
    tuples = sorted(z, key=lambda x: (x[0], x[1]))

    return tuples


if __name__ == '__main__':

    # graph_file = '../../../data/index_ENSP/graph_index.csv'
    # data_set = pd.read_csv(graph_file, delimiter=',')
    # data_set.drop_duplicates()
    # tup_tmp = list(data_set.itertuples(index=False, name=None))
    #
    # print('original tuples length: ' + str(len(tup_tmp)))
    # list_of_list = [list(elem) for elem in tup_tmp]
    #
    # # def exchange(a, b):
    # #     tmp = a
    # #     a = b
    # #     b = tmp
    #
    # #
    # for item in list_of_list:
    #     if item[0] < item[1]:
    #         continue
    #     tmp = item[0]
    #     item[0] = item[1]
    #     item[1] = tmp
    #     # [exchange(item[0], item[1]) for item in tup_tmp if (item[0] > item[1])]
    #
    # # y = list(set(tup_tmp))
    # tuples = [tuple(l) for l in list_of_list]
    # y = np.unique(tuples, axis=0)
    # print('changed tuples length' + str(len(y)))
    #
    # df = pd.DataFrame(y, columns=['protein1', 'protein2', 'combined_score'])
    # df.to_csv('../../../data/index_ENSP/undirected_graph_index.csv', sep=',', index=False)

    graph_file = './index_ENSP/undirected_graph_index.csv'
    data_set = pd.read_csv(graph_file, delimiter=',')
    data_set.drop_duplicates()

    tup_tmp = list(data_set.itertuples(index=False, name=None))
    print(len(tup_tmp))
    tuples = sorted(tup_tmp, key=lambda x: (x[0], x[1]))
    df = pd.DataFrame(tuples, columns=['protein1', 'protein2', 'combined_score'])
    df.to_csv('./index_ENSP/undirected_graph_index.csv', sep=',', index=False)
