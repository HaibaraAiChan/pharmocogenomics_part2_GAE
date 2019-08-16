"""
so far, August,7,2019
igraph 0.7.3
formula.py now is only compatible with python 3.6 below.(no python3.7)
many packages may be compatible with 3.7 in the future, reader can check current version
"""
import argparse

import pandas as pd
from igraph import *
import numpy as np
from multiprocessing import Pool


def find_all_target_nodes(file):
    df = pd.read_csv(file, delimiter=',')
    uni_pro_list = df['Protein'].unique()
    return uni_pro_list


def worker(path_list, start_node, g):
    sum_path_list = []

    num = 0
    for path in path_list:
        if len(path) == 1:
            continue
        sum_path = 0
        w_list =[]
        for i in range(1, len(path)):
            edges = g.es.select(_within=(path[i - 1], path[i]))
            dist = [e['weight'] for e in edges][0]
            sum_path = sum_path + (i / dist)
            w_list.append((i/dist))

        dest_node = g.vs[path[-1]]['name']
        path_name_list = [g.vs[i]['name'] for i in path]
        sum_path_list.append((start_node, dest_node, sum_path, str(path_name_list), w_list))

        if num % 40 == 0:
            print(num)
        num = num + 1
    return sum_path_list


def shortest_path_4_one_node(start_node, g, out_folder):
    node_idx = g.vs.select(name_eq=int(start_node))
    if node_idx.__len__() == 0:
        print('error: start_node nis not in graph!')
        return
    idx = [v.index for v in node_idx][0]
    sh_path = g.get_all_shortest_paths(v=idx, weights=g.es['weight'], mode=OUT)
    print(sh_path[0:10])
    """
    then calculate the weights sum of each path
    and sort them write to file for each node
    """

    # sum_path_list = worker(sh_path,  start_node,  g)

    size = len(sh_path)

    P_NUM = 10
    p = Pool(P_NUM)
    rest_list = []
    if size % P_NUM != 0:
        tail = size % P_NUM
        tail_list = sh_path[-(tail + 1):-1]
        rest_list = worker(tail_list, start_node, g, )
        print('tail end')

    main_list = [p.apply_async(worker, args=(sh_path[size // P_NUM * i:size // P_NUM * (i + 1)], start_node, g,)) for i
                 in range(P_NUM)]
    p.close()
    p.join()

    output = [p.get() for p in main_list]
    flat_list = [item for sublist in output for item in sublist]
    sum_path_list = flat_list + rest_list
    sum_path_list = sorted(sum_path_list, key=lambda x: x[2])
    print(sum_path_list[0:10])
    """
    write start node to all the nodes' shortest path sum to one csv file
    """
    df_out = pd.DataFrame(sum_path_list,
                          columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq', 'weight_list'])
    df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) + '.csv', sep=',', index=False)


def read_file(filename):
    f = open(filename, 'r')
    list_ = [i.strip("\n").strip(" ") for i in list(f)]
    f.close()
    return list_


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outFolder',
                        required=False,
                        default='./target_node_shortest_path/',
                        help='path of output folder.')
    parser.add_argument('--infile',
                        required=True,
                        default='',
                        help='input file path and name.')
    parser.add_argument('--graph',
                        required=False,
                        default='./undirected_graph_index.csv',
                        help='graph file.')

    return parser.parse_args()


if __name__ == '__main__':
    # output_folder = '../../../data/index_ENSP/target_node_shortest_path/'
    # file = '../../../data/index_ENSP/cellline_index.csv'
    # # protein_list = find_all_target_nodes(file)
    # protein_list_file = './input/0.in'

    parse = getArgs()
    output_folder = parse.outFolder
    protein_list_file = parse.infile
    protein_list = read_file(protein_list_file)

    # graph_file = 'graph.csv'
    # graph_file = '../../../data/index_ENSP/undirected_graph_index.csv'
    # df = pd.read_csv(graph_file, delimiter=',')
    # tup_tmp = list(df.itertuples(index=False, name=None))

    # g = Graph.TupleList(tup_tmp, weights=True)
    # g.write_pickle(fname='test_graph.pkl', version=-1)
    # g.write_pickle(fname='undirected_graph.pkl', version=-1)
    g = Graph()
    g = g.Read_Pickle('undirected_graph.pkl')

    [shortest_path_4_one_node(start_node, g, output_folder) for start_node in protein_list]
    print('end of all')
