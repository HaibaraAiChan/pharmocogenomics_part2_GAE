"""
so far, August,7,2019
igraph 0.7.3
formula.py now is only compatible with python 3.6 below.(no python3.7)
many packages may be compatible with 3.7 in the future, reader can check current version
"""
import pandas as pd
from igraph import *
import numpy as np
from multiprocessing import Pool


def find_all_target_nodes(file):
    df = pd.read_csv(file, delimiter=',')
    uni_pro_list = df['Protein'].unique()
    return uni_pro_list


def shortest_path_4_one_node(start_node, df, out_folder):
    node_idx = g.vs.select(name_eq=start_node)
    if node_idx.__len__() == 0:
        return
    idx = [v.index for v in node_idx][0]
    sh_path = g.get_all_shortest_paths(v=idx, weights=g.es['weight'], mode=OUT)
    print(sh_path[0:10])
    """
    then calculate the weights sum of each path
    and sort them write to file for each node
    """
    vs = VertexSeq(g)

    sum_path_list = worker_1(sh_path, start_node, df, vs, out_folder)

    sum_path_list = sorted(sum_path_list, key=lambda x: x[2])
    print(sum_path_list)
    """
    write start node to all the nodes' shortest path sum to one csv file
    """
    df_out = pd.DataFrame(sum_path_list, columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq'])
    df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) + '.csv', sep=',', index=False)


def worker_1(path_list, start_node, df, vs, out_folder):
    sum_path_list = []
    num = 0
    for path in path_list:

        if len(path) == 1:
            continue
        sum_path = 0
        i = 1
        while i in range(len(path)):
            prev = path[i - 1]
            node_prev = vs[prev]['name']
            pos = path[i]
            node_pos = vs[pos]['name']
            tt = df.loc[((df.protein1 == node_prev) & (df.protein2 == node_pos)) | ((df.protein2 == node_prev) & (df.protein1 == node_pos))]
            distance = tt.iloc[0]['combined_score']
            sum_path = sum_path + (i / distance)

            i = i + 1
        dest_node_idx = path[-1]
        dest_node = g.vs[dest_node_idx]['name']
        # print(path)
        path_name_list = [g.vs[i]['name'] for i in path]
        # print(sum_path)
        sum_path_list.append((start_node, dest_node, sum_path, str(path_name_list)))
        num = num+1
        if num % 40 == 0:
            print(num)
    print('start node: '+ str(start_node)+'part end')
    df_out = pd.DataFrame(sum_path_list, columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq'])
    df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) +'.csv', sep=',', index=False)
    return sum_path_list

#
# def worker(path_list, idx, start_node, df, vs, out_folder):
#     sum_path_list = []
#     num =0
#     for path in path_list:
#         if len(path) == 1:
#             continue
#         sum_path = 0
#         i = 1
#         while i in range(len(path)):
#             prev = path[i - 1]
#             node_prev = vs[prev]['name']
#             pos = path[i]
#             node_pos = vs[pos]['name']
#             tt = df.loc[((df.protein1 == node_prev) & (df.protein2 == node_pos)) | ((df.protein2 == node_prev) & (df.protein1 == node_pos))]
#             distance = tt.iloc[0]['combined_score']
#             sum_path = sum_path + (i / distance)
#
#             i = i + 1
#         dest_node_idx = path[-1]
#         dest_node = g.vs[dest_node_idx]['name']
#         # print(path)
#         path_name_list = [g.vs[i]['name'] for i in path]
#         # print(sum_path)
#         sum_path_list.append((start_node, dest_node, sum_path, str(path_name_list)))
#         # num=num+1
#         # if num%10==0:
#         #     print(num)
#     print('start node: '+ str(start_node)+'_'+ str(idx)+'part end')
#     df_out = pd.DataFrame(sum_path_list, columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq'])
#     df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) + '_' + str(idx)+'.csv', sep=',', index=False)
#     # return sum_path_list
#
#
# def shortest_path_4_one_node(start_node, df, out_folder):
#     node_idx = g.vs.select(name_eq=start_node)
#     idx = [v.index for v in node_idx][0]
#     sh_path = g.get_all_shortest_paths(v=idx, weights=g.es['weight'], mode=OUT)
#     print(sh_path[0:10])
#     """
#     then calculate the weights sum of each path
#     and sort them write to file for each node
#     """
#     vs = VertexSeq(g)
#
#     sum_path_list = []
#     # size = len(sh_path)
#     size = 1
#     P_NUM = 1
#     p = Pool(P_NUM)
#     rest_list = []
#     if size % P_NUM != 0:
#         tail = size % P_NUM
#         tail_list = sh_path[-(tail+1):-1]
#         rest_list = worker(tail_list, -1, start_node, df, vs, out_folder)
#     # for i in range(P_NUM):
#     #     a = int(size / P_NUM * i)
#     #     b = int(size / P_NUM * (i + 1))
#     #     tmp_list = sh_path[size / P_NUM * i:size / P_NUM * (i + 1)]
#     #     p.apply_async(worker, args=(tmp_list, start_node, tuples, vs,))
#     for i in range(P_NUM):
#         a = size//P_NUM*i
#         b = size//P_NUM*(i+1)
#         tmp_list = sh_path[a: b]
#         p.apply_async(worker, args=(tmp_list, i, start_node, df, vs, out_folder))
#         # p.apply_async(worker, args=(sh_path[size//P_NUM*i:size//P_NUM*(i+1)], i, start_node, df, vs, out_folder))
#     p.close()
#     p.join()
#     # tmp = main_list[0].get()
#     # output = [p.get() for p in main_list]
#     # sum_path_list = output + rest_list
#     # sum_path_list = sorted(sum_path_list, key=lambda x: x[2])
#     # print(sum_path_list)
#     """
#     write start node to all the nodes' shortest path sum to one csv file
#     """
#     # df_out = pd.DataFrame(sum_path_list, columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq'])
#     # df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) + '.csv', sep=',', index=False)


if __name__ == '__main__':
    output_folder = '../../../data/index_ENSP/target_node_shortest_path/'
    file = '../../../data/index_ENSP/cellline_index.csv'
    protein_list = find_all_target_nodes(file)
    graph_file = 'graph.csv'
    # graph_file = '../../../data/index_ENSP/undirected_graph_index.csv'
    df = pd.read_csv(graph_file, delimiter=',')
    tup_tmp = list(df.itertuples(index=False, name=None))

    g = Graph.TupleList(tup_tmp, weights=True)
    g.write_pickle(fname='test_graph.pkl', version=-1)
    # g.write_pickle(fname='undirected_graph.pkl', version=-1)
    # g = Graph()
    # g = g.Read_Pickle('undirected_graph.pkl')

    tmp = 0
    for e in g.es:
        print(str(g.vs[e.tuple[0]]['name']) + ' ' + str(g.vs[e.tuple[1]]['name']) + ' ' + str(e['weight']))
        if tmp > 10:
            break
        tmp = tmp + 1
    print('end')

    shortest_path_4_one_node(protein_list[0], df, output_folder)

    # [shortest_path_4_one_node(start_node, df, output_folder) for start_node in protein_list]
    print('end of all')
