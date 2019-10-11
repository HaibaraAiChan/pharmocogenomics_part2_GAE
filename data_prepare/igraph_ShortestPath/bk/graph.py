"""
so far, August,7,2019
igraph 0.7.3
formula.py now is only compatible with python 3.6 below.(no python3.7)
many packages may be compatible with 3.7 in the future, reader can check current version
"""
import pandas as pd
from igraph import *

def read2tuple(filename):
    data_set = pd.read_csv(filename, delimiter=',')
    print(data_set)
    tuples = list(data_set.itertuples(index=False, name=None))
    return tuples


if __name__ == '__main__':
    filename = 'graph.csv'
    start_node = 1759
    # filename = '../../data/index_ENSP/graph_index.csv'
    tup = read2tuple(filename)
    print(tup)
    g = Graph.TupleList(tup, weights=True)
    print(g)
    # print(g.es.attributes())
    print(g.es.select(weight_gt=173))
    print(g.es['weight'])
    for e in g.es:
        print(str(g.vs[e.tuple[0]]['name'])+' ' + str(g.vs[e.tuple[1]]['name'])+' ' + str(e['weight']))
        # print(e['weight'])
    print('end')
    node_idx = g.vs.select(name_eq=start_node)
    idx = [v.index for v in node_idx][0]
    sh_path = g.get_all_shortest_paths(v=idx, weights=g.es['weight'], mode=OUT)
    print(sh_path)
    """
    then calculate the weights sum of each path
    and sort them write to file for each node
    """
    vs = VertexSeq(g)

    sum_path_list = []
    for path in sh_path:
        if len(path) == 1:
            continue
        sum_path = 0
        i = 1
        while i in range(len(path)):
            prev = path[i-1]
            node_prev = vs[prev]['name']
            pos = path[i]
            node_pos = vs[pos]['name']
            distance = [item[2] for item in tup if((item[0] == node_prev and item[1] == node_pos) or (item[0] == node_pos and item[1] == node_prev))]

            sum_path = sum_path + (i/distance[0])

            i = i+1
        dest_node_idx = path[-1]
        dest_node = g.vs[dest_node_idx]['name']
        print(path)
        path_name_list = [g.vs[i]['name'] for i in path]
        print(sum_path)
        sum_path_list.append((start_node, dest_node, sum_path, str(path_name_list)))
    sum_path_list = sorted(sum_path_list, key=lambda x: x[2])
    print(sum_path_list)
    """
    write start node to all the nodes' shortest path sum to one csv file
    """
    df = pd.DataFrame(sum_path_list, columns=['start_index', 'destination_index', 'shortest_path_distance', 'shortest_path_seq'])
    df.to_csv('path_dist_index_'+str(start_node)+'.csv', sep=',', index=False)





    # layout = g.layout("kamada_kawai")
    # plot(g, layout=layout)
