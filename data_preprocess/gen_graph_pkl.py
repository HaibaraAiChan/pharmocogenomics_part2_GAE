"""
so far, August,7,2019
igraph 0.7.3
formula.py now is only compatible with python 3.6 below.(no python3.7)
many packages may be compatible with 3.7 in the future, reader can check current version
"""
import pandas as pd
from igraph import *

graph_file = './index_ENSP/undirected_graph_index.csv'
df = pd.read_csv(graph_file, delimiter=',')
tup_tmp = list(df.itertuples(index=False, name=None))

g = Graph.TupleList(tup_tmp, weights=True)
g.write_pickle(fname='undirected_graph.pkl', version=-1)
