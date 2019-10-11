"""
convert Q9P0X4--> ENSP00000385019
input:  new_subgraph_instance_d.csv
output: new_subgraph_instance_s.csv
"""

import pandas as pd
import time

import shutil
import os
import pickle


def convert(uni_file, ifile,ofile):

    df_data = pd.read_csv(ifile, delimiter=',')
    df_uni = pd.read_csv(uni_file, delimiter=',')
    df_uni = df_uni.set_index('Protein')
    key_dict = df_uni.to_dict()

    df_data.loc[:, 'Protein'] = df_data['Protein'].replace(key_dict['Protein_name'])
    df_data = df_data[['file', 'Protein', 'sig_id', 'Drug', 'Cell_lines', 'pert_idose']]

    print(df_data[0:10])
    print(df_data.count())
    df_data = df_data.drop_duplicates()
    print(df_data.count())
    # the length of whole df_data before drop duplicates and after is the same.
    df_data.to_csv(ofile, index=False)
    return df_data


if __name__ == "__main__":
    uni_file = '../../data/positive-uni-ENSP'
    input_file = '../../data/sub_graph/new_subgraph_instance_d.csv'
    output_file = '../../data/sub_graph/new_subgraph_instance_ENSP.csv'
    ofile_idx='../../data/sub_graph/new_subgraph_instance_ENSP_idx.csv'
    node_index_name = '../../data/index_ENSP/node_index.csv'
    df_idx = pd.read_csv(node_index_name)
    print(df_idx.head())

    start = time.time()
    df_data = convert(uni_file, input_file,output_file)

    df_data['Protein'] = df_data['Protein'].map(df_idx.set_index('Protein')['idx'])
    df_data = df_data.dropna()
    df_data['Protein'] = df_data['Protein'].astype(int)
    sig = df_data['sig_id']
    sig = sig.drop_duplicates()
    print(len(list(sig)))
    df_data.to_csv(ofile_idx, index=False)
    end = time.time()
    print('vector time elapsed :' + str(end - start))
    # print(data)