
import os
import numpy as np
import pandas as pd


"""
convert Q9P0X4--> ENSP00000385019
input: all the raw data sub graph
ouput: all the ENSP target protein sub graph 
"""


def convert(key_dict,node_idx, ifile, ofolder):
    df_data = pd.read_csv(ifile, delimiter=',', names=['Protein', 'Drug', 'Cell_lines', 'sig_id', 'pert_idose', 'Gene', 'Expression'], header=None)
    # df_uni = pd.read_csv(uni_file, delimiter=',')
    # df_uni = df_uni.set_index('Protein')
    # key_dict = df_uni.to_dict()

    df_data.loc[:, 'Protein'] = df_data['Protein'].replace(key_dict['Protein_name'])
    df_data = df_data[['Protein', 'sig_id', 'Drug', 'Cell_lines', 'pert_idose', 'Gene', 'Expression']]

    df_data['Protein'] = df_data['Protein'].map(node_idx.set_index('Protein')['idx'])
    df_data['Protein'] = df_data['Protein'].astype(int)
    df_data['Gene'] = df_data['Gene'].map(node_idx.set_index('Protein')['idx'])
    df_data = df_data.dropna()
    df_data['Gene'] = df_data['Gene'].astype(int)

    # ores = df_data.columns[df_data.nunique() == 1]
    # print(ores)
    ENSP_idx = df_data.iloc[0]['Protein']
    sig_id = df_data.iloc[0]['sig_id']
    ofile = str(ENSP_idx) + '_' + sig_id + '_ENSP_idx.csv'
    # print(df_data[0:10])
    # print(df_data.count())
    df_data = df_data.drop_duplicates()
    df_data = df_data[['Gene', 'Expression']]
    # print(df_data.count())
    # the length of whole df_data before drop duplicates and after is the same.
    df_data.to_csv(ofolder+ofile, index=False)



if __name__ == "__main__":
    input_folder = '../../data/sub_graph/raw_data/'
    uni_file = '../../data/positive-uni-ENSP'
    output_folder = '../../data/sub_graph/index_ENSP/'
    node_index_name = '../../data/index_ENSP/node_index.csv'
    df_uni = pd.read_csv(uni_file, delimiter=',')
    df_uni = df_uni.set_index('Protein')
    key_dict = df_uni.to_dict()

    df_idx = pd.read_csv(node_index_name)
    # print(df_idx.head())

    folder_files = os.listdir(input_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for ifile in folder_files:
        convert(key_dict, df_idx, input_folder+ifile, output_folder)

    print('all is done')