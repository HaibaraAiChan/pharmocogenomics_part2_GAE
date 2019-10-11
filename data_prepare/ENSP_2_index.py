
import numpy as np
import scipy.sparse as sp
import os
import pandas as pd


def gen_gene_index(gene_df, file_name,  output):
    if not os.path.exists(file_name):
        print('no node index file')
        return
    df = pd.read_csv(file_name)

    print(df.head())
    gene_df['Protein'] = gene_df['Protein'].map(df.set_index('Protein')['idx'])
    gene_df['Gene'] = gene_df['Gene'].map(df.set_index('Protein')['idx'])
    num_nan = gene_df.isna().sum()
    print(num_nan)
    gene_df.astype({'Protein': int, 'Gene': int})
    gene_df.to_csv(output+'gene_index.csv', index=False)
    return gene_df


def gen_cellline_index(cell_df, file_name, output):
    if not os.path.exists(file_name):
        print('no node index file')
        return
    df = pd.read_csv(file_name)

    print(df.head())
    cell_df['Protein'] = cell_df['Protein'].map(df.set_index('Protein')['idx'])
    cell_df.astype({'Protein': int})
    cell_df.to_csv(output + 'cellline_index.csv', index=False)
    return cell_df


def gen_graph_index(graph_df, dataset):
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    node_array = np.unique(graph_df[['protein1', 'protein2']].values)
    node_array = np.sort(node_array)
    print(len(node_array))
    df = pd.DataFrame({'Protein': node_array, })
    node_index_name = dataset+'node_index.csv'

    idx = [i for i in range(len(node_array))]
    df['idx'] = idx
    df.to_csv(dataset + 'node_index.csv', index=False)
    print(df.head())
    graph_df['protein1'] = graph_df['protein1'].map(df.set_index('Protein')['idx'])
    graph_df['protein2'] = graph_df['protein2'].map(df.set_index('Protein')['idx'])
    graph_df.to_csv(dataset+'graph_index.csv', index=False)
    return graph_df, node_index_name


if __name__ == "__main__":
    input_folder = '../../data/ENSP/'
    output_folder = '../../data/index_ENSP/'
    names = ['interactions']
    objects = []
    folder_files = os.listdir(input_folder)
    interaction_file = [f for f in folder_files if names[0] in f][0]

    interaction_data = pd.read_csv(input_folder + interaction_file, delimiter='\t')

    graph_index, node_index_name = gen_graph_index(interaction_data, output_folder)
    # node_index_name = '../../data/index_ENSP/node_index.csv'
    # gene_index = gen_gene_index(gen_data, node_index_name, output_folder)

    # cellline_index = gen_cellline_index(cellline_data, node_index_name,  output_folder)

    print('pre-process data-set is ready')