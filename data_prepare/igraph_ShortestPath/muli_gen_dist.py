
import pandas as pd
import numpy as np


def find_all_target_nodes(file):
    df = pd.read_csv(file, delimiter=',')
    uni_pro_list = df['Protein'].unique()
    return uni_pro_list


if __name__ == '__main__':
    output_folder = '../../../data/index_ENSP/target_node_shortest_path/'
    file = '../../../data/index_ENSP/cellline_index.csv'
    protein_list = find_all_target_nodes(file)

    protein_list = np.sort(protein_list)
    print(protein_list)
    print(len(protein_list))
    length = len(protein_list)
    i = 0
    num = 0
    while i in range(length):
        str_tmp = protein_list[i:i+43]

        file = open('input/'+str(num)+".in", "w+")
        # str_tmp = '\n'.join(str(str_tmp))
        str_t = '\n'.join(map(str, str_tmp))
        file.write(str_t)
        num = num+1
        i = i+43
    print(len(protein_list))
