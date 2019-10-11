import pandas as pd
import os


def add_media(filename):
    df = pd.read_csv(filename, delimiter=',')
    summ = df['Expression'].sum()
    numm= df['Expression'].count()
    df['media'] = float(summ/numm)
    df.to_csv(filename, index=False)
    return

if __name__ == "__main__":

    file = '../../data/sub_graph/new_subgraph_instance_ENSP_idx.csv'
    input_folder = '../../data/index_ENSP/target_node_shortest_path/'
    # node_index_name = '../../data/index_ENSP/node_index.csv'
    df = pd.read_csv(file, delimiter=',')
    # df['idx_file'] = df['Protein'].map(str) + '_' + df['sig_id'] +'_ENSP_idx.csv'
    #
    # file_list = df['idx_file'].tolist()
    # print('file_list:')
    # print('\t' + str(len(file_list)))
    # folder_files = os.listdir(input_folder)
    #
    # dup_set = set([x for x in file_list if file_list.count(x) > 1])
    # print(dup_set)
    #
    # print('all is done')

    folder_files = os.listdir(input_folder)
    folder_files = sorted(folder_files)
    # [add_media(input_folder+x) for x in folder_files]
    # print('add media done')
    spath_list = [int(f.split('_')[-1][:-4]) for f in folder_files]
    target_list = list(set(df['Protein'].tolist()))
    # print('target list: ' + str(len(set(target_list))))
    # print('spath_list: ' + str(len(set(spath_list))))
    # diff = list(set(target_list)-set(spath_list))
    if set(target_list) <= set(spath_list):
        print('jobs done')

    # print(len(diff))
    # target_list = sorted(target_list)
    # spath_list = sorted(spath_list)
    #
    # for i in zip(target_list, spath_list):
    #     print(i)

