import torch
from torch.utils import data
import numpy as np
import pickle
import scipy.sparse as sp
import pandas as pd
from scipy.linalg import block_diag


class DataGenerator(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, dim, batch_size, adj_dim=(11, 11), feature_dim=(11, 1), shuffle=True, path=''):
        '''Initialization'''
        self.dim = dim
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path = path
        self.adj_dim = adj_dim
        self.feature_dim = feature_dim
        self.on_epoch_end()

    def __len__(self):
        'Denotes the total number of samples'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generates one sample of data'
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # block_diagonal
        # Adj: (batch_size*adj_dim, batch_size*adj_dim), original adj(adj_dim, adj_dim)
        # feature:(batch_size*adj_dim, batch_size*f_dim), original feature(adj_dim,f_dim), in this project f_dim=1, only gene expression
        # Initialization

        adj_list = []
        adj_orig_list = []
        feature_list = []
        # Generate data
        numm=0
        print('list_IDs_temp')
        for i, sub_path in enumerate(list_IDs_temp):

            # if sub_path != ['10_neighbors/1535/1535.pkl', '10_neighbors/1535/1535_DB00502_MCF7.csv']:
            #     print('no')
            #     continue
            print(sub_path)
            adj_tmp_file = sub_path[0]
            adj_tmp_path = self.path + adj_tmp_file
            pickle_file = open(adj_tmp_path, 'rb')
            cur_adj = pickle.load(pickle_file).toarray()
            adj_list.append(cur_adj)
            # adj_orig = cur_adj
            # # tmp_debug1 = adj_orig.diagonal()
            # # tmp_debug2 = tmp_debug1[np.newaxis, :]
            # # tmp_debug3 = sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            # adj_orig.eliminate_zeros()
            # adj_orig_list.append(adj_orig)
            # remove diagonal distances(lambda), in this project, it is not necessary. They are both zeros.

            feat_tmp_file = sub_path[1]
            feat_tmp_path = self.path + feat_tmp_file
            feature_file = open(feat_tmp_path, 'rb')
            df_data = pd.read_csv(feature_file, delimiter=',')
            f_list = df_data['expression'].tolist()
            f_list = [[el] for el in f_list]
            # f_t = torch.FloatTensor(np.array(f_list))
            feature_list.append(f_list)


        j = 1
        tmp_adj_array = adj_list[j-1]
        tmp_feature_array = feature_list[j-1]
        while j in range(len(adj_list)):
            tmp_adj_array = block_diag(tmp_adj_array, adj_list[j])
            tmp_feature_array = block_diag(tmp_feature_array, feature_list[j])
            j = j + 1
        adj_array = tmp_adj_array
        feature_array = tmp_feature_array
        adj_bs = sp.csr_matrix(adj_array)
        feature_bs = torch.FloatTensor(np.array(feature_array))
        return adj_bs, feature_bs


if __name__ == '__main__':
    import numpy, scipy.sparse
    A = numpy.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    Asp = scipy.sparse.csr_matrix(A)
    print(Asp.toarray())