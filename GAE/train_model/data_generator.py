import torch
from torch.utils import data
import numpy as np
import pickle
import scipy.sparse as sp
import pandas as pd


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
        'Generates data containing batch_size samples'  # Adj : (n_samples, *a_dim), feature :(n_samples, *f_dim)
        # Initialization
        # ll = len(self.adj_dim)
        # adj_tmp = []
        # for i in range(ll):
        #     adj_tmp.append(int(self.adj_dim[i]))
        # adj = np.empty((self.batch_size, adj_tmp[0], adj_tmp[1]))
        # ll = len(self.feature_dim)
        # for i in range(ll):
        # f_tmp.append(int(self.feature_dim[i]))
        # f_tmp = [int(self.feature_dim[i]) for i in range(len(self.feature_dim))]
        #
        # feature = np.empty((self.batch_size, f_tmp[0], f_tmp[1]))
        # adj_orig_list = np.empty((self.batch_size, adj_tmp[0], adj_tmp[1]))
        adj_list = []
        adj_orig_list = []
        feature_list = []
        # Generate data
        for i, sub_path in enumerate(list_IDs_temp):
            print(sub_path)

            adj_tmp_file = sub_path[0]
            adj_tmp_path = self.path + adj_tmp_file
            pickle_file = open(adj_tmp_path, 'rb')
            cur_adj = pickle.load(pickle_file)
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
            f_t = torch.FloatTensor(np.array(f_list))
            feature_list.append(f_t)

        return adj_list, feature_list
