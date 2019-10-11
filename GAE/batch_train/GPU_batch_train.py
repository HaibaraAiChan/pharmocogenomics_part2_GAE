from __future__ import division
from __future__ import print_function

import argparse
import os
from data_generator import DataGenerator
import time
import shutil
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import pickle
from model import GCNModelVAE
from optimizer import loss_function
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from utils import preprocess_graph, get_roc_score, draw_loss_curve


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=9, help="batch size")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=4, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=8, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--K', type=int, default=10, help='the nearest K neighbors')
    parser.add_argument('--f_length', type=int, default=10, help='the length of feature vector')
    parser.add_argument('--dataset', type=str, default='../../../data/index_ENSP/GAE_INPUT/', help='index ENSP folder.')
    parser.add_argument('--outfolder', type=str, default='../../../data/index_ENSP/GAE_OUTPUT/',
                        help='model output folder.')
    args = parser.parse_args()
    return args


def load_list(data_folder, K):
    sub_adj_path_list = []
    sub_feat_path_list = []
    folder_list = os.listdir(data_folder)
    folder_list.sort()
    print(folder_list)
    K_folder = [f for f in folder_list if str(K)==f.split('_')[0] in f][0]
    print(K_folder)
    K_path = data_folder + K_folder

    for folder_name in os.listdir(data_folder + K_folder):
        adj_path = data_folder + K_folder + "/" + folder_name + '/'
        feature_path = K_folder + "/" + folder_name + '/'
        K_adj_file_path = K_folder + "/" + folder_name + '/' + folder_name + '.pkl'

        n_f = len(os.listdir(adj_path)) - 1
        Adj_file_list = [K_adj_file_path] * n_f

        num_file = 0
        feature_list = []
        for filename in os.listdir(adj_path):
            if '.pkl' in filename:
                continue

            feature_list.append(feature_path + filename)
            # Adj_file_list.append(K_adj_file_path)

            num_file = num_file + 1
            print(num_file, end=' ')
        if num_file == n_f:
            print(' ' + K_adj_file_path + 'is done')
            sub_adj_path_list = sub_adj_path_list + Adj_file_list
            sub_feat_path_list = sub_feat_path_list + feature_list
    print('adj and feature list are done')
    list_t = [list(a) for a in zip(sub_adj_path_list, sub_feat_path_list)]
    return list_t


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cuda = torch.device('cuda:0')  # https://pytorch.org/docs/stable/notes/cuda.html
    # cudnn.benchmark = True
    args = args()

    path = args.dataset
    output = args.outfolder

    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    dim = args.K + 1
    feature_length = args.f_length
    model_out = output + 'batch_'+str(batch_size) + '/model_K_' + str(args.K)+'_' + str(batch_size) + '_' + str(epochs) + '_'+str(args.hidden1)+ '_' +str(args.hidden2)+ '_' +str(args.f_length)+ '_'+ str(lr)+'/'
    # if os.path.exists(model_out):
    #     shutil.rmtree(model_out)
    if not os.path.exists(model_out):
        os.makedirs(model_out)


        cnt = 0

    adj_feat_list = load_list(args.dataset, args.K)

    partition = {"train": adj_feat_list}

    # Parameters
    train_params = {'dim': dim,
                    'adj_dim': (dim, dim),
                    'feature_dim': (dim, 1),
                    'batch_size': batch_size,
                    'shuffle': False,
                    'path': path}
    training_generator = DataGenerator(partition['train'], **train_params)
    print("the training data is ready")
    model = GCNModelVAE(batch_size, args.hidden1, args.hidden2, args.dropout)
    model.cuda()  ###################################################################
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    loss_record = []
    start_time = time.time()

    for epoch in range(epochs):
        scheduler.step()
        # Training
        #     numm=0
        cur_loss_list = []
        for adj_bs, features_bs in training_generator:
            # Transfer to GPU

            #   print(int(adj_bs.shape[1]/dim))

            n_nodes = np.array(features_bs).shape[0]
            feat_dim = int(adj_bs.shape[1] / dim)
            adj_norm = preprocess_graph(adj_bs)
            adj_label = adj_bs + sp.eye(adj_bs.shape[0])
            # adj_label = sparse_to_tuple(adj_label)
            adj_label = torch.FloatTensor(adj_label.toarray())
            # adj_label = torch.LongTensor(adj_label.toarray())

            pos_weight = float(adj_bs.shape[0] * adj_bs.shape[0] - adj_bs.sum()) / adj_bs.sum()
            norm = adj_bs.shape[0] * adj_bs.shape[0] / float((adj_bs.shape[0] * adj_bs.shape[0] - adj_bs.sum()) * 2)

            # model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
            # model.cuda()  ###################################################################
            # optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # model = torch.nn.DataParallel(model)  ###################################################################


            hidden_emb = None
            # t = time.time()
            model.train()
            optimizer.zero_grad()
            features_bs = features_bs.cuda()
            adj_norm = adj_norm.cuda()

            # print('features_bs :')
            # print(features_bs)
            # print('adj_norm :')
            # print(adj_norm)
            recovered, mu, logvar = model(features_bs, adj_norm)
            loss = loss_function(preds=recovered,
                                 labels=adj_label,
                                 mu=mu,
                                 logvar=logvar,
                                 n_nodes=n_nodes,
                                 norm=norm,
                                 pos_weight=pos_weight)
            cpu_loss = loss.cpu()
            cur_loss_list.append(cpu_loss.data.numpy().tolist())
            loss.backward()
            # cur_loss = loss.item()
            optimizer.step()
            # scheduler.step()

            hidden_emb = mu.cpu().data.numpy()
            # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_bs)
        # print('cur_loss_list: ')
        # print(cur_loss_list)
        # print(len(cur_loss_list))
        # one epoch one mean loss
        # total training samples num: len(adj_feat_list)
        # sum up all the batch losses and average them to get current epoch mean loss
        mean_loss = sum(cur_loss_list) / len(adj_feat_list)
        print('Epoch '+ str(epoch)+'  mean_loss: ')
        print(mean_loss)
        loss_record.append(mean_loss)
    print('loss record by epoch: ')
    print(loss_record)
    fout = open(model_out+'loss_by_epochs.txt', 'w')
    fout.writelines(["%s\n" % item for item in loss_record])
    # draw_loss_curve(loss_record)
    end_time = time.time()
    print('time elapse is : ' + str(end_time-start_time))
    model_file = 'model_'+str(batch_size)+'_'+str(epochs)+'_'+str(lr)+'.pkl'
    print(model.cpu())
    # of = open(model_file, 'w')
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), model_out + model_file)
    ### load model dict
    # model = GCNModelVAE(batch_size, args.hidden1, args.hidden2, args.dropout)
    # model.load_state_dict(torch.load(model_file))
    # print(model)
