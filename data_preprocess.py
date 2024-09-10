import numpy as np
import random
import torch
import pandas as pd
import dgl
# import networkx as nx
from sklearn.model_selection import StratifiedKFold
import copy

device = torch.device('cuda')


def get_sim_topk(sim_matrix, k):
    sim_list = []
    num = sim_matrix.shape[0]
    idx_sort = np.argsort(-(sim_matrix - np.eye(num)), axis=1)
    for i in range(num):
        for j in range(k):
            sim_list.append([i, idx_sort[i, j]])
            sim_list.append([idx_sort[i, j], i])
    return sim_list


def get_seqs(g, node_list, nei_len):
    n = 0
    node_seq = torch.zeros([len(node_list), nei_len]).long()

    for x in node_list:
        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        while (cnt < nei_len):
            hop1_list = g.successors(start).numpy().tolist()
            if len(hop1_list) == 0:
                hop1_list.append(start)
            nsampled = max(len(hop1_list), 1)
            sampled_list = random.sample(hop1_list, nsampled)
            for i in range(nsampled):
                node_seq[n, cnt] = sampled_list[i]
                cnt += 1
                if cnt == nei_len:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()
        n += 1
    return node_seq


def get_data(args):
    data = dict()

    drs = pd.read_csv(args.data_dir + 'Drug_sim.csv', header=None).to_numpy()
    ens = pd.read_csv(args.data_dir + args.entity + '_sim.csv', header=None).to_numpy()

    data['drug_number'] = int(drs.shape[0])
    data['entity_number'] = int(ens.shape[0])
    data['drs'] = drs
    data['ens'] = ens
    data['link'] = pd.read_csv(args.data_dir + args.task + '.csv', dtype=int).to_numpy()
    data['link_ent'] = data['link'][:, [1, 0]]
    drug_information = pd.read_csv(args.data_dir + 'DrugInformation.csv')
    data['smiles'] = drug_information['Smiles'].values
    return data


def data_processing(data, args):
    link_matrix = pd.read_csv(args.data_dir + 'Adj.csv', header=None).to_numpy()
    one_index = []
    zero_index = []
    for i in range(link_matrix.shape[0]):
        for j in range(link_matrix.shape[1]):
            if link_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]
    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)
    link_p = samples[samples[:, 2] == 1, :2]
    link_n = samples[samples[:, 2] == 0, :2]

    data['all_samples'] = samples
    data['all_link'] = samples[:, :2]
    data['all_link_p'] = link_p
    data['all_link_n'] = link_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    data['link_adj'] = link_matrix
    return data


def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_link']
    Y = data['all_label']
    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_p = X_train[Y_train[:, 0] == 1, :]
        X_train_n = X_train[Y_train[:, 0] == 0, :]
        X_test_p = X_test[Y_test[:, 0] == 1, :]
        X_test_n = X_test[Y_test[:, 0] == 0, :]
        X_train_all.append(X_train)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_all.append(X_test)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)

        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    data['X_train'] = X_train_all
    data['X_train_p'] = X_train_p_all
    data['X_train_n'] = X_train_n_all
    data['X_test'] = X_test_all
    data['X_test_p'] = X_test_p_all
    data['X_test_n'] = X_test_n_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def dgl_heterograph(data, link, args):
    link_ent = link[:, [1, 0]]
    link_list = link.tolist()
    link_ent_list = link_ent.tolist()

    drug_sim = get_sim_topk(data['drs'], args.net_k)
    entity_sim = get_sim_topk(data['ens'], args.net_k)

    data['drug_sim'] = drug_sim
    data['entity_sim'] = entity_sim

    node_dict = {
        'drug': args.drug_number,
        'entity': args.entity_number
    }

    heterograph_dict = {
        ('drug', 'drug-entity', 'entity'): (link_list),
        ('entity', 'entity-drug', 'drug'): (link_ent_list),
        ('drug', 'drug-drug', 'drug'): (drug_sim),
        ('entity', 'entity-entity', 'entity'): (entity_sim)
    }

    data['feature_dict'] = {
        'drug': torch.tensor(data['drs']),
        'entity': torch.tensor(data['ens'])
    }

    biomedical_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)
    biomedical_graph.ndata['h'] = data['feature_dict']

    edge_types = {
        'drug-entity': ['drug', 'entity'],
        'drug-drug': ['drug', 'drug'],
        'entity-entity': ['entity', 'entity']
    }

    data['biomedical_graph'] = biomedical_graph
    data['edge_types'] = edge_types
    return biomedical_graph, edge_types, data





