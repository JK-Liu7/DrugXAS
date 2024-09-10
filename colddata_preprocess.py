import numpy as np
import random
import torch
import pandas as pd
import copy

device = torch.device('cuda')


def prepare_fold(link, node_list, node_type):
    whole_positive_index = []
    whole_negetive_index = []

    for i in range(link.shape[0]):
        for j in range(link.shape[1]):
            if node_type == 0 and i not in node_list:
                continue
            if node_type == 1 and j not in node_list:
                continue
            if link[i][j] == 1:
                whole_positive_index.append([i, j, 1])
            else:
                whole_negetive_index.append([i, j, 0])
    positive_num = len(whole_positive_index)
    select_positive_num = int(positive_num)
    positive_index = [i for i in range(positive_num)]
    np.random.shuffle(positive_index)
    negative_index = np.random.choice(np.arange(len(whole_negetive_index)), size=positive_num, replace=False)

    fold_index_sel = []
    fold_index_res = []
    for index in range(positive_num):
        if index < select_positive_num:
            fold_index_sel.append(whole_positive_index[positive_index[index]])
            fold_index_sel.append(whole_negetive_index[negative_index[index]])
        else:
            fold_index_res.append(whole_positive_index[positive_index[index]])
            fold_index_res.append(whole_negetive_index[negative_index[index]])
    return fold_index_sel, fold_index_res


def prepare_fold_pair(link, drug_list, entity_list):
    whole_positive_index = []
    whole_negetive_index = []

    for i in range(link.shape[0]):
        for j in range(link.shape[1]):
            if i not in drug_list or j not in entity_list:
                continue
            if link[i][j] == 1:
                whole_positive_index.append([i, j, 1])
            else:
                whole_negetive_index.append([i, j, 0])
    positive_num = len(whole_positive_index)
    select_positive_num = int(positive_num)
    positive_index = [i for i in range(positive_num)]
    np.random.shuffle(positive_index)
    negative_index = np.random.choice(np.arange(len(whole_negetive_index)), size=positive_num, replace=False)

    fold_index_sel = []
    fold_index_res = []
    for index in range(positive_num):
        if index < select_positive_num:
            fold_index_sel.append(whole_positive_index[positive_index[index]])
            fold_index_sel.append(whole_negetive_index[negative_index[index]])
        else:
            fold_index_res.append(whole_positive_index[positive_index[index]])
            fold_index_res.append(whole_negetive_index[negative_index[index]])
    return fold_index_sel, fold_index_res


def build_fold_cold(mode, link, cold_list, train_list):
    if mode == 'drug':
        nodetype = 0
    elif mode == 'entity':
        nodetype = 1

    fold_train_index = []
    fold_test_index, fold_train_index_ = prepare_fold(link, cold_list, nodetype)
    fold_train_index.extend(fold_train_index_)
    fold_train_index_, _ = prepare_fold(link, train_list, nodetype)
    fold_train_index.extend(fold_train_index_)

    X_test = np.array(fold_test_index)[:, :2]
    X_train = np.array(fold_train_index)[:, :2]
    Y_test = np.array(fold_test_index)[:, -1]
    Y_train = np.array(fold_train_index)[:, -1]

    return X_test, X_train, Y_test, Y_train


def build_fold_coldpair(link, drug_cold, drug_train, entity_cold, entity_train):
    fold_train_index = []
    fold_test_index, fold_train_index_ = prepare_fold_pair(link, drug_cold, entity_cold)
    fold_train_index.extend(fold_train_index_)
    fold_train_index_, _ = prepare_fold_pair(link, drug_train, entity_train)
    fold_train_index.extend(fold_train_index_)

    X_test = np.array(fold_test_index)[:, :2]
    X_train = np.array(fold_train_index)[:, :2]
    Y_test = np.array(fold_test_index)[:, -1]
    Y_train = np.array(fold_train_index)[:, -1]

    return X_test, X_train, Y_test, Y_train


def build_cold_drug(data, args):
    link = data['link_adj'].copy()
    fold_nums = args.k_fold
    drug_num = args.drug_number
    drug_list = [i for i in range(drug_num)]
    random.seed(args.random_seed)
    random.shuffle(drug_list)
    split_len = int(drug_num // fold_nums)
    cnt = 0

    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []

    for fold_num in range(fold_nums):
        if fold_num < fold_nums - 1:
            cold_drug_list = drug_list[cnt: cnt + split_len]
            train_drug_list = drug_list[:cnt] + drug_list[cnt + split_len:]
        else:
            cold_drug_list = drug_list[cnt:]
            train_drug_list = drug_list[:cnt]
        X_test, X_train, Y_test, Y_train = build_fold_cold('drug', link, cold_drug_list, train_drug_list)
        X_train_p = X_train[Y_train == 1, :]
        X_train_n = X_train[Y_train == 0, :]
        X_test_p = X_test[Y_test == 1, :]
        X_test_n = X_test[Y_test == 0, :]

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)
        cnt += split_len

    data['X_train_dr'] = X_train_all
    data['X_train_p_dr'] = X_train_p_all
    data['X_train_n_dr'] = X_train_n_all
    data['X_test_dr'] = X_test_all
    data['X_test_p_dr'] = X_test_p_all
    data['X_test_n_dr'] = X_test_n_all
    data['Y_train_dr'] = Y_train_all
    data['Y_test_dr'] = Y_test_all
    return data


def build_cold_entity(data, args):
    link = data['link_adj'].copy()
    fold_nums = args.k_fold
    entity_num = args.entity_number
    entity_list = [i for i in range(entity_num)]
    random.seed(args.random_seed)
    random.shuffle(entity_list)
    split_len = int(entity_num // fold_nums)
    cnt = 0

    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []

    for fold_num in range(fold_nums):
        if fold_num < fold_nums - 1:
            cold_entity_list = entity_list[cnt: cnt + split_len]
            train_entity_list = entity_list[:cnt] + entity_list[cnt + split_len:]
        else:
            cold_entity_list = entity_list[cnt:]
            train_entity_list = entity_list[:cnt]
        X_test, X_train, Y_test, Y_train = build_fold_cold('entity', link, cold_entity_list, train_entity_list)
        X_train_p = X_train[Y_train == 1, :]
        X_train_n = X_train[Y_train == 0, :]
        X_test_p = X_test[Y_test == 1, :]
        X_test_n = X_test[Y_test == 0, :]

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)
        cnt += split_len

    data['X_train_ent'] = X_train_all
    data['X_train_p_ent'] = X_train_p_all
    data['X_train_n_ent'] = X_train_n_all
    data['X_test_ent'] = X_test_all
    data['X_test_p_ent'] = X_test_p_all
    data['X_test_n_ent'] = X_test_n_all
    data['Y_train_ent'] = Y_train_all
    data['Y_test_ent'] = Y_test_all
    return data


def build_cold_pair(data, args):
    link = data['link_adj'].copy()
    fold_nums = args.k_fold
    random.seed(args.random_seed)

    drug_num = args.drug_number
    drug_list = [i for i in range(drug_num)]
    random.shuffle(drug_list)
    entity_num = args.entity_number
    entity_list = [i for i in range(entity_num)]
    random.shuffle(entity_list)
    drug_split_len = int(drug_num // fold_nums)
    entity_split_len = int(entity_num // fold_nums)
    drug_cnt, entity_cnt = 0, 0

    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []

    for fold_num in range(fold_nums):
        if fold_num < fold_nums - 1:
            cold_entity_list = entity_list[entity_cnt: entity_cnt + entity_split_len]
            train_entity_list = entity_list[:entity_cnt] + entity_list[entity_cnt + entity_split_len:]
            cold_drug_list = drug_list[drug_cnt: drug_cnt + drug_split_len]
            train_drug_list = drug_list[:drug_cnt] + drug_list[drug_cnt + drug_split_len:]
        else:
            cold_entity_list = entity_list[entity_cnt:]
            train_entity_list = entity_list[:entity_cnt]
            cold_drug_list = drug_list[drug_cnt:]
            train_drug_list = drug_list[:drug_cnt]
        X_test, X_train, Y_test, Y_train = build_fold_coldpair(link, cold_drug_list, train_drug_list, cold_entity_list, train_entity_list)
        X_train_p = X_train[Y_train == 1, :]
        X_train_n = X_train[Y_train == 0, :]
        X_test_p = X_test[Y_test == 1, :]
        X_test_n = X_test[Y_test == 0, :]

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)

        entity_cnt += entity_split_len
        drug_cnt += drug_split_len

    data['X_train_pair'] = X_train_all
    data['X_train_p_pair'] = X_train_p_all
    data['X_train_n_pair'] = X_train_n_all
    data['X_test_pair'] = X_test_all
    data['X_test_p_pair'] = X_test_p_all
    data['X_test_n_pair'] = X_test_n_all
    data['Y_train_pair'] = Y_train_all
    data['Y_test_pair'] = Y_test_all
    return data