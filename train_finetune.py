import copy
import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from drug_dataset import DrugDataset
from torch.utils.data import DataLoader
from model.DrugXAS_SSL import DrugXAS
from model.net_hgnn import get_seqs
from pos_contrast import mp_pos, mp_data
from metric import *
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda')


def pretrain(args, data, drug_loader):
    model = DrugXAS(args)
    model = model.to(device)
    optimizer_cl = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_cl, mode='min', factor=0.8, patience=100, verbose=True, min_lr=2e-5)

    het_graph, edge_types, data = dgl_heterograph(data, data['all_link_p'], args)
    het_graph = het_graph.to(device)
    ded, ede = mp_data(data['all_link_p'], args)
    pos_d, pos_e = mp_pos(ded, ede, data['drs'], data['ens'], args, data)
    pos_d = torch.IntTensor(pos_d.toarray()).to(device)
    pos_e = torch.IntTensor(pos_e.toarray()).to(device)

    node_list = list(range(args.drug_number + args.entity_number))
    g_homo = dgl.to_homogeneous(het_graph)
    node_seq, node_seq_hop = get_seqs(g_homo, node_list, args.net_len)

    best_lcl = 100

    for epoch in range(args.epochs):
        model.train()

        for _, batch_data in enumerate(drug_loader):
            drug_graph, drug_node, drug_edge = batch_data
            l_cl, drug_emb, ent_emb, g_new1, g_new1_homo, node_seq2 = \
                model(drug_graph, drug_node, drug_edge, het_graph, drug_feature, entity_feature, node_seq, node_seq_hop,
                      node_seq, pos_d, pos_e, _, _, epoch)
            optimizer_cl.zero_grad()
            l_cl.backward()
            optimizer_cl.step()
            l_cl = l_cl.detach().cpu().numpy()
            scheduler.step(l_cl)

            end = timeit.default_timer()
            time = end - start

            if l_cl < best_lcl:
                best_epoch_cl = epoch + 1
                best_lcl = l_cl
                best_epoch = epoch + 1
                model_cl = copy.deepcopy(model)
                if args.save_model:
                    torch.save(model.state_dict(), args.model_save_dir + str(args.dataset)
                               + '_dim_' + str(args.drug_dim) + '_k' + str(args.net_k) + '_len' + str(args.net_len)
                               + '_gl' + str(args.gnn_layer) + '_hgl' + str(args.hgnn_layer) + '_ink' + str(
                        args.in_k) + '.pth')
                drug_emb_cl = drug_emb.detach()
                ent_emb_cl = ent_emb.detach()

            if (epoch + 1) % 20 == 0:
                show = [epoch + 1, round(time, 2), best_lcl, best_epoch]
                print('\t\t'.join(map(str, show)))
            node_seq1, node_seq_hop1 = get_seqs(g_new1_homo, node_list, args.net_len)
            node_seq = node_seq1.clone()
            node_seq_hop = node_seq_hop1.clone()
            het_graph = g_new1.clone()
    print("\nBest Epoch: {}".format(best_epoch_cl))
    return model_cl


def finetune(args, data, drug_loader, model_cl):
    for i in range(args.k_fold):
        print('fold:', i)

        model_finetune = copy.deepcopy(model_cl)
        model_finetune.set_fine_tune()
        cnt_wait = 0
        best_auc, best_aupr = 0, 0

        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        X_train_p = torch.LongTensor(data['X_train_p'][i]).to(device)
        X_train_n = torch.LongTensor(data['X_train_n'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        X_test_p = torch.LongTensor(data['X_test_p'][i]).to(device)
        X_test_n = torch.LongTensor(data['X_test_n'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        het_graph, edge_types, data = dgl_heterograph(data, data['X_train_p'][i], args)
        het_graph = het_graph.to(device)

        ded, ede = mp_data(data['X_train_p'][i], args)
        pos_d, pos_e = mp_pos(ded, ede, data['drs'], data['ens'], args, data)
        pos_d = torch.IntTensor(pos_d.toarray()).to(device)
        pos_e = torch.IntTensor(pos_e.toarray()).to(device)

        node_list = list(range(args.drug_number + args.entity_number))
        g_homo = dgl.to_homogeneous(het_graph)
        node_seq, node_seq_hop = get_seqs(g_homo, node_list, args.net_len)

        optimizer_link = optim.Adam(model_finetune.parameters(), lr=args.link_lr)

        for epoch in range(args.link_epochs):
            model_finetune.train()

            for _, batch_data in enumerate(drug_loader):
                drug_graph, drug_node, drug_edge = batch_data
                train_score, drug_emb_cl, ent_emb_cl, link_emb_p, link_emb_n, g_new1, g_new1_homo, node_seq2 = \
                    model_finetune(drug_graph, drug_node, drug_edge, het_graph, drug_feature, entity_feature,
                                   node_seq, node_seq_hop, node_seq, pos_d, pos_e, X_train_p, X_train_n, epoch)

            l_link = cross_entropy(train_score, torch.flatten(Y_train))

            with torch.autograd.set_detect_anomaly(True):
                optimizer_link.zero_grad()
                l_link.backward()
                optimizer_link.step()
                l_link = l_link.detach().cpu().numpy()

            with torch.no_grad():
                model_finetune.eval()

                test_score = \
                    model_finetune(drug_graph, drug_node, drug_edge, het_graph, drug_feature, entity_feature,
                                   node_seq, node_seq_hop, node_seq2, pos_d, pos_e, X_test_p, X_test_n, epoch)

                test_prob = fn.softmax(test_score, dim=-1)
                test_score = torch.argmax(test_score, dim=-1)
                test_prob = test_prob[:, 1]
                test_prob = test_prob.cpu().numpy()
                test_score = test_score.cpu().numpy()

                AUC, AUPR, _, _, _, _, _ = get_metric(Y_test, test_score, test_prob)
                end = timeit.default_timer()
                time = end - start

                if AUC > best_auc:
                    cnt_wait = 0
                    best_epoch = epoch + 1
                    best_auc = AUC
                    best_aupr = AUPR
                else:
                    cnt_wait += 1
                if cnt_wait == args.patience:
                    print('Early stopping at epoch', epoch)
                    break

                if (epoch + 1) % 20 == 0:
                    show = [epoch + 1, round(time, 2), round(l_link.item(), 5), round(best_auc, 5), round(best_aupr, 5), best_epoch]
                    print('\t\t'.join(map(str, show)))

                node_seq1, node_seq_hop1 = get_seqs(g_new1_homo, node_list, args.net_len)
                node_seq = node_seq1.clone()
                node_seq_hop = node_seq_hop1.clone()
                het_graph = g_new1.clone()

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', round(AUC_mean.item(), 4), '(', round(AUC_std.item(), 4), ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', round(AUPR_mean.item(), 4), '(', round(AUPR_std.item(), 4), ')')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', default=True, help='whether pre_train')
    parser.add_argument('--save_model', default=True, help='whether to save model')
    parser.add_argument('--task', default='CDA', help='task')
    parser.add_argument('--dataset', default='DengCDA', help='dataset')
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--batch', type=int, default=2048, help='batchsize')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=77, help='random seed')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout ratio')

    parser.add_argument('--drug_dim', default='128', type=int, help='drug molecular embedding dimension')
    parser.add_argument('--drug_head', default='4', type=int, help='number of head of drug GAT')
    parser.add_argument('--drug_readout', default='mean', help='readout layer of drug embedding')
    parser.add_argument('--drug_ratio', default='0.2,0.001,0.3', type=str, help='mask ratio of drug molecular augmentation')

    parser.add_argument('--net_k', default='3', type=int, help='number of the highest similarity neighbors added in graph')
    parser.add_argument('--net_len', default='6', type=int, help='number of the neighbors in the subgraph')
    parser.add_argument('--gnn_layer', default='2', type=int, help='gnn layers of local encoder')
    parser.add_argument('--hgnn_layer', default='2', type=int, help='hgnn layers')
    parser.add_argument('--net_dim', default='128', type=int, help='network embedding dimension')
    parser.add_argument('--net_head', default='4', type=int, help='number of heads in hgnn')
    parser.add_argument('--net_ratio', default='0.2,0.001,0.3', type=str, help='mask ratio of network augmentation')
    parser.add_argument('--in_k', default='5', type=int)

    parser.add_argument('--diff_dim', default='128', type=int, help='diffusion model embedding dimension')
    parser.add_argument('--diff_head', default='4', type=int, help='number of head of diffusion model')
    parser.add_argument('--diff_layer', default='2', type=int, help='number of gnn layers of diffusion model')
    parser.add_argument('--diff_T', default='1000', type=int)

    parser.add_argument('--link_epochs', type=int, default=1000, help='number of epochs for link prediction')
    parser.add_argument('--link_lr', type=float, default=2e-4, help='learning rate of link prediction')
    parser.add_argument('--patience', default='400', type=int, help='patience of early stopping in link prediction')
    parser.add_argument('--mlp_dropout', default='0.4', type=float, help='dropout of MLP')
    parser.add_argument('--pos_num', default='15', type=int, help='threshold of positive samples selection')
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)

    args = parser.parse_args()
    args.data_dir = 'data/' + args.task + '/' + args.dataset + '/'
    args.model_save_dir = 'model_save/' + args.task + '/' + args.dataset + '/'
    args.result_dir = 'result/' + args.task + '/' + args.dataset + '/'

    entity_name = {
        'DTI': 'Protein',
        'DDA': 'Disease',
        'DSE': 'SideEffect',
        'MDA': 'miRNA',
        'CDA': 'circRNA'
    }

    args.entity = entity_name[args.task]
    data = get_data(args)
    args.drug_number = data['drug_number']
    args.entity_number = data['entity_number']
    args.batch = args.batch

    data = data_processing(data, args)
    data = k_fold(data, args)

    drug_feature = torch.FloatTensor(data['drs']).to(device)
    entity_feature = torch.FloatTensor(data['ens']).to(device)

    all_sample = torch.tensor(data['all_link']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric_ssl = ('Epoch\t\tTime\t\tL_cl\t\tBest epoch')
    Metric_link = ('Epoch\t\tTime\t\tL_link\t\tBest AUC\t\tBest AUPR\t\tBest epoch')
    AUCs, AUPRs = [], []

    drug_smiles = data['smiles']
    drug_data = DrugDataset(drug_smiles)
    drug_loader = DataLoader(drug_data, batch_size=args.batch, shuffle=False, collate_fn=drug_data.collate, drop_last=False)

    # Pre-train
    if args.pre_train:
        print('Pre_training on Dataset:', args.dataset)
        print(Metric_ssl)
        model_cl = pretrain(args, data, drug_loader)

    # Load pre-trained model
    else:
        model_cl = DrugXAS(args)
        model_cl = model_cl.to(device)
        model_cl.load_state_dict(torch.load(args.model_save_dir + str(args.dataset)
                                   + '_dim_' + str(args.drug_dim) + '_k' + str(args.net_k) + '_len' + str(args.net_len)
                                   + '_gl' + str(args.gnn_layer) + '_hgl' + str(args.hgnn_layer) + '_ink' + str(args.in_k) + '.pth'))

    # Fine-tune
    print('Fine_tuning on Dataset:', args.dataset)
    print(Metric_link)
    finetune(args, data, drug_loader, model_cl)




