import random
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from dgl.nn.pytorch import GraphConv
from model.net_hgnn import get_seqs
from model.attention_layer import AttLayer



class NetEncoder0(nn.Module):
    def __init__(self, args, fine_tune=False):
        super(NetEncoder0, self).__init__()
        self.args = args
        self.gnn_layer = self.args.gnn_layer
        self.hgnn_layer = self.args.hgnn_layer
        self.emb_dim = self.args.net_dim
        self.k = self.args.net_k
        self.len = self.args.net_len
        self.drug_number = self.args.drug_number
        self.entity_number = self.args.entity_number
        self.ratio = self.args.net_ratio

        self.ATT_drug = AttLayer(self.args.net_head, self.drug_number, self.emb_dim)
        self.ATT_entity = AttLayer(self.args.net_head, self.entity_number, self.emb_dim)

        self.drug_linear = nn.Linear(self.drug_number, self.emb_dim)
        self.entity_linear = nn.Linear(self.entity_number, self.emb_dim)

        self.GCNLayers = nn.ModuleList()
        self.drug_linear1 = nn.Linear(self.args.drug_number, self.emb_dim)
        self.entity_linear1 = nn.Linear(self.args.entity_number, self.emb_dim)
        self.Drop = nn.Dropout(self.args.dropout)

        self.fine_tune = fine_tune

        for layer in range(self.gnn_layer):
            self.GCNLayers.append(GraphConv(self.emb_dim, self.emb_dim, activation=F.relu, allow_zero_in_degree=True))


    def set_fine_tune(self):
        self.fine_tune = True


    def forward(self, g, drug_feat, entity_feat, node_seq, node_seq_hop, node_seq0, ratio):
        drug_seq = node_seq[:self.drug_number, :]
        entity_seq = node_seq[self.drug_number:, :]
        drug_seq_hop = node_seq_hop[:self.drug_number, :]
        entity_seq_hop = node_seq_hop[self.drug_number:, :]

        drug_emb1 = self.drug_linear(drug_feat)
        entity_emb1 = self.entity_linear(entity_feat)
        emb1 = torch.cat((drug_emb1, entity_emb1), dim=0)

        h0_drug = emb1[drug_seq]
        h0_entity = emb1[entity_seq]
        score_drug = self.ATT_drug(h0_drug)
        score_entity = self.ATT_entity(h0_entity)

        g_homo = dgl.to_homogeneous(g)
        g_aug, drug_feat1, entity_feat1 = self.attention_aug(g_homo, score_drug, score_entity, drug_feat, entity_feat, drug_seq, entity_seq, drug_seq_hop, entity_seq_hop, ratio)

        node_list = list(range(self.drug_number + self.entity_number))

        if self.training:
            node_seq2, _ = get_seqs(g_aug, node_list, self.len)
        else:
            node_seq2 = node_seq0

        gh = []
        drug_feat2 = self.drug_linear1(drug_feat1)
        entity_feat2 = self.entity_linear1(entity_feat1)

        feat = torch.cat((drug_feat2, entity_feat2), dim=0)
        if g_aug.is_homogeneous:
            g_aug_homo = g_aug.clone()
        else:
            g_aug_homo = dgl.to_homogeneous(g_aug)
        g_aug_homo = dgl.add_self_loop(g_aug_homo)

        drug_seq = node_seq2[:self.drug_number, :]
        entity_seq = node_seq2[self.drug_number:, :]

        for layer in range(self.gnn_layer):
            gh = self.GCNLayers[layer](g_aug_homo, feat)
            gh = self.Drop(gh)

        h_drug = gh[drug_seq]
        h_entity = gh[entity_seq]
        return h_drug, h_entity, node_seq2, g_aug


    def attention_aug(self, g, score_drug, score_entity, drug_feat, entity_feat, drug_seq, entity_seq, drug_seq_hop, entity_seq_hop, ratio):
        with torch.no_grad():
            g_aug = g.clone()
            exc_index_dr, exc_index_ent, del_index, add_index = \
                self.get_allIndex(score_drug, score_entity, drug_seq, entity_seq, drug_seq_hop, entity_seq_hop, ratio)

            drug_feat1, entity_feat1 = self.exchange_node(exc_index_dr, exc_index_ent, drug_feat, entity_feat)
            g_aug = self.del_edge(g_aug, del_index)
            g_aug = self.add_edge(g_aug, add_index)
        return g_aug, drug_feat1, entity_feat1


    def exchange_node(self, exc_index_dr, exc_index_ent, drug_feat, entity_feat):
        drug_feat_exc = drug_feat.clone()
        drug_feat_exc[exc_index_dr] = 0
        entity_feat_exc = entity_feat.clone()
        entity_feat_exc[exc_index_ent] = 0
        return drug_feat_exc, entity_feat_exc


    def del_edge(self, g, del_index):
        g_aug = g.clone()
        del_index = np.array(del_index)
        if len(del_index) != 0:
            _, _, del_id = g_aug.edge_ids(del_index[:, 0], del_index[:, 1], return_uv=True)
            g_aug = dgl.remove_edges(g_aug, del_id)
        return g_aug


    def add_edge(self, g, add_index):
        g_aug = g.clone()
        add_index = np.array(add_index)
        if len(add_index) != 0:
            add_index = np.unique(add_index, axis=0)
            g_aug = dgl.add_edges(g_aug, add_index[:, 0], add_index[:, 1])
        return g_aug


    def get_allIndex(self, drug_weight, entity_weight, drug_seq, entity_seq, drug_seq_hop, entity_seq_hop, ratio):
        # removing 1-hop, adding 2-hop
        # min weight to remove, max weight to add
        drug_seq_hop = drug_seq_hop[:, 1:].cpu().numpy()
        entity_seq_hop = entity_seq_hop[:, 1:].cpu().numpy()
        drug_seq1 = drug_seq[:, 1:].cpu().numpy()
        entity_seq1 = entity_seq[:, 1:].cpu().numpy()

        drug_weight0 = drug_weight[:, 1:].cpu().numpy()
        entity_weight0 = entity_weight[:, 1:].cpu().numpy()
        drug_weightn = drug_weight[:, 0].cpu().numpy()
        entity_weightn = entity_weight[:, 0].cpu().numpy()

        p = ratio
        drug_node_indices1 = np.argsort(drug_weightn)
        entity_node_indices1 = np.argsort(entity_weightn)
        drug_excn = int(p * self.drug_number)
        entity_excn = int(p * self.entity_number)
        exc_index_dr = drug_node_indices1[:drug_excn].tolist()
        exc_index_ent = entity_node_indices1[:entity_excn].tolist()
        drug_weight1 = np.where(drug_seq_hop == 1, drug_weight0, np.ones_like(drug_weight0))
        drug_weight2 = np.where(drug_seq_hop == 2, drug_weight0, -np.ones_like(drug_weight0))
        entity_weight1 = np.where(entity_seq_hop == 1, entity_weight0, np.ones_like(entity_weight0))
        entity_weight2 = np.where(entity_seq_hop == 2, entity_weight0, -np.ones_like(entity_weight0))
        drug_indices1 = np.argsort(drug_weight1, axis=1)
        drug_indices2 = np.argsort(-drug_weight2, axis=1)
        entity_indices1 = np.argsort(entity_weight1, axis=1)
        entity_indices2 = np.argsort(-entity_weight2, axis=1)

        drug_hop = [[np.sum(drug_seq_hop[i] == 1), np.sum(drug_seq_hop[i] == 2)] for i in range(len(drug_seq))]
        entity_hop = [[np.sum(entity_seq_hop[i] == 1), np.sum(entity_seq_hop[i] == 2)] for i in range(len(entity_seq))]

        drug_deln = [[int(i * p) for i in row] for row in drug_hop]
        entity_deln = [[int(i * p) for i in row] for row in entity_hop]
        drug_addn = [[int(2 * i * p) for i in row] for row in drug_hop]
        entity_addn = [[int(2 * i * p) for i in row] for row in entity_hop]

        del_index, add_index = [], []

        for i in range(len(drug_seq)):
            drug_del1 = drug_indices1[i][:drug_deln[i][0]].tolist()
            drug_add1 = drug_indices2[i][:drug_addn[i][1]].tolist()
            drug_del2 = drug_seq1[i][drug_del1].tolist()
            drug_add2 = drug_seq1[i][drug_add1].tolist()
            del_index.extend([[drug_seq[i, 0], item] for item in drug_del2])
            add_index.extend([[drug_seq[i, 0], item] for item in drug_add2])

        for i in range(len(entity_seq)):
            entity_del1 = entity_indices1[i][:entity_deln[i][0]].tolist()
            entity_add1 = entity_indices2[i][:entity_addn[i][1]].tolist()
            entity_del2 = entity_seq1[i][entity_del1].tolist()
            entity_add2 = entity_seq1[i][entity_add1].tolist()
            del_index.extend([[entity_seq[i, 0], item] for item in entity_del2])
            add_index.extend([[entity_seq[i, 0], item] for item in entity_add2])

        exc_index_dr = torch.tensor(exc_index_dr, dtype=torch.long)
        exc_index_ent = torch.tensor(exc_index_ent, dtype=torch.long)
        del_index = torch.tensor(del_index, dtype=torch.long)
        add_index = torch.tensor(add_index, dtype=torch.long)
        return exc_index_dr, exc_index_ent, del_index, add_index