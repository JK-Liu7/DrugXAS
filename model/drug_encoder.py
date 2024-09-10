import torch
import torch.nn as nn
import numpy as np
import dgl
import dgl.nn.pytorch
import random
from model.drug_gat import DrugGAT


class DrugEncoder0(nn.Module):
    def __init__(self, args):
        super(DrugEncoder0, self).__init__()
        self.args = args
        self.ratio = self.args.drug_ratio

        self.gat1 = DrugGAT(self.args, in_feats=44)
        self.gat2 = DrugGAT(self.args, in_feats=44)

    def forward(self, g, node_num, edge_num, ratio):
        node_sim = g.ndata['atom'].clone()
        _, h1, score1 = self.gat1(g)
        g_aug, _ = self.attention_aug(g, score1, node_num, edge_num, node_sim, ratio)
        h_atom, h2, _ = self.gat2(g_aug)
        return h_atom, h2

    def attention_aug(self, g, score, node_num, edge_num, node_sim, ratio):
        g_aug = g.clone()
        with torch.no_grad():

            edge_index = g_aug.edges()
            len_w = g_aug.num_edges()

            node_pos = np.where(edge_index[0].cpu().numpy() == edge_index[1].cpu().numpy())[0]
            edge_pos = np.setdiff1d(np.arange(len_w), node_pos)

            x_weight_fc = torch.mean(score[node_pos], dim=1)
            edge_weight_fc = torch.mean(score[edge_pos], dim=1)
            x_del_weights = []
            edge_del_weights = []
            node_count, edge_count1, edge_count2 = 0, 0, 0

            for i in range(len(node_num)):
                x_weight = x_weight_fc[node_count:node_count + node_num[i]] if node_count < len_w else torch.tensor([])
                edge_weight = edge_weight_fc[edge_count1:edge_count1 + edge_num[i] - node_num[i]] if edge_count1 < len_w else torch.tensor([])
                if len(edge_weight) > 0:
                    edge_del_indices = self.get_allIndex(edge_weight, len(edge_weight), ratio) + edge_count2
                    edge_del_weights.append(edge_del_indices)
                if len(x_weight) > 0:
                    x_del_indices = self.get_allIndex(x_weight, len(x_weight), ratio) + node_count
                    x_del_weights.append(x_del_indices)
                edge_count1 += edge_num[i] - node_num[i]
                edge_count2 += edge_num[i]
                node_count += node_num[i]

            if x_del_weights:
                x_del_weights = torch.cat(x_del_weights).cuda()
                node_sim = self.del_node(node_sim, x_del_weights)
            if edge_del_weights:
                edge_del_weights = torch.cat(edge_del_weights).cuda()
                g_aug.remove_edges(edge_del_weights.squeeze())
            g_aug.ndata['atom'] = node_sim
        return g_aug, node_sim

    def del_node(self, x, del_indices):
        x[del_indices] = 0
        return x

    def get_allIndex(self, weight, len_w, ratio):
        # min weight deletion
        weight = weight.cpu().numpy()
        sorted_indices = np.argsort(weight)
        sorted_weight = weight[sorted_indices]
        p = ratio
        len_q = int(len_w * p)
        top_indices = np.sort(sorted_indices[:len_q])
        top_slice = random.sample(top_indices.tolist(), int(len_q))
        del_indices = np.sort(np.array(top_slice))[::-1]
        del_indices = torch.tensor(del_indices.copy(), dtype=torch.long)
        return del_indices.cuda()


class Drug_GNN(nn.Module):
    def __init__(self, args):
        super(Drug_GNN, self).__init__()
        self.args = args
        self.dim = self.args.drug_dim * 2
        self.gat = DrugGAT(self.args, in_feats=self.dim)

    def forward(self, g, h_atom, h_share, node_num):
        g1 = g.clone()
        h_share_x = torch.repeat_interleave(h_share, node_num, dim=0)
        h_node = torch.cat((torch.add(h_atom, h_share_x), torch.sub(h_atom, h_share_x)), -1)
        g1.ndata['atom'] = h_node
        h_atom1, h, _ = self.gat(g1)
        return h_atom1, h

