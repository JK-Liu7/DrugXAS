import dgl
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from model.contrast import Contrast
from model.drug_encoder import DrugEncoder0, Drug_GNN
from model.net_encoder import NetEncoder0
from model.net_hgnn import HGNN
from model.InSMN import InSMN
from model.diffusion import Diffusion


class DrugXAS(nn.Module):
    def __init__(self, args, fine_tune=False):

        super(DrugXAS, self).__init__()
        self.args = args
        self.drug_ratio = self.args.drug_ratio
        self.net_ratio = self.args.net_ratio
        self.drug_dim = self.args.drug_dim
        self.drug_number = self.args.drug_number
        self.entity_number = self.args.entity_number
        self.hgnn_layer = self.args.hgnn_layer
        self.k = self.args.net_k
        self.dropout = self.args.mlp_dropout
        self.gamma = self.args.gamma

        self.drug_encoder0 = DrugEncoder0(self.args)
        self.drug_encoder = nn.ModuleList()

        self.net_encoder0 = NetEncoder0(self.args, fine_tune=False)
        self.net_encoder = nn.ModuleList()

        self.drug_contrast = Contrast(self.args)
        self.InSMN = nn.ModuleList()
        self.Diffusion_M2N = Diffusion(args)
        self.Diffusion_N2M = Diffusion(args)

        self.mol_linear = nn.ModuleList()

        for layer in range(self.hgnn_layer - 1):
            self.drug_encoder.append(Drug_GNN(self.args))

        for layer in range(self.hgnn_layer):
            self.net_encoder.append(HGNN(self.args))

        for layer in range(self.hgnn_layer):
            self.InSMN.append(InSMN(self.args))

        self.fine_tune = fine_tune
        self.mlp = nn.Sequential(
            nn.Linear(self.args.net_dim, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 2)
        )

    def set_fine_tune(self):
        self.fine_tune = True

    def forward(self, drug_graph, drug_node, drug_edge, het_graph, drug_feature, entity_feature,
                node_seq, node_seq_hop, node_seq0, drug_pos, entity_pos, sample_p, sample_n, epoch):
        cur_drug_ratio = self.get_mask_rate(self.drug_ratio, epoch)
        cur_net_ratio = self.get_mask_rate(self.net_ratio, epoch)
        h_atom, h_mol = self.drug_encoder0(drug_graph, drug_node, drug_edge, cur_drug_ratio)

        if self.fine_tune:
            self.net_encoder0.set_fine_tune()

        h_drug, h_entity, node_seq2, g_aug = self.net_encoder0(het_graph, drug_feature, entity_feature, node_seq, node_seq_hop, node_seq0, cur_net_ratio)
        drug_seq = node_seq2[:self.args.drug_number, :]

        for layer in range(self.hgnn_layer):
            h_drug, h_entity = self.net_encoder[layer](h_drug, h_entity)
            h_drug0 = h_drug[:, 0, :]
            h_entity0 = h_entity[:, 0, :]
            h_net_all = torch.cat((h_drug0, h_entity0), dim=0)
            h_mol1, h_drug1 = self.InSMN[layer](h_mol, h_net_all, drug_seq)
            if layer < self.hgnn_layer - 1:
                h_atom, h_mol = self.drug_encoder[layer](drug_graph, h_atom, h_mol1, drug_node)
            h_drug[:, 0, :] = h_drug1.clone()

        h_drug2 = h_drug[:, 0, :]
        h_entity2 = h_entity[:, 0, :]

        drug_sim = torch.cosine_similarity(h_drug2.unsqueeze(1), h_drug2.unsqueeze(0), dim=-1)
        entity_sim = torch.cosine_similarity(h_entity2.unsqueeze(1), h_entity2.unsqueeze(0), dim=-1)

        if self.training:
            drug_nei = self.neighbour_sampler(drug_sim, self.k)
            entity_nei = self.neighbour_sampler(entity_sim, self.k)
            node_nei = torch.cat((drug_nei, entity_nei), dim=0)
            g_new1 = self.graph_update(het_graph, node_nei)
            g_new1_homo = dgl.to_homogeneous(g_new1)
        else:
            g_new1, g_new1_homo = [], []

        sim_graph = self.sim_graph(drug_sim, self.k).to(het_graph.device)
        h_mol_new = h_mol.clone()
        h_net_new = h_drug2.clone()
        l_m2n = self.Diffusion_M2N(sim_graph, h_mol_new, h_net_new)
        l_n2m = self.Diffusion_N2M(sim_graph, h_net_new, h_mol_new)
        l_drug = self.drug_contrast(h_net_new, h_mol_new, drug_pos)

        drug_emb = h_net_new.clone()
        l_ssl = self.gamma * l_drug + (1 - self.gamma) * (l_m2n + l_n2m)

        if self.fine_tune:
            link_emb_p = torch.mul(drug_emb[sample_p[:, 0]], h_entity2[sample_p[:, 1]])
            link_emb_n = torch.mul(drug_emb[sample_n[:, 0]], h_entity2[sample_n[:, 1]])
            link_emb = torch.cat((link_emb_p, link_emb_n), dim=0)
            output = self.mlp(link_emb)
            if self.training:
                return output, drug_emb, h_entity2, link_emb_p, link_emb_n, g_new1, g_new1_homo, node_seq2
            else:
                return output
        else:
            return l_ssl, drug_emb, h_entity2, g_new1, g_new1_homo, node_seq2


    def neighbour_sampler(self, sim, len):
        sim = sim - torch.eye(sim.shape[0]).to(sim.device)
        _, indices = torch.sort(sim, dim=1, descending=True)
        seq_new = indices[:, :len]
        return seq_new


    def graph_update(self, g, node_nei):
        g_new = g.clone()
        drug_sim = g_new.edges('all', etype='drug-drug')[2]
        entity_sim = g_new.edges('all', etype='entity-entity')[2]
        g_new = dgl.remove_edges(g_new, drug_sim, etype='drug-drug')
        g_new = dgl.remove_edges(g_new, entity_sim, etype='entity-entity')

        node_nei_drug = [(i, node_nei[i][j].item()) for i in range(self.drug_number) for j in range(node_nei.shape[1])]
        node_nei_ent = [(i, node_nei[i + self.drug_number][j].item()) for i in range(self.entity_number) for j in range(node_nei.shape[1])]

        src_drug, dst_drug = zip(*node_nei_drug)
        src_ent, dst_ent = zip(*node_nei_ent)
        g_new = dgl.add_edges(g_new, src_drug, dst_drug, etype='drug-drug')
        g_new = dgl.add_edges(g_new, dst_drug, src_drug, etype='drug-drug')
        g_new = dgl.add_edges(g_new, src_ent, dst_ent, etype='entity-entity')
        g_new = dgl.add_edges(g_new, dst_ent, src_ent, etype='entity-entity')
        return g_new


    def sim_graph(self, drug_sim, k):
        drug_matrix = drug_sim.cpu().detach().numpy()
        num = drug_matrix.shape[0]
        knn_graph = np.zeros(drug_matrix.shape)
        idx_sort = np.argsort(-(drug_matrix - np.eye(num)), axis=1)
        for i in range(num):
            knn_graph[i, idx_sort[i, :k + 1]] = drug_matrix[i, idx_sort[i, :k + 1]]
            knn_graph[idx_sort[i, :k + 1], i] = drug_matrix[idx_sort[i, :k + 1], i]
        knn_graph = knn_graph + np.eye(num)
        drug_nx = nx.from_numpy_array(knn_graph)
        drug_simgraph = dgl.from_networkx(drug_nx)
        return drug_simgraph


    def get_mask_rate(self, input_mask_rate, epoch):
        if "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
            mask_rate = [float(i) for i in input_mask_rate.split(',')]
            assert len(mask_rate) == 3
            start = mask_rate[0]
            step = mask_rate[1]
            end = mask_rate[2]
            cur_mask_rate = start + epoch * step
            if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                return end
            return cur_mask_rate
        else:
            return float(input_mask_rate)



