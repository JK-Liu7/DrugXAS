import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')


class GTLayer(nn.Module):
    def __init__(self, args):

        super(GTLayer, self).__init__()
        self.args = args
        self.nheads = self.args.net_head
        self.emb_dim = self.args.net_dim
        self.head_dim = self.emb_dim // self.nheads
        self.leaky = nn.LeakyReLU(0.01)

        self.linear_l = nn.Linear(
            self.emb_dim, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.emb_dim, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.emb_dim, bias=False)
        self.dropout1 = nn.Dropout(self.args.dropout)
        self.dropout2 = nn.Dropout(self.args.dropout)
        self.LN = nn.LayerNorm(self.emb_dim)

    def forward(self, h):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)
        attn = torch.mean(score, dim=1)
        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)
        context = score @ fr
        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)
        h = self.LN(h + fh)
        return h, attn


class HGNN(nn.Module):
    def __init__(self, args):

        super(HGNN, self).__init__()

        self.args = args
        self.gnn_layer = self.args.gnn_layer
        self.drug_number = self.args.drug_number
        self.entity_number = self.args.entity_number
        self.drug_gt = GTLayer(args)
        self.entity_gt = GTLayer(args)


    def forward(self, h_drug, h_entity):
        h_drug, attn_drug = self.drug_gt(h_drug)
        h_entity, attn_entity = self.entity_gt(h_entity)
        return h_drug, h_entity


def get_seqs(g, node_list, nei_len):
    successors_dict = {node: g.successors(node).cpu().numpy() for node in node_list}
    node_seq = np.zeros((len(node_list), nei_len), dtype=int)
    node_seq_hop = np.zeros((len(node_list), nei_len), dtype=int)

    for n, x in enumerate(node_list):
        node_seq[n, 0] = x
        node_seq_hop[n, 0] = 1
        hop1_list = successors_dict[x]
        if len(hop1_list) == 0:
            hop1_list = np.array([x])
        cnt = 1
        scnt = 0
        start = x
        while cnt < nei_len:
            hop_list = successors_dict[start]
            if len(hop_list) == 0:
                hop_list = np.array([start])
            num_to_sample = min(nei_len - cnt, len(hop_list))
            sampled_list = np.random.choice(hop_list, num_to_sample, replace=True)
            end_idx = cnt + num_to_sample
            node_seq[n, cnt:end_idx] = sampled_list
            node_seq_hop[n, cnt:end_idx] = np.where(np.isin(sampled_list, hop1_list), 1, 2)
            cnt = end_idx
            scnt += 1
            if scnt < cnt:
                start = node_seq[n, scnt]
            else:
                break

    node_seq = torch.tensor(node_seq, dtype=torch.long)
    node_seq_hop = torch.tensor(node_seq_hop, dtype=torch.long)
    return node_seq, node_seq_hop


