import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch


class DrugGAT(nn.Module):
    def __init__(self, args, in_feats):
        super(DrugGAT, self).__init__()
        self.args = args
        self.in_feats = in_feats
        self.out_dim = self.args.drug_dim
        self.num_heads = self.args.drug_head
        self.readout = self.args.drug_readout
        self.gat1 = dgl.nn.pytorch.GATv2Conv(in_feats=self.in_feats, out_feats=args.drug_dim, num_heads=self.num_heads, allow_zero_in_degree=True)

    def forward(self, g):

        x = g.ndata['atom'].clone()
        x1, score = self.gat1(g, x, get_attention=True)
        x1 = torch.mean(x1, dim=1).squeeze()
        g.ndata['h'] = x1.clone()

        if self.readout == "sum":
            h = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            h = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            h = dgl.mean_nodes(g, 'h')
        else:
            h = dgl.mean_nodes(g, 'h')

        return x1, h, score

