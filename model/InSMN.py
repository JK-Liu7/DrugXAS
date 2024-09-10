import torch
import torch.nn as nn
import torch.nn.functional as F


class InSMN(nn.Module):
    def __init__(self, args):

        super(InSMN, self).__init__()
        self.args = args
        self.k = self.args.in_k
        self.emb_dim = self.args.net_dim

        self.drug_linear1 = nn.Linear(3 * self.emb_dim, self.emb_dim)

        self.mlp1 = MLP(self.emb_dim, self.emb_dim * self.k, self.emb_dim // 2, self.emb_dim * self.k)
        self.mlp2 = MLP(self.emb_dim, self.emb_dim * self.k, self.emb_dim // 2, self.emb_dim * self.k)


    def forward(self, h_mol, h_net, drug_seq):
        net_nei1 = h_net[drug_seq].mean(dim=1)
        drug_net = h_net[:self.args.drug_number, :]

        drug_emb = self.drug_linear1(torch.cat((h_mol, drug_net, net_nei1), dim=1))
        drug_meta1 = self.mlp1(drug_emb).reshape(-1, self.emb_dim, self.k)
        drug_meta2 = self.mlp2(drug_emb).reshape(-1, self.k, self.emb_dim)

        meta_bias1 = (torch.mean(drug_meta1, dim=0))
        meta_bias2 = (torch.mean(drug_meta2, dim=0))

        low_weight1 = F.softmax(drug_meta1 + meta_bias1, dim=1)
        low_weight2 = F.softmax(drug_meta2 + meta_bias2, dim=1)

        mol_emb = torch.sum(torch.multiply((h_mol).unsqueeze(-1), low_weight1), dim=1)
        mol_emb = torch.sum(torch.multiply((mol_emb).unsqueeze(-1), low_weight2), dim=1)

        net_emb = torch.sum(torch.multiply((drug_net).unsqueeze(-1), low_weight1), dim=1)
        net_emb = torch.sum(torch.multiply((net_emb).unsqueeze(-1), low_weight2), dim=1)

        h_mol1 = h_mol + mol_emb
        h_net1 = drug_net + net_emb
        return h_mol1, h_net1


class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=True):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu = nn.PReLU().cuda()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x