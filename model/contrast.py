import torch
import torch.nn as nn


class Contrast(nn.Module):
    def __init__(self, args):
        super(Contrast, self).__init__()
        self.args = args
        self.tau = self.args.tau
        self.lam = self.args.lam
        self.emb_dim = self.args.net_dim
        self.proj = nn.Sequential(
            nn.Linear(self.emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.emb_dim)
        )

        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos).sum(dim=-1)).mean()
        return self.lam * lori_a + (1 - self.lam) * lori_b


