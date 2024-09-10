import dgl
import dgl.function as fn
import numpy as np
from numpy.linalg import norm
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import pandas as pd
import random


class AttLayer(nn.Module):
    def __init__(self, nheads, in_dim, emb_dim):

        super(AttLayer, self).__init__()
        # self.args = args
        self.nheads = nheads
        self.in_dim = in_dim
        self.emb_dim = emb_dim
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

    def forward(self, h):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)
        attn = torch.mean(score, dim=1)
        attn = attn[:, 0, :].squeeze()
        return attn