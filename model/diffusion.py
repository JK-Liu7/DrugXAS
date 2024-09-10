import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import dgl
import dgl.nn.pytorch


class Diffusion(nn.Module):
    def __init__(
            self,
            args,
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
    ):
        super(Diffusion, self).__init__()
        self.args = args
        self.dim = self.args.diff_dim
        self.nhead = self.args.diff_head
        self.layer = self.args.diff_layer
        self.T = self.args.diff_T

        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
            'betas', beta
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

        self.alpha_l = alpha_l

        self.net = Denoising_GNN(in_dim=self.dim,
                                  num_hidden=self.dim,
                                  out_dim=self.dim,
                                  num_layers=self.layer,
                                  nhead=self.nhead)

        self.time_embedding = nn.Embedding(T, self.dim)
        self.norm_x = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, g, x, y):

        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))

        t = torch.randint(self.T, size=(x.shape[0],), device=x.device)
        x_t, time_embed, g = self.sample_q(t, x, g)
        loss = self.node_denoising(y, x_t, time_embed, g)
        return loss

    def sample_q(self, t, x, g):
        noise = torch.randn_like(x, device=x.device)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed, g

    def node_denoising(self, y, x_t, time_embed, g):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed)
        loss = loss_fn(out, y, self.alpha_l)
        return loss

    def embed(self, g, x, T):
        t = torch.full((1,), T, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))
        x_t, time_embed, g = self.sample_q(t, x, g)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed)
        return hidden


def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class Denoising_GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 ):
        super(Denoising_GNN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.mlp_in_t = MlpBlock(in_dim=in_dim, hidden_dim=num_hidden * 2, out_dim=num_hidden)
        self.mlp_middle = MlpBlock(num_hidden, num_hidden, num_hidden)
        self.mlp_out = MlpBlock(num_hidden, out_dim, out_dim)

        self.down_layers.append(dgl.nn.pytorch.GATv2Conv(num_hidden, num_hidden // nhead, nhead))
        self.up_layers.append(dgl.nn.pytorch.GATv2Conv(num_hidden, num_hidden, 1))

        for _ in range(1, num_layers):
            self.down_layers.append(dgl.nn.pytorch.GATv2Conv(num_hidden, num_hidden // nhead, nhead))
            self.up_layers.append(dgl.nn.pytorch.GATv2Conv(num_hidden, num_hidden // nhead, nhead))
        self.up_layers = self.up_layers[::-1]

    def forward(self, g, x_t, time_embed):
        h_t = self.mlp_in_t(x_t)
        down_hidden = []
        for l in range(self.num_layers):
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.down_layers[l](g, h_t)
            h_t = h_t.flatten(1)
            down_hidden.append(h_t)
        h_middle = self.mlp_middle(h_t)
        h_t = h_middle
        for l in range(self.num_layers):
            h_t = h_t + down_hidden[self.num_layers - l - 1 ]
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.up_layers[l](g, h_t)
            h_t = h_t.flatten(1)
        out = self.mlp_out(h_t)
        return out, h_t

class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(MlpBlock, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, hidden_dim)))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x