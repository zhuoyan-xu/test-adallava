import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def gumbel_softmax(logits, tau = 1, hard = False, eps = 1e-10, dim = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def masked_gumbel_softmax(logits, masks, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    if training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        gumbels = gumbels.masked_fill_(masks.bool(), float('-inf')) # mask out the already picked items
        y_soft = gumbels.softmax(dim)
    else:
        gumbels = logits.masked_fill_(masks.bool(), float('-inf'))
        y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def n_times_gumbel_softmax(logits, n = 1, tau = 1, hard = False, eps = 1e-10, dim = -1, training = True):
    cumulate_mask = torch.zeros_like(logits, device=logits.device)
    for i in range(int(n)):
        mask = masked_gumbel_softmax(logits, cumulate_mask, tau, hard=True, dim=dim, training=training)
        cumulate_mask = cumulate_mask + mask
    return cumulate_mask

def posemb_sincos_1d(latency_num, dim=256, temperature=10000, dtype=torch.float32):
    n = latency_num

    n = torch.arange(n)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1. / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)
    

def latency_encoding(latency):

    # Scale the batch latency to a range of [0, 2π]
    scaled_values = latency * 2 * torch.pi  # Shape: [batch_size]

    # Generate frequency indices to create a diverse range of sine and cosine values
    frequencies = torch.linspace(1, 128, 128).to(latency.device)  # 128 frequencies for each sin and cos

    # Expand dimensions to compute sine and cosine for each value in the batch with each frequency
    # `scaled_values[:, None]` adds a dimension to match (batch_size, 1) with (128), resulting in (batch_size, 128)
    sin_values = torch.sin(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]
    cos_values = torch.cos(scaled_values[:, None] * frequencies)  # Shape: [batch_size, 128]

    # Concatenate sin and cos values along the last dimension to get a tensor of size [batch_size, 256]
    latency = torch.cat((sin_values, cos_values), dim=1)  # Shape: [batch_size, 256]
    return latency

class SimpleScheduler(nn.Module):
    def __init__(self, dim_in=4096, num_sub_layer=16, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.random_policy = False
        self.random_layer = False
        self.random_layer_ratio = 1.

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x, latency):
        logits = self.mlp_head(x) # [bs, 16]
        output_samples = []
        for logits_, latency_ in zip (logits, latency):
            sample = n_times_gumbel_softmax(logits_, latency_.item(), self.tau, self.is_hard, training=self.training)
            output_samples.append(sample)
        
        output_samples = torch.stack(output_samples)
        return output_samples