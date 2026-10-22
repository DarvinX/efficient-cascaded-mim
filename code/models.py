"""Explicit reconstructed architectures. No claim of matching absent original code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


def time_embedding(t, dim):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2 - 1))
    a = t.float()[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], -1)


class ImageFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = efficientnet_b0(weights=None).features[:7]

    def forward(self, x):
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
        return F.adaptive_avg_pool2d(self.net(x), (7, 7))


class MaskedModel(nn.Module):
    mask_id = 8192

    def __init__(self, dim=768, depth=12, heads=12):
        super().__init__()
        self.embed = nn.Embedding(8193, dim)
        self.down = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.text = nn.Linear(768, dim)
        self.feature = nn.Linear(192, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(1, 64 + 49 + 77 + 1, dim) * .02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(32, dim), nn.SiLU())
        self.out = nn.Conv2d(dim, 8192, 1)

    def forward(self, tokens, text, features, mask_ratio):
        assert tokens.shape[1:] == (16, 16), tokens.shape
        assert features.shape[1:] == (192, 7, 7), features.shape
        skip = self.embed(tokens).permute(0, 3, 1, 2)
        im = self.down(skip)
        b, d, h, w = im.shape
        seq = torch.cat([im.flatten(2).transpose(1, 2), self.feature(features.flatten(2).transpose(1, 2)), self.text(text), self.time(mask_ratio[:, None])[:, None, :]], 1)
        seq = seq + self.pos
        for block in self.blocks:
            seq = block(seq)
        im = self.norm(seq[:, :h * w]).transpose(1, 2).reshape(b, d, h, w)
        im = F.interpolate(im, (16, 16), mode="nearest")
        return self.out(self.up(im) + skip)


def build_masked_model(cfg):
    model=MaskedModel(**cfg["b"])
    if cfg.get("b_gilorat"):
        from low_rank import factorize_transformer
        # Factor initialization must not change the following EfficientNet's
        # initialization or its CUDA stochastic-depth stream in the paired run.
        with torch.random.fork_rng(devices=[]):
            cpu=torch.Generator(device="cpu").manual_seed(cfg["seed"]+61117)
            torch.set_rng_state(cpu.get_state())
            model=factorize_transformer(model,cfg["b_gilorat"])
    return model


class FeatureDiffusion(nn.Module):
    def __init__(self, dim=1152, depth=16, heads=16):
        super().__init__()
        self.input = nn.Linear(192, dim)
        self.text = nn.Linear(768, dim)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(1, 49 + 77 + 1, dim) * .02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 192)
        self.dim = dim

    def forward(self, x, t, text):
        b = x.shape[0]
        seq = torch.cat([self.input(x.flatten(2).transpose(1, 2)), self.text(text), self.time(time_embedding(t, self.dim))[:, None, :]], 1) + self.pos
        for block in self.blocks:
            seq = block(seq)
        return self.output(self.norm(seq[:, :49])).transpose(1, 2).reshape(b, 192, 7, 7)


class FeatureCrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, text):
        b, d, h, w = x.shape
        flat = x.flatten(2).transpose(1, 2)
        q = self.q(self.norm(flat)).reshape(b, h * w, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(text).reshape(b, text.shape[1], 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, h * w, d)
        return ((flat + self.out(attended)) / math.sqrt(2)).transpose(1, 2).reshape(b, d, h, w)


class FeatureResidualBlock(nn.Module):
    def __init__(self, dim, heads, attention=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.time = nn.Linear(dim, dim * 2)
        self.attention = FeatureCrossAttention(dim, heads) if attention else None

    def forward(self, x, time, text):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = (x + self.conv2(F.silu(h))) / math.sqrt(2)
        return self.attention(x, text) if self.attention is not None else x


class FeatureResNet(nn.Module):
    """Representative missing ResNet+attention ablation, not original source code."""
    def __init__(self, dim=512, depth=8, heads=8):
        super().__init__()
        assert dim % 32 == 0 and dim % heads == 0
        self.dim = dim
        self.input = nn.Conv2d(192, dim, 3, padding=1)
        self.text = nn.Linear(768, dim)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([FeatureResidualBlock(dim, heads, attention=(i % 2 == 1)) for i in range(depth)])
        self.norm = nn.GroupNorm(32, dim)
        self.output = nn.Conv2d(dim, 192, 3, padding=1)

    def forward(self, x, t, text):
        assert x.shape[1:] == (192, 7, 7)
        time = self.time(time_embedding(t, self.dim))
        text = self.text(text)
        x = self.input(x)
        for block in self.blocks:
            x = block(x, time, text)
        return self.output(F.silu(self.norm(x)))


def noise_schedule(device):
    t = torch.linspace(0, 1, 1001, device=device)
    a = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    betas = (1 - a[1:] / a[:-1]).clamp(max=.999)
    return torch.cumprod(1 - betas, 0)


def mask_tokens(tokens, generator=None, fixed_ratio=None):
    b = tokens.shape[0]
    if fixed_ratio is None:
        ratio = (torch.rand(b, device=tokens.device, generator=generator) * math.pi / 2).cos().clamp(min=.05)
    else:
        ratio = torch.full((b,), fixed_ratio, device=tokens.device)
    count = (ratio * 256).round().long().clamp(min=1)
    ranks = torch.rand(b, 256, device=tokens.device, generator=generator).argsort(-1).argsort(-1)
    mask = (ranks < count[:, None]).reshape_as(tokens)
    return torch.where(mask, 8192, tokens), torch.where(mask, tokens, -100), ratio
