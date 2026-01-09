import math
from typing import Optional, Tuple

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols import ModelProtocol
from torchtitan.tools.logging import logger

from .args import WanModelArgs

# TODO (limou)
# use F.SDPA
try:
    from flash_attn import flash_attn_func

    # FLASH_ATTN_2_AVAILABLE = True
    FLASH_ATTN_2_AVAILABLE = False
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False
    logger.warning("Flash Attention not available, using standard attention")


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    if FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.norm_q = RMSNorm(hidden_size, eps=eps)
        self.norm_k = RMSNorm(hidden_size, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.norm_q = RMSNorm(hidden_size, eps=eps)
        self.norm_k = RMSNorm(hidden_size, eps=eps)

        self.attn = AttentionModule(num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.self_attn = SelfAttention(hidden_size, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_size, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)
        self.gate = GateModule()

    def reset_parameters(self):
        # Initialize modulation
        nn.init.normal_(self.modulation, mean=0.0, std=1.0 / self.hidden_size**0.5)
        
        # Recurse for submodules that might need it if they are custom, 
        # but usually FSDP handles recursion or we can trust standard layers.
        # For safety/completeness if FSDP relies on top-level call for this module:
        for module in self.modules():
            if module != self and hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def reset_parameters(self):
        # Default init for Sequential layers is usually fine/handled.
        if self.has_pos_emb:
            nn.init.zeros_(self.emb_pos)
        
        # Explicitly reset Sequential submodules
        for layer in self.proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
        eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_size, out_channels * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.head.reset_parameters()
        nn.init.normal_(self.modulation, mean=0.0, std=1.0 / self.hidden_size**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x

class WanDitModel(nn.Module, ModelProtocol):
    def __init__(self, model_args: WanModelArgs):
        super().__init__()
        self.hidden_size = model_args.dit_hidden_size
        self.in_channels = model_args.dit_in_channels
        self.intermediate_size = model_args.dit_intermediate_size
        self.freq_dim = model_args.dit_freq_dim
        self.text_dim = model_args.dit_text_dim
        self.out_channels = model_args.dit_out_channels
        self.num_layers = model_args.dit_num_layers
        self.num_heads = model_args.dit_num_heads
        self.eps = model_args.dit_eps
        self.patch_size = model_args.dit_patch_size

        # build the WanDit model
        head_dim = self.hidden_size // self.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.patch_embedding = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.intermediate_size,
                    self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = Head(self.hidden_size, self.out_channels, self.patch_size, self.eps)

    def patchify(
        self,
        x: torch.Tensor,
    ):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_size))
        context = self.text_embedding(context)  # self.text_embedding is an adapter.

        x, (f, h, w) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        #TODO (limou)
        logger.info("dit init_weights ...")
        pass