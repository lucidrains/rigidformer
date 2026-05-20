from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter

import einx
from einops import einsum, repeat
from einops.layers.torch import Rearrange
from torch_einops_utils import pack_with_inverse

import roma

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_queries_gates = Linear(dim, dim_inner * 2, bias = False)
        self.to_keys_values = Linear(dim, dim_inner * 2, bias = False)

        self.to_out = Linear(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
        context = None
    ):

        context = default(context, tokens)

        queries, gates, keys, values = (
            *self.to_queries_gates(tokens).chunk(2, dim = -1),
            *self.to_keys_values(context).chunk(2, dim = -1)
        )

        queries, keys, values = (self.split_heads(t) for t in (queries, keys, values))

        queries = queries * self.scale

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        out = out * gates.sigmoid()
        return self.to_out(out)

class SwiGluFeedforward(Module):
    # Shazeer et al

    def __init__(
        self,
        dim,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(
        self,
        tokens
    ):
        hiddens, gates = self.proj_in(tokens).chunk(2, dim = -1)

        hiddens = hiddens * F.gelu(gates)

        return self.proj_out(hiddens)

# main class

class Rigidformer(Module):
    def __init__(
        self,
        dim,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_expansion = 4.,
        num_register_tokens = 16
    ):
        super().__init__()

        layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )

            ff = SwiGluFeedforward(
                dim = dim,
                expansion_factor = ff_expansion
            )

            layers.append(ModuleList([attn, ff]))

        self.layers = layers
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

    def forward(
        self,
        tokens
    ):
        batch = tokens.shape[0]

        registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        tokens, inverse_pack_registers = pack_with_inverse((registers, tokens), 'b * d')

        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        _, tokens = inverse_pack_registers(tokens)

        return tokens
