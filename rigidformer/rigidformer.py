from __future__ import annotations
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor, Tensor
from torch.nn import Module, ModuleList, Linear, Parameter

import einx
from einops import einsum, rearrange, repeat, pack
from einops.layers.torch import Rearrange

from torch_einops_utils import pack_with_inverse, maybe, pad_left_at_dim

import roma

# constants

Predictions = namedtuple('Predictions', ('acceleration', 'anchor_next_positions', 'object_next_positions'))

Losses = namedtuple('Losses', ('acceleration', 'position'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def mean_and_expand_back(t, dim):
    shape = t.shape
    t = t.mean(dim = dim, keepdim = True)
    return t.expand(shape)

# 3d axial rotary embeddings
# the anchor rope mentioned is simply where they mean pool rotary embeddings for the 4 anchors, iiuc

class AxialRotaryEmbeddings(Module):
    def __init__(
        self,
        dim,
        omega = 10_000
    ):
        super().__init__()
        assert divisible_by(dim, 6), f'{dim} must be divisible by 6'
        inv_freq = omega ** (-torch.arange(0, dim, 6).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    def forward(
        self,
        pos, # (... 3)
    ):
        freqs = einsum(pos, self.inv_freq, '... p, f -> ... p f')
        freqs = rearrange(freqs, 'b ... p f -> b 1 ... (p f)')
        return cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# film

class FiLM(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine = False)

        self.to_gamma_beta = Linear(dim_cond, dim * 2, bias = False)
        nn.init.zeros_(self.to_gamma_beta.weight)

    def forward(
        self,
        tokens,
        cond
    ):
        normed = self.norm(tokens)

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)

        scaled = einx.add('b n d, b d', normed, gamma + 1.)
        return einx.add('b n d, b d', scaled, beta)

# attention residual pooler

class AttentionResidualPool(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.queries = nn.Parameter(torch.randn(dim) * 1e-2)

        self.key_rmsnorm = nn.RMSNorm(dim)

    def forward(
        self,
        hiddens: list[Tensor], # [(b n d)]
    ):
        assert len(hiddens) > 0

        layer_hiddens = stack(hiddens, dim = -2)

        # queries, keys, values

        q, k, v = self.queries, self.key_rmsnorm(layer_hiddens), layer_hiddens

        q = q * self.scale

        # attention

        sim = einsum(q, k, 'd, b n l d -> b n l')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b n l, b n l d -> b n d')

        return out

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_queries_gates = Linear(dim, dim_inner * 2, bias = False)
        self.to_keys_values = Linear(dim, dim_inner * 2, bias = False)

        self.to_out = Linear(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # qk rmsnorm

        self.has_qk_rmsnorm = qk_rmsnorm

        self.qk_rmsnorm = nn.RMSNorm(dim_head, elementwise_affine = False)
        self.qk_rmsnorm_scales = nn.Parameter(torch.ones(2, heads, dim_head))

    def forward(
        self,
        tokens,
        context = None,
        rotary_pos_emb = None,
        context_rotary_pos_emb = None
    ):

        context = default(context, tokens)

        queries, gates, keys, values = (
            *self.to_queries_gates(tokens).chunk(2, dim = -1),
            *self.to_keys_values(context).chunk(2, dim = -1)
        )

        queries, keys, values = (self.split_heads(t) for t in (queries, keys, values))

        if self.has_qk_rmsnorm:
            queries, keys = tuple(self.qk_rmsnorm(t) for t in (queries, keys))
            queries, keys = tuple(einx.multiply('b h n d, h d', t, scale) for t, scale in zip((queries, keys), self.qk_rmsnorm_scales))

        if exists(rotary_pos_emb):
            context_rotary_pos_emb = default(context_rotary_pos_emb, rotary_pos_emb)

            queries = apply_rotary_pos_emb(rotary_pos_emb, queries)
            keys = apply_rotary_pos_emb(context_rotary_pos_emb, keys)

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j') * self.scale

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
        dim_head = 192,
        heads = 6,
        ff_expansion = 2.5,
        num_register_tokens = 16,
        object_self_attn_depth = 4,
        anchor_cross_attn_depth = 4,
        object_hidden_layers: tuple[int, ...] = (0, 1, 2, 4),  # the hidden object layer outputs that the anchor decoder cross attends to
        pos_loss_weight = 10.,
        acc_loss_weight = 1.,
        axial_rope_kwargs: dict = dict(
            omega = 10_000
        ),
        register_pos = -1000. # unsure what position to give the registers, so just make it far away
    ):
        super().__init__()

        # rotary embeddings

        self.rope_3d = AxialRotaryEmbeddings(dim_head, **axial_rope_kwargs)

        self.register_pos = register_pos # todo - spend some time building / vibing a custom kernel for both rope and pope to be able to omit rotary for certain tokens (registers / cls etc)

        # object self attention related

        layers = ModuleList([])

        for i in range(object_self_attn_depth):
            is_last = i == (object_self_attn_depth - 1)

            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )

            ff = SwiGluFeedforward(
                dim = dim,
                expansion_factor = ff_expansion
            )

            attn_film = FiLM(dim, 2)
            ff_film = FiLM(dim, 2)

            attn_residual = AttentionResidualPool(dim) if not is_last else None

            layers.append(ModuleList([attn_film, attn, ff_film, ff, attn_residual]))

        self.self_attn_layers = layers

        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # anchor cross attention related

        assert object_self_attn_depth in object_hidden_layers, '`object_hidden_layers` should attend to the output of the object transformer ({object_self_attn_depth})'
        assert all([0 <= l <= object_self_attn_depth for l in object_hidden_layers])
        assert len(object_hidden_layers) == anchor_cross_attn_depth, 'length of `object_hidden_layers` must be equal to the depth of the anchor cross attention transformer'

        layers = ModuleList([])

        for _ in range(anchor_cross_attn_depth):

            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )

            ff = SwiGluFeedforward(
                dim = dim,
                expansion_factor = ff_expansion
            )

            attn_film = FiLM(dim, 2)
            ff_film = FiLM(dim, 2)

            attn_residual = AttentionResidualPool(dim)

            layers.append(ModuleList([attn_film, attn, ff_film, ff, attn_residual]))

        self.cross_attn_layers = layers

        self.object_hidden_layers = object_hidden_layers

        self.to_acc_pred = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, 3, bias = False)
        )

        # loss related

        self.loss_fn = nn.SmoothL1Loss()

        self.pos_loss_weight = pos_loss_weight
        self.acc_loss_weight = acc_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        object_tokens,           # (b no n d)
        anchor_tokens,           # (b no na d)
        *,
        delta_times,             # (b)
        object_pos,              # (b no n 3)
        anchor_pos,              # (b no na 3)
        anchor_pos_prev = None,  # (b no na 3)
        anchor_pos_next = None   # (b no na 3)
    ):
        batch, max_num_objects = object_tokens.shape[:2]

        # time conditioning

        delta_times = delta_times.float()
        delta_times_squared = delta_times.pow(2)
        time_cond = stack((delta_times, delta_times_squared), dim = -1) # t and t^2

        # register tokens

        registers = repeat(self.register_tokens, 'r d -> b no r d', b = batch, no = max_num_objects)

        object_tokens, inverse_pack_registers = pack_with_inverse((registers, object_tokens), 'b no * d')

        object_tokens, inverse_pack_num_objects = pack_with_inverse(object_tokens, 'b * d')

        # object rotary embeddings

        object_rotary_pos_emb = self.rope_3d(object_pos)
        object_rotary_pos_emb_with_registers = pad_left_at_dim(object_rotary_pos_emb, self.num_register_tokens, dim = -2, value = self.register_pos)

        object_rotary_pos_emb_with_registers, object_rotary_pos_emb = (rearrange(t, 'b h no na f -> b h (no na) f') for t in (object_rotary_pos_emb_with_registers, object_rotary_pos_emb))

        # handle the "ARoPE" - all anchors for the same object share the same mean pooled positional embedding, it seems

        anchor_rotary_pos_emb = self.rope_3d(anchor_pos)
        anchor_rotary_pos_emb = mean_and_expand_back(anchor_rotary_pos_emb, dim = -2) # (b no na f)
        anchor_rotary_pos_emb = rearrange(anchor_rotary_pos_emb, 'b h no na f -> b h (no na) f')

        # object self attention

        object_hiddens = [object_tokens]

        for attn_film, attn, ff_film, ff, attn_residual in self.self_attn_layers:

            filmed = attn_film(object_tokens, time_cond)

            object_tokens = attn(filmed, rotary_pos_emb = object_rotary_pos_emb_with_registers) + object_tokens

            filmed = ff_film(object_tokens, time_cond)
            object_tokens = ff(object_tokens) + object_tokens

            object_hiddens.append(object_tokens)

            object_tokens = maybe(attn_residual)(object_hiddens)

        # anchor cross attention

        anchor_tokens, inverse_pack_objects_num_anchors = pack_with_inverse(anchor_tokens, 'b * d')

        anchor_hiddens = [anchor_tokens]

        object_contexts = [object_hiddens[layer] for layer in self.object_hidden_layers] # gather all the object self attention hidden layers for cross attending

        for (attn_film, attn, ff_film, ff, attn_residual), object_context in zip(self.cross_attn_layers, object_contexts):

            object_context = inverse_pack_num_objects(object_context)
            _, object_context = inverse_pack_registers(object_context) # remove register tokens
            object_context = rearrange(object_context, 'b na n d -> b (na n) d')

            filmed = attn_film(anchor_tokens, time_cond)
            anchor_tokens = attn(filmed, rotary_pos_emb = anchor_rotary_pos_emb, context_rotary_pos_emb = object_rotary_pos_emb, context = object_context) + anchor_tokens

            filmed = ff_film(anchor_tokens, time_cond)
            anchor_tokens = ff(filmed) + anchor_tokens

            anchor_hiddens.append(anchor_tokens)

            anchor_tokens = attn_residual(anchor_hiddens)

        anchor_tokens = inverse_pack_objects_num_anchors(anchor_tokens)

        pred_acc = self.to_acc_pred(anchor_tokens)

        assert exists(anchor_pos) == exists(anchor_pos_prev)

        pred = pred_acc

        # early return prediction if ground truth not passed in

        return_loss = exists(anchor_pos) and exists(anchor_pos_next)

        # calculate predicted next position if not returning loss - kabsch and then return next anchors and object positions

        pred_anchor_pos_next = 2 * anchor_pos - anchor_pos_prev + einx.multiply('b ..., b', pred_acc, delta_times_squared) # verlet

        if not return_loss:

            R, T = roma.rigid_points_registration(anchor_pos, pred_anchor_pos_next)

            rigid_anchor_pos_next = einx.add('b no c, b no na c', T, einsum(anchor_pos, R, 'b no na c1, b no c2 c1 -> b no na c2'))
            rigid_object_pos_next = einx.add('b no c, b no n c', T, einsum(object_pos, R, 'b no n c1, b no c2 c1 -> b no n c2'))

            pred = Predictions(pred_acc, rigid_anchor_pos_next, rigid_object_pos_next)

        if not return_loss:
            return pred

        delta_times_squared = repeat(delta_times_squared, 'b -> b 1 1 1')

        # calculate loss with roma + kabsch

        anchor_acc = (anchor_pos_next - 2 * anchor_pos + anchor_pos_prev) / delta_times_squared

        # handle acceleration

        R = roma.rigid_vectors_registration(pred_acc, anchor_acc)

        pred_acc_rigid = einsum(pred_acc, R, 'b no na c1, b no c2 c1 -> b no na c2')

        # handle points

        R, T = roma.rigid_points_registration(pred_anchor_pos_next, anchor_pos_next)

        pred_pos_next_rotated = einsum(pred_anchor_pos_next, R, 'b no na c1, b no c2 c1 -> b no na c2')
        pred_pos_next_translated = einx.add('b no na c, b no c', pred_pos_next_rotated, T)

        pred_pos_next_rigid = pred_pos_next_translated

        # losses, using smooth l1 loss, which they had a lot of success with it appears

        loss_fn = self.loss_fn

        acc_loss = loss_fn(pred_acc, anchor_acc) + loss_fn(pred_acc_rigid, anchor_acc)
        pos_loss = loss_fn(pred_anchor_pos_next, anchor_pos_next) + loss_fn(pred_pos_next_rigid, anchor_pos_next)

        total_loss = (
            acc_loss * self.acc_loss_weight +
            pos_loss * self.pos_loss_weight
        )

        return total_loss, Losses(acc_loss, pos_loss)
