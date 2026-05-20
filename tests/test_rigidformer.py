
import pytest

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer, Attention

    import torch

    x = torch.randn(1, 1024, 512)
    attn = Attention(512)

    out = attn(x)
    assert x.shape == out.shape
