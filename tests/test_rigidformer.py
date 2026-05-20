
import pytest

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer

    import torch

    x = torch.randn(2, 1024, 512)
    t = torch.randn(2)

    rigidformer = Rigidformer(512)

    out = rigidformer(x, t)
    assert x.shape == out.shape
