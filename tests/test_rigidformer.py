
import pytest

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer

    import torch

    x = torch.randn(1, 1024, 512)
    rigidformer = Rigidformer(512)

    out = rigidformer(x)
    assert x.shape == out.shape
