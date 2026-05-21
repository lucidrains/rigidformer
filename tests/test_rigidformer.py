
import pytest

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer

    import torch

    object_tokens = torch.randn(2, 256, 512)
    anchor_tokens = torch.randn(2, 16, 512)

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(512)

    anchor_pos_prev = torch.randn(2, 16, 3)
    anchor_pos = anchor_pos_prev + 1.
    anchor_pos_next = anchor_pos + 2.

    loss, loss_breakdown = rigidformer(object_tokens, anchor_tokens, delta_times = delta_times, anchor_pos = anchor_pos, anchor_pos_prev = anchor_pos_prev, anchor_pos_next = anchor_pos_next)
    loss.backward()

    pred = rigidformer(object_tokens, anchor_tokens, delta_times = delta_times, anchor_pos = anchor_pos, anchor_pos_prev = anchor_pos_prev)

    assert pred.position.shape == (2, 16, 3)
