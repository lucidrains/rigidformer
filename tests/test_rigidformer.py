
import pytest

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer

    import torch

    object_tokens = torch.randn(2, 2, 256, 512)
    object_pos = torch.randn(2, 2, 256, 3)

    anchor_tokens = torch.randn(2, 2, 4, 512)
    anchor_pos = torch.randn(2, 2, 4, 3)

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(512)

    anchor_pos_prev = anchor_pos - .5
    anchor_pos_next = anchor_pos + 2.

    loss, loss_breakdown = rigidformer(object_tokens, anchor_tokens, delta_times = delta_times, object_pos = object_pos, anchor_pos = anchor_pos, anchor_pos_prev = anchor_pos_prev, anchor_pos_next = anchor_pos_next)
    loss.backward()

    pred = rigidformer(object_tokens, anchor_tokens, delta_times = delta_times, object_pos = object_pos, anchor_pos = anchor_pos, anchor_pos_prev = anchor_pos_prev)

    assert pred.anchor_next_positions.shape == (2, 2, 4, 3)
    assert pred.object_next_positions.shape == (2, 2, 256, 3)
