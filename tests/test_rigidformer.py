import torch

import pytest
param = pytest.mark.parametrize

@param('anchor_indices_given', (False, True))
def test_rigidformer(
    anchor_indices_given
):
    from rigidformer.rigidformer import Rigidformer


    object_tokens = torch.randn(2, 2, 256, 512)
    object_pos = torch.randn(2, 2, 256, 3)

    if anchor_indices_given:
        anchor_kwargs = dict(
            anchor_indices = torch.randint(0, 256, (2, 2, 4))
        )
    else:
        anchor_kwargs = dict(
            anchor_tokens = torch.randn(2, 2, 4, 512),
            anchor_pos = torch.randn(2, 2, 4, 3),
        )

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(512)

    anchor_pos_prev = torch.randn(2, 2, 4, 3)
    anchor_pos_next = torch.randn(2, 2, 4, 3)

    loss, loss_breakdown = rigidformer(object_tokens, delta_times = delta_times, object_pos = object_pos, anchor_pos_prev = anchor_pos_prev, anchor_pos_next = anchor_pos_next, **anchor_kwargs)
    loss.backward()

    pred = rigidformer(object_tokens, delta_times = delta_times, object_pos = object_pos, anchor_pos_prev = anchor_pos_prev, **anchor_kwargs)

    assert pred.anchor_next_positions.shape == (2, 2, 4, 3)
    assert pred.object_next_positions.shape == (2, 2, 256, 3)
