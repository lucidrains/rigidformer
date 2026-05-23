import torch

import pytest
param = pytest.mark.parametrize

def test_rigidformer():
    from rigidformer.rigidformer import Rigidformer

    object_pos = torch.randn(2, 2, 256, 3)
    object_pos_prev = torch.randn(2, 2, 256, 3)
    object_pos_next = torch.randn(2, 2, 256, 3)
    vertex_properties = torch.randn(2, 2, 3)

    anchor_indices = torch.randint(0, 256, (2, 2, 4))

    from einops.layers.torch import Reduce

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(
        512,
        hierarchical_encoder = Reduce('b no n d -> b no d', 'mean') # mock before building out pointnet++ and platonic transformer
    )

    loss, loss_breakdown = rigidformer(
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_pos = object_pos,
        object_pos_prev = object_pos_prev,
        object_pos_next = object_pos_next,
        anchor_indices = anchor_indices
    )
    loss.backward()

    pred = rigidformer(
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_pos = object_pos,
        object_pos_prev = object_pos_prev,
        anchor_indices = anchor_indices
    )

    assert pred.anchor_next_positions.shape == (2, 2, 4, 3)
    assert pred.object_next_positions.shape == (2, 2, 256, 3)
