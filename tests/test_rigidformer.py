import torch

import pytest
param = pytest.mark.parametrize

@param('fps', (False, True))
@param('test_rand_steps', (False, True))
@param('attn_residual_learned_pooling', (False, True))
def test_rigidformer(
    fps,
    test_rand_steps,
    attn_residual_learned_pooling
):
    from rigidformer.rigidformer import Rigidformer, RigidformerRolloutWrapper

    object_pos = torch.randn(2, 2, 256, 3)
    object_pos_prev = torch.randn(2, 2, 256, 3)
    object_pos_next = torch.randn(2, 2, 256, 3)
    vertex_properties = torch.randn(2, 2, 3)

    anchor_indices = torch.randint(0, 256, (2, 2, 4))

    from einops.layers.torch import Reduce

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(
        512,
        hierarchical_encoder = Reduce('b no n d -> b no d', 'mean'), # mock before building out pointnet++ and platonic transformer
        attn_residual_learned_pooling = attn_residual_learned_pooling
    )

    kwargs = dict()
    if not fps:
        kwargs.update(anchor_indices = anchor_indices)

    loss, loss_breakdown = rigidformer(
        delta_times = delta_times,
        vertex_properties = vertex_properties,
        object_pos = object_pos,
        object_pos_prev = object_pos_prev,
        object_pos_next = object_pos_next,
        **kwargs
    )

    loss.backward()

    rollout_wrapper = RigidformerRolloutWrapper(rigidformer)

    if test_rand_steps:
        delta_times_input = rollout_wrapper.rand_steps(
            delta_times = delta_times,
            num_rand_substeps = 4,
            max_step_weight = 3
        )
        assert delta_times_input.shape == (2, 4)
        assert torch.allclose(delta_times_input.sum(dim = -1), delta_times)

        num_steps = None
    else:
        delta_times_input = delta_times
        num_steps = 4

    object_positions = rollout_wrapper(
        delta_times = delta_times_input,
        num_steps = num_steps,
        vertex_properties = vertex_properties,
        object_positions = [object_pos_prev, object_pos],
        **kwargs
    )

    assert len(object_positions) == 6

    last_position = object_positions[-1]

    assert last_position.shape == (2, 2, 256, 3)
