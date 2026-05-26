import torch

import pytest
param = pytest.mark.parametrize

@param('fps', (False, True))
@param('test_rand_steps', (False, True))
@param('attn_residual_learned_pooling', (False, True))
@param('variable_object_lens', (False, True))
@param('variable_point_lens', (False, True))
@param('anchor_self_attn', (False, True))
def test_rigidformer(
    fps,
    test_rand_steps,
    attn_residual_learned_pooling,
    variable_object_lens,
    variable_point_lens,
    anchor_self_attn
):
    from rigidformer.rigidformer import Rigidformer, RigidformerRolloutWrapper, PointNet

    object_pos = torch.randn(2, 2, 256, 3)
    object_pos_prev = torch.randn(2, 2, 256, 3)
    object_pos_next = torch.randn(2, 2, 256, 3)
    vertex_properties = torch.randn(2, 2, 3)

    anchor_indices = torch.randint(0, 256, (2, 2, 4))

    delta_times = torch.randn(2)

    rigidformer = Rigidformer(
        512,
        hierarchical_encoder = PointNet(dim = 512, dim_out = 512),
        attn_residual_learned_pooling = attn_residual_learned_pooling,
        anchor_self_attn = anchor_self_attn
    )

    kwargs = dict()
    if not fps:
        kwargs.update(anchor_indices = anchor_indices)

    if variable_object_lens:
        kwargs.update(object_lens = torch.tensor([1, 2]))

    if variable_point_lens:
        kwargs.update(object_point_lens = torch.randint(128, 257, (2, 2)))

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

def test_pointnet():
    from rigidformer.rigidformer import PointNet

    features = torch.randn(2, 2, 256, 64)
    pos = torch.randn(2, 2, 256, 3)

    net = PointNet(dim = 64, dim_out = 128)
    out = net(features, pos)

    assert out.shape == (2, 2, 128)
