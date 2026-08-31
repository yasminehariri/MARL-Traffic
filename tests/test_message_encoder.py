import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from env.message_encoder import MessageEncoder


def test_message_shape():
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)
    dummy_obs = torch.randn(1, 9)
    message = encoder(dummy_obs)
    assert message.shape == (1, 8)


def test_message_shape_batch():
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)
    dummy_obs = torch.randn(32, 9)
    message = encoder(dummy_obs)
    assert message.shape == (32, 8)


def test_gradient_flow():
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)
    dummy_obs = torch.randn(1, 9)
    message = encoder(dummy_obs)
    target = torch.zeros_like(message)
    loss = ((message - target) ** 2).mean()

    fc1_grad = torch.autograd.grad(loss, encoder.fc1.weight, retain_graph=True)[0]
    fc2_grad = torch.autograd.grad(loss, encoder.fc2.weight, retain_graph=True)[0]

    assert fc1_grad is not None
    assert fc2_grad is not None
    assert torch.count_nonzero(fc1_grad).item() > 0
    assert torch.count_nonzero(fc2_grad).item() > 0


def test_neighbour_aggregation():
    msg_a = torch.tensor([1.0] * 8)
    msg_b = torch.tensor([3.0] * 8)
    neighbour_msgs = torch.stack([msg_a, msg_b])
    mean_msg = neighbour_msgs.mean(dim=0)

    assert mean_msg.shape == (8,)
    assert torch.allclose(mean_msg, torch.tensor([2.0] * 8))


if __name__ == "__main__":
    test_message_shape()
    test_message_shape_batch()
    test_gradient_flow()
    test_neighbour_aggregation()
    print("All tests PASSED")