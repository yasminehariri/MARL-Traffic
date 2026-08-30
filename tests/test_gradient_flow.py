import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from env.message_encoder import MessageEncoder


def test_gradient_flow():
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)
    dummy_obs = torch.randn(1, 9)

    message = encoder(dummy_obs)
    loss = message.sum()
    loss.backward()

    grad = encoder.fc1.weight.grad
    assert grad is not None, "No gradient reached fc1 — gradient flow is BROKEN"
    assert grad.abs().sum() > 0, "Gradient reached fc1 but is all zeros — no real signal"
    print("test_gradient_flow PASSED")

if __name__ == "__main__":
    test_gradient_flow()