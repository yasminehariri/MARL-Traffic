import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from env.message_encoder import MessageEncoder

def test_message_shape():
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)
    dummy_obs = torch.randn(1, 9)
    message = encoder(dummy_obs)
    'assert: the output shape MUST be (1, 8) — 1 sample of 8 numbers.'
    assert message.shape == (1, 8), f"Expected shape (1, 8) but got {message.shape}"
    print("test_message_shape PASSED")

if __name__ == "__main__":
    test_message_shape()