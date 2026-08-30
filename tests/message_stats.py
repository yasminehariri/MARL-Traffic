import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from env.message_encoder import MessageEncoder


def log_message_statistics(num_samples=1000):
    """
    Runs random observations through the MessageEncoder and logs
    message-vector statistics (norm, mean, variance) to confirm the
    messages are neither all-zero nor all-saturated/identical.
    Satisfies Week 5 Action 5: 'Log message vector norms and variance.'
    """
    encoder = MessageEncoder(obs_dim=9, msg_dim=8)

    random_obs = torch.randn(num_samples, 9)
    messages = encoder(random_obs)

    norms = messages.norm(dim=1)          # length of each message vector
    variance = messages.var(dim=0)        # variance per message dimension

    print(f"Samples: {num_samples}")
    print(f"Message norm  -> mean: {norms.mean().item():.4f}, min: {norms.min().item():.4f}, max: {norms.max().item():.4f}")
    print(f"Message variance per dim: {variance.detach().numpy()}")
    print(f"Overall message variance: {messages.var().item():.4f}")

    # Basic health checks
    assert messages.abs().sum() > 0, "Messages are all zero — encoder produces no signal"
    assert messages.var().item() > 0, "Messages have zero variance — all identical (collapsed)"
    print("Message statistics OK — messages are non-zero and varied")


if __name__ == "__main__":
    log_message_statistics()