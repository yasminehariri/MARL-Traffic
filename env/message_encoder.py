import torch
import torch.nn as nn

class MessageEncoder(nn.Module):
    def __init__(self, obs_dim=9, msg_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, msg_dim)

    def forward(self, obs):
        x = self.fc1(obs)
        x = self.relu(x)
        msg = self.fc2(x)
        return msg

# Quick check: does it accept a 9-dim input and produce an 8-dim output?
encoder = MessageEncoder(obs_dim=9, msg_dim=8)
dummy_obs = torch.randn(1, 9)  # 1 sample, 9 features (batch of 1)
message = encoder(dummy_obs)

print("Input shape:", dummy_obs.shape)
print("Output shape:", message.shape)
print("Output values:", message)