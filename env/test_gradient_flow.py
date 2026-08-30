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

torch.manual_seed(0)  # fix randomness so results are reproducible
encoder = MessageEncoder(obs_dim=9, msg_dim=8)
dummy_obs = torch.randn(1, 9)

message = encoder(dummy_obs)
target = torch.zeros_like(message)
loss = ((message - target) ** 2).mean()

grads = torch.autograd.grad(loss, [encoder.fc1.weight, encoder.fc2.weight])
fc1_grad, fc2_grad = grads

print("fc1 gradient — total nonzero entries:", torch.count_nonzero(fc1_grad).item(), "out of", fc1_grad.numel())
print("fc1 gradient — max absolute value:", fc1_grad.abs().max().item())
print("fc1 gradient — mean absolute value:", fc1_grad.abs().mean().item())

print("\nfc2 gradient — total nonzero entries:", torch.count_nonzero(fc2_grad).item(), "out of", fc2_grad.numel())
print("fc2 gradient — max absolute value:", fc2_grad.abs().max().item())