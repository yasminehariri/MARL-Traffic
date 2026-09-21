import torch
from env.comm_model import E1CommModel
from gymnasium import spaces
import numpy as np

obs_space = spaces.Box(low=0.0, high=1.0, shape=(49,), dtype=np.float32)
action_space = spaces.Discrete(4)
model = E1CommModel(obs_space, action_space, num_outputs=4, model_config={}, name="neighbor_test")

# Simulate 3 agents with DIFFERENT real neighbor counts: 2, 3, and 4
def make_fake_obs(num_real_neighbors):
    own = torch.randn(9)
    neighbor_block = torch.zeros(36)
    mask = torch.zeros(4)
    for i in range(num_real_neighbors):
        neighbor_block[i*9:(i+1)*9] = torch.randn(9)
        mask[i] = 1.0
    return torch.cat([own, neighbor_block, mask])

batch = torch.stack([make_fake_obs(2), make_fake_obs(3), make_fake_obs(4)])
action_logits, _ = model.forward({"obs": batch}, state=[], seq_lens=None)

print("Action logits shape:", action_logits.shape)  # expect (3, 4)
print("No NaN?", not torch.isnan(action_logits).any().item())

# Gradient check -- same rigor as every previous architecture change
fake_loss = action_logits.pow(2).mean()
grad = torch.autograd.grad(fake_loss, model.encoder.fc1.weight, retain_graph=True, allow_unused=True)[0]
print("Gradient reaches encoder?", grad is not None)
if grad is not None:
    print("Nonzero entries:", torch.count_nonzero(grad).item(), "/", grad.numel())