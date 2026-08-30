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
        return self.fc2(x)


class CommPolicyNetwork(nn.Module):
    def __init__(self, input_dim=17, num_actions=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


torch.manual_seed(0)
encoder = MessageEncoder()
policy = CommPolicyNetwork()

# Simulate: agent A0 receives a (non-zero, "already computed") neighbour message,
# combines it with its own obs, and produces an action.
own_obs = torch.randn(9)
neighbor_message = torch.randn(8)  # pretend this came from a neighbour's encoder last step

policy_input = torch.cat([own_obs, neighbor_message])
action_logits = policy(policy_input)

# Also: this agent computes its OWN new message from its own obs (for its neighbours next step)
own_new_message = encoder(own_obs)

# Combine both outputs into a single fake "loss" -- pretending both the action
# AND the message need to improve, exactly like real training would optimize both.
fake_action_loss = action_logits.pow(2).mean()
fake_message_loss = own_new_message.pow(2).mean()
total_loss = fake_action_loss + fake_message_loss

# Check: does gradient reach BOTH the policy network's weights AND the encoder's weights?
policy_grad = torch.autograd.grad(total_loss, policy.fc1.weight, retain_graph=True)[0]
encoder_grad = torch.autograd.grad(total_loss, encoder.fc1.weight, retain_graph=True)[0]

print("Policy fc1 gradient — nonzero entries:", torch.count_nonzero(policy_grad).item(), "/", policy_grad.numel())
print("Policy fc1 gradient — max abs value:", policy_grad.abs().max().item())

print("\nEncoder fc1 gradient — nonzero entries:", torch.count_nonzero(encoder_grad).item(), "/", encoder_grad.numel())
print("Encoder fc1 gradient — max abs value:", encoder_grad.abs().max().item())