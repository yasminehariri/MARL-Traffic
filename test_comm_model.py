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
    """
    Combines own obs (9) + averaged neighbour messages (8) = 17 floats -> 4 action logits.
    Matches scope doc 5.5: Linear(17,64) -> ReLU -> Linear(64,4)
    """
    def __init__(self, input_dim=17, num_actions=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class CommModel:
    """
    Standalone (no-RLlib) version of the one-step-delayed communication logic,
    for isolated testing before integration.
    """
    def __init__(self, agent_ids, obs_dim=9, msg_dim=8):
        self.agent_ids = agent_ids
        self.encoder = MessageEncoder(obs_dim, msg_dim)
        self.policy = CommPolicyNetwork(input_dim=obs_dim + msg_dim, num_actions=4)
        # message_cache holds each agent's most recently computed message
        # initialized to zero vectors (agents have "nothing to say" before the first real step)
        self.message_cache = {aid: torch.zeros(msg_dim) for aid in agent_ids}

    def forward(self, agent_id, obs, neighbor_ids):
        # 1. Gather neighbours' messages from LAST timestep (one-step delay)
        neighbor_msgs = torch.stack([self.message_cache[n] for n in neighbor_ids])
        mean_neighbor_msg = neighbor_msgs.mean(dim=0)

        # 2. Combine own obs with averaged neighbour messages -> policy input
        policy_input = torch.cat([obs, mean_neighbor_msg])
        action_logits = self.policy(policy_input)

        # 3. Compute THIS agent's own new message (from its own obs) and cache it
        #    for neighbours to read on the NEXT timestep
        new_message = self.encoder(obs)
        self.message_cache[agent_id] = new_message.detach()  # detach: cache stores values, not graph

        return action_logits, new_message


# ---- Test the logic with a tiny 2-agent example ----
agent_ids = ["A0", "A1"]
model = CommModel(agent_ids)

obs_A0 = torch.randn(9)
obs_A1 = torch.randn(9)

print("=== Timestep 1 ===")
logits_A0, msg_A0 = model.forward("A0", obs_A0, neighbor_ids=["A1"])
logits_A1, msg_A1 = model.forward("A1", obs_A1, neighbor_ids=["A0"])
print("A0 action logits:", logits_A0)
print("A0's neighbour (A1) message was ZERO (no prior message yet) — expected at t=1")

print("\n=== Timestep 2 ===")
obs_A0_t2 = torch.randn(9)
obs_A1_t2 = torch.randn(9)
logits_A0_t2, msg_A0_t2 = model.forward("A0", obs_A0_t2, neighbor_ids=["A1"])
print("A0 action logits (t=2):", logits_A0_t2)
print("This time A0 should have used A1's REAL message from timestep 1, not zero")
print("A1's cached message from t=1:", msg_A1)