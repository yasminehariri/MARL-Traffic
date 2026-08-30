import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN

from env.message_encoder import MessageEncoder

# Same 2x2 ring topology used in B2 -- kept consistent across conditions
NEIGHBORS = {
    "A0": ["A1", "B0"],
    "A1": ["A0", "B1"],
    "B0": ["A0", "B1"],
    "B1": ["A1", "B0"],
}

OBS_DIM = 9
MSG_DIM = 8
POLICY_INPUT_DIM = OBS_DIM + MSG_DIM  # 17


class CommPolicyNetwork(nn.Module):
    """Linear(17,64) -> ReLU -> Linear(64,4), per scope doc 5.5"""
    def __init__(self, input_dim=POLICY_INPUT_DIM, num_actions=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class E1CommModel(TorchModelV2, nn.Module):
    """
    RLlib-compatible model implementing E1's learned communication.
    Each forward pass:
      1. Looks up neighbours' messages from the PREVIOUS timestep (one-step delay)
      2. Averages them, concatenates with this agent's own observation
      3. Produces action logits
      4. Computes and caches this agent's OWN new message for neighbours to use next step
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.encoder = MessageEncoder(obs_dim=OBS_DIM, msg_dim=MSG_DIM)
        self.policy_net = CommPolicyNetwork(input_dim=POLICY_INPUT_DIM, num_actions=num_outputs)

        # Shared value function head (needed for PPO's critic)
        self.value_head = nn.Linear(POLICY_INPUT_DIM, 1)
        self._last_value = None

        # message_cache: agent_id -> most recently computed message (detached tensor)
        # Shared across all agents since this model instance IS the shared policy.
        self.message_cache = {aid: torch.zeros(MSG_DIM) for aid in NEIGHBORS.keys()}

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()  # shape: (batch, 9)
        agent_ids = input_dict.get("agent_id", None)

        batch_size = obs.shape[0]
        mean_neighbor_msgs = torch.zeros(batch_size, MSG_DIM)

        # Look up each sample's neighbours' cached messages (one-step delay)
        for i in range(batch_size):
            aid = agent_ids[i] if agent_ids is not None else None
            if aid in NEIGHBORS:
                neighbor_msgs = torch.stack([self.message_cache[n] for n in NEIGHBORS[aid]])
                mean_neighbor_msgs[i] = neighbor_msgs.mean(dim=0)
            # if agent_id unknown/missing, mean stays zero (safe fallback)

        policy_input = torch.cat([obs, mean_neighbor_msgs], dim=1)  # (batch, 17)
        action_logits = self.policy_net(policy_input)
        self._last_value = self.value_head(policy_input).squeeze(1)

        # Compute and cache each agent's OWN new message for neighbours to read next step
        new_messages = self.encoder(obs)  # (batch, 8)
        for i in range(batch_size):
            aid = agent_ids[i] if agent_ids is not None else None
            if aid in self.message_cache:
                self.message_cache[aid] = new_messages[i].detach()

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value