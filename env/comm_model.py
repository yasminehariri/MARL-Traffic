import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from env.message_encoder import MessageEncoder

NEIGHBORS = {
    "A0": ["A1", "B0"],
    "A1": ["A0", "B1"],
    "B0": ["A0", "B1"],
    "B1": ["A1", "B0"],
}

# Order MUST match AGENT_ID_ONEHOT in rllib_env_e1.py
AGENT_ORDER = ["A0", "A1", "B0", "B1"]

OBS_DIM = 9
AGENT_ID_DIM = 4
MSG_DIM = 8
POLICY_INPUT_DIM = OBS_DIM + MSG_DIM  # 17 -- policy still only sees obs + messages, not the ID tag


class CommPolicyNetwork(nn.Module):
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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.encoder = MessageEncoder(obs_dim=OBS_DIM, msg_dim=MSG_DIM)
        self.policy_net = CommPolicyNetwork(input_dim=POLICY_INPUT_DIM, num_actions=num_outputs)
        self.value_head = nn.Linear(POLICY_INPUT_DIM, 1)
        self._last_value = None

        self.message_cache = {aid: torch.zeros(MSG_DIM) for aid in AGENT_ORDER}

    def _decode_agent_ids(self, id_onehot_batch):
        """Convert one-hot rows back into agent ID strings, e.g. [1,0,0,0] -> 'A0'."""
        indices = torch.argmax(id_onehot_batch, dim=1)
        return [AGENT_ORDER[i] for i in indices.tolist()]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        full_obs = input_dict["obs"].float()  # shape: (batch, 13)

        real_obs = full_obs[:, :OBS_DIM]              # first 9 columns: real traffic obs
        id_onehot = full_obs[:, OBS_DIM:]              # last 4 columns: agent ID tag

        agent_ids = self._decode_agent_ids(id_onehot)
        batch_size = real_obs.shape[0]

        mean_neighbor_msgs = torch.zeros(batch_size, MSG_DIM)
        for i in range(batch_size):
            aid = agent_ids[i]
            neighbor_msgs = torch.stack([self.message_cache[n] for n in NEIGHBORS[aid]])
            mean_neighbor_msgs[i] = neighbor_msgs.mean(dim=0)

        policy_input = torch.cat([real_obs, mean_neighbor_msgs], dim=1)  # (batch, 17)
        action_logits = self.policy_net(policy_input)
        self._last_value = self.value_head(policy_input).squeeze(1)

        new_messages = self.encoder(real_obs)  # encoder only sees REAL obs, not the ID tag
        for i in range(batch_size):
            aid = agent_ids[i]
            self.message_cache[aid] = new_messages[i].detach()

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value