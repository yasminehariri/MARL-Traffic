import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from env.message_encoder import MessageEncoder

OBS_DIM = 9
MAX_NEIGHBORS = 4
MSG_DIM = 8
POLICY_INPUT_DIM = OBS_DIM + MSG_DIM  # 17, unchanged


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

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        full_obs = input_dict["obs"].float()  # (batch, 49)

        own_obs = full_obs[:, :OBS_DIM]                                          # (batch, 9)
        neighbor_block = full_obs[:, OBS_DIM:OBS_DIM + MAX_NEIGHBORS * OBS_DIM]   # (batch, 36)
        mask = full_obs[:, OBS_DIM + MAX_NEIGHBORS * OBS_DIM:]                    # (batch, 4)

        neighbor_obs = neighbor_block.view(-1, MAX_NEIGHBORS, OBS_DIM)            # (batch, 4, 9)

        # Compute messages for all 4 slots (padding slots get a message too,
        # from a zero-input -- harmless, since the mask excludes them next)
        neighbor_msgs = self.encoder(neighbor_obs.reshape(-1, OBS_DIM))           # (batch*4, 8)
        neighbor_msgs = neighbor_msgs.view(-1, MAX_NEIGHBORS, MSG_DIM)            # (batch, 4, 8)

        # Masked mean: only average over REAL neighbours, never padding
        mask_expanded = mask.unsqueeze(-1)                                       # (batch, 4, 1)
        num_real = mask.sum(dim=1, keepdim=True).clamp(min=1.0)                  # (batch, 1) -- clamp avoids div-by-zero
        mean_neighbor_msg = (neighbor_msgs * mask_expanded).sum(dim=1) / num_real # (batch, 8)

        policy_input = torch.cat([own_obs, mean_neighbor_msg], dim=1)            # (batch, 17)
        action_logits = self.policy_net(policy_input)
        self._last_value = self.value_head(policy_input).squeeze(1)

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value