import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from env.message_encoder import MessageEncoder

OBS_DIM = 9
NUM_NEIGHBORS = 2
MSG_DIM = 8
POLICY_INPUT_DIM = OBS_DIM + MSG_DIM  # 17, same as before


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
    """
    Fully stateless communication model. Every forward() call independently
    recomputes messages for this agent AND its neighbours from their raw
    observations (bundled into the input by MARLTrafficEnvE1), using the
    SAME encoder weights for everyone (shared policy / parameter sharing).
    No cache, no cross-call memory -- safe under RLlib's shuffled,
    multi-epoch minibatch training.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.encoder = MessageEncoder(obs_dim=OBS_DIM, msg_dim=MSG_DIM)
        self.policy_net = CommPolicyNetwork(input_dim=POLICY_INPUT_DIM, num_actions=num_outputs)
        self.value_head = nn.Linear(POLICY_INPUT_DIM, 1)
        self._last_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        full_obs = input_dict["obs"].float()  # shape: (batch, 27)

        own_obs = full_obs[:, :OBS_DIM]                                    # (batch, 9)
        neighbor_obs = full_obs[:, OBS_DIM:]                                # (batch, 18)
        neighbor_obs = neighbor_obs.view(-1, NUM_NEIGHBORS, OBS_DIM)        # (batch, 2, 9)

        # Recompute neighbours' messages FRESH, right now, using the SAME encoder
        # that's being trained -- this is what keeps everything within one
        # self-contained computation graph, with nothing left over for a later
        # backward pass to conflict with.
        neighbor_msgs = self.encoder(neighbor_obs.reshape(-1, OBS_DIM))     # (batch*2, 8)
        neighbor_msgs = neighbor_msgs.view(-1, NUM_NEIGHBORS, MSG_DIM)      # (batch, 2, 8)
        mean_neighbor_msg = neighbor_msgs.mean(dim=1)                       # (batch, 8)

        policy_input = torch.cat([own_obs, mean_neighbor_msg], dim=1)      # (batch, 17)
        action_logits = self.policy_net(policy_input)
        self._last_value = self.value_head(policy_input).squeeze(1)

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value