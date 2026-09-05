import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from env.message_encoder import MessageEncoder

OBS_DIM = 9
NUM_NEIGHBORS = 2
MSG_DIM = 8
POLICY_INPUT_DIM = OBS_DIM + MSG_DIM

DIVERSITY_WEIGHT = 1.0  # how strongly to penalize similarity -- starting value, tunable


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
        self._last_own_messages = None  # NEW: stores this batch's messages for custom_loss()

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        full_obs = input_dict["obs"].float()
        device = full_obs.device

        own_obs = full_obs[:, :OBS_DIM]
        neighbor_obs = full_obs[:, OBS_DIM:]
        neighbor_obs = neighbor_obs.view(-1, NUM_NEIGHBORS, OBS_DIM)

        neighbor_msgs = self.encoder(neighbor_obs.reshape(-1, OBS_DIM))
        neighbor_msgs = neighbor_msgs.view(-1, NUM_NEIGHBORS, MSG_DIM)
        mean_neighbor_msg = neighbor_msgs.mean(dim=1)

        policy_input = torch.cat([own_obs, mean_neighbor_msg], dim=1)
        action_logits = self.policy_net(policy_input)
        self._last_value = self.value_head(policy_input).squeeze(1)

        # Store THIS agent's own message (computed fresh from own_obs) for the diversity penalty
        self._last_own_messages = self.encoder(own_obs)  # (batch, 8)

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._last_value

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """
        Adds a diversity penalty: discourages different samples' messages
        (within this training batch) from being too similar to each other,
        directly counteracting the collapse we measured (mean cosine sim 0.9991).
        """
        if self._last_own_messages is None or self._last_own_messages.shape[0] < 2:
            return policy_loss  # not enough messages to compare, skip penalty safely

        messages = self._last_own_messages  # (batch, 8)
        normalized = F.normalize(messages, dim=1)  # unit vectors, so dot product = cosine similarity

        # Pairwise cosine similarity matrix: (batch, batch)
        similarity_matrix = torch.matmul(normalized, normalized.T)

        batch_size = similarity_matrix.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        off_diagonal_similarities = similarity_matrix[mask]

        mean_similarity = off_diagonal_similarities.mean()
        diversity_penalty = DIVERSITY_WEIGHT * (mean_similarity ** 2)  # L2-style: square the similarity

        if isinstance(policy_loss, list):
            return [pl + diversity_penalty for pl in policy_loss]
        return policy_loss + diversity_penalty