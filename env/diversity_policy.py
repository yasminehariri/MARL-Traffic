import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import override

from env.comm_model import OBS_DIM

DIVERSITY_WEIGHT = 1.0


class E1DiversityPPOPolicy(PPOTorchPolicy):
    """
    PPO policy with an added diversity penalty, computed directly from the
    real training batch -- not cached state -- to counteract the message
    collapse found via cosine similarity (0.9991 measured on the un-fixed
    checkpoint).
    """

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        base_loss = super().loss(model, dist_class, train_batch)

        obs = train_batch["obs"].float()
        own_obs = obs[:, :OBS_DIM]

        messages = model.encoder(own_obs)

        if messages.shape[0] < 2:
            return base_loss

        norms = messages.norm(dim=1)
        if (norms < 1e-6).any():
            return base_loss

        normalized = F.normalize(messages, dim=1, eps=1e-6)
        similarity_matrix = torch.matmul(normalized, normalized.T)

        batch_size = similarity_matrix.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        off_diagonal_similarities = similarity_matrix[mask]
        mean_similarity = off_diagonal_similarities.mean()

        diversity_penalty = DIVERSITY_WEIGHT * (mean_similarity ** 2)
        if not torch.isfinite(diversity_penalty):
            return base_loss

        return base_loss + diversity_penalty