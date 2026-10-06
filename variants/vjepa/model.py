"""V-JEPA (Video-JEPA): context encoder, EMA target encoder + predictor."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from jepa import ema, encoder, predictor


class VJEPA(nn.Module):
    """V-JEPA. Context encoder, target encoder + predictor."""

    def __init__(
            self,
            image_size: int = 32,
            patch_size: int = 4,
            num_frames: int = 16,
            tubelet_size: int = 2,
            embedding_dimension: int = 192,
            depth: int = 6,
            num_heads: int = 3,
            predictor_dimension: int = 96,
            predictor_depth: int = 4,
    ) -> None:
        super().__init__()

        self.context_encoder = encoder.VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dimension=embedding_dimension,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )

        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        self.predictor = predictor.Predictor(
            self.context_encoder.grid_size,
            embedding_dimension=embedding_dimension,
            predictor_dimension=predictor_dimension,
            depth=predictor_depth,
            num_heads=num_heads,
            num_time_steps=self.context_encoder.num_time_steps,
        )

    def forward(
            self,
            videos: torch.Tensor,
            context_idxs: list[torch.Tensor],
            target_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss on a batch.

        Args:
            videos: video of shape [B, C, T, H, W].
            context_idxs: one [B, K] tensor of visible token indices per mask type.
            target_idxs: one [B, M] tensor of token indices to predict per mask type.

        Returns:
            Scalar loss: equation 2 from paper.
        """

        with torch.no_grad():
          
            target_embeddings = self.target_encoder(videos)  # [B, N, D]
            target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.size(-1),))

        losses = []
        # masks may be different sizes so each gets its own context + predictor pass.
        for context_idx, target_idx in zip(context_idxs, target_idxs, strict=True):
            context_embeddings = self.context_encoder(videos, keep_idx=context_idx)  # [B, K, D]
            prediction = self.predictor(context_embeddings, context_idx, target_idx)  # [B, M, D]
            target = encoder.gather_tokens(target_embeddings, target_idx)  # [B, M, D]

            losses.append((prediction - target).abs().sum(dim=-1).mean())

        return torch.stack(losses).mean()

    def update_target_encoder(self, momentum: float) -> None:
        """Moves the target encoder via EMA to the context encoder."""

        ema.ema_update(self.target_encoder, self.context_encoder, momentum)