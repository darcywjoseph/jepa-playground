"""I-JEPA (Image-JEPA): context encoder, EMA target encoder + predictor."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from jepa import ema, encoder, predictor


class IJEPA(nn.Module):
    """I-JEPA comprised of cotext encoder, target encoder and predictor."""

    def __init__(
            self,
            image_size: int = 32,
            patch_size: int = 4,
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
        )

        #target encoder starts as an exact copy of the context encoder.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        self.predictor = predictor.Predictor(
            self.context_encoder.grid_size,
            embedding_dimension=embedding_dimension,
            predictor_dimension=predictor_dimension,
            depth=predictor_depth,
            num_heads=num_heads,
        )

    def forward(
            self,
            images: torch.Tensor,
            context_idx: torch.Tensor,
            target_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for a batch."""

        with torch.no_grad():

            # target embeddings – target encoder is given the whole image
            target_embeddings = self.target_encoder(images) # [B, N, D]
            #each patch's embeddings is rescaled to mean 0, spread 1 (no learned scale nor shift)
            target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.size(-1),))

        # context encoder sees only the visible patches
        context_embeddings = self.context_encoder(images, keep_idx=context_idx) # [B, no.patches, D]

        # predictor predicts every target block in a single predictor call. (stacked them along B dim)
        # parallelised version of applying predictor every time
        num_blocks = len(target_idxs)
        all_target_idx = torch.cat(target_idxs, dim=0)
        prediction = self.predictor(
            context_embeddings.repeat(num_blocks, 1, 1),
            context_idx.repeat(num_blocks, 1),
            all_target_idx
        )
        target = torch.cat([encoder.gather_tokens(target_embeddings, idx) for idx in target_idxs], dim=0)

        return (prediction - target).pow(2).sum(dim=(1,2)).mean()

    def update_target_encoder(self, momentum: float) -> None:
        """Moves the target encoder via EMA to the context encoder."""

        ema.ema_update(self.target_encoder, self.context_encoder, momentum)


