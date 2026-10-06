"""JEPA predictor. Predict target embeddings from context embeddings."""

import torch
from torch import nn

from jepa import encoder


class Predictor(nn.Module):
    """A transformer that predicts embeddings at masked positions.
    
    Input is the encoders output for for the context patches. 
    For every target position, a token representing a masked patch plus its positional embedding is provided.
    i.e. Ask "what is at position p?

    Can be used for I-JEPA (num_time_steps == 1) and V-JEPA (num_time_steps > 1).
    """

    position_embedding: torch.Tensor

    def __init__(
            self,
            grid_size: int,
            embedding_dimension: int = 192,
            predictor_dimension: int = 96,
            depth: int = 4,
            num_heads: int = 3,
            num_time_steps: int = 1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(embedding_dimension, predictor_dimension)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dimension))

        position_embedding = encoder.get_pos_embed(predictor_dimension, grid_size, num_time_steps)
        self.register_buffer("position_embedding", position_embedding.unsqueeze(0))  # [1, N, P]

        self.blocks = nn.ModuleList([encoder.Block(predictor_dimension, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dimension)
        self.output_projection = nn.Linear(predictor_dimension, embedding_dimension)

        self.apply(encoder.init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
            self,
            context_tokens: torch.Tensor,
            context_idx: torch.Tensor,
            target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict embeddings for a single target block.
        
        Args:
            context_tokens: encoder embedding of the context
            context_idx: patch index of context tokens
            target_idx: patch indeces to predict

        Returns:
            predicted embeddings at the target positions.
        """

        batch = context_tokens.size(0)
        position_embedding = self.position_embedding.expand(batch, -1, -1)

        # encoders adds positions but in encoder-space but we need it in predictor-space
        x = self.input_projection(context_tokens)
        x = x + encoder.gather_tokens(position_embedding, context_idx)

        num_targets = target_idx.size(1)
        queries = self.mask_token.expand(batch, num_targets, -1)
        queries = queries + encoder.gather_tokens(position_embedding, target_idx)

        x = torch.cat([x, queries], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        #filter to outputs at query positions (predictions). Removing context outputs.
        x = x[:, -num_targets:]
        return self.output_projection(x)