"""Vision Transformer (ViT) encoder. turns an image into a list of vectors, one vector per image patch.

B = batch size
N = number of patches in a full image (grid_size * grid_size)
K = number of patches actually kept (after masking)
D = embedding dimension
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def get_1d_pos_embed(embedding_dimension: int, positions: torch.Tensor) -> torch.Tensor:
    """Calculate sin, cos positional embeddings at two 'speeds' for a list of 1D positions."""

    half = embedding_dimension // 2
    # frequenxies 1, 1/10000^(1/half), ..., 1/10000^((half-1)/half)
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
    angles = positions.float()[:, None] * omega[None, :] # [M, half]

    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1) # [M, embedding_dimension]

def get_2d_pos_embed(embedding_dimension: int, grid_size: int) -> torch.Tensor:
    """
    Build positional embeddigs for a square grid of patches via the 1D sin, 
        cos embeddings of each patches row and column.
    """
    
    rows, cols, = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    row_embedding = get_1d_pos_embed(embedding_dimension // 2, rows.flatten())
    col_embedding = get_1d_pos_embed(embedding_dimension // 2, cols.flatten())

    return torch.cat([row_embedding, col_embedding], dim=1)

def get_3d_pos_embed(embedding_dimension: int, grid_size: int, num_time_steps: int) -> torch.Tensor:
    """Build positional embeddings for a video patches.

    Each axis gets an equal share of the dimensions.
    The share is rounded up to an even number, so the result is cut back to embedding_dimension.

    Args:
        embedding_dimension: numbers per patch.
        grid_size: patches along one side of a frame.
        num_time_steps: number of tubelets along time (num_frames // tubelet_size).

    Returns:
        Embeddings with shape [num_time_steps * grid_size * grid_size, embedding_dimension],
        ordered by time, then row.
    """

    times, rows, cols = torch.meshgrid(
        torch.arange(num_time_steps), torch.arange(grid_size), torch.arange(grid_size), indexing="ij")

    axis_dimension = math.ceil(embedding_dimension / 6) * 2
    time_embedding = get_1d_pos_embed(axis_dimension, times.flatten())
    row_embedding = get_1d_pos_embed(axis_dimension, rows.flatten())
    col_embedding = get_1d_pos_embed(axis_dimension, cols.flatten())

    embedding = torch.cat([time_embedding, row_embedding, col_embedding], dim=1)
    return embedding[:, :embedding_dimension]

def get_pos_embed(embedding_dimension: int, grid_size: int, num_time_steps: int = 1) -> torch.Tensor:
    """route to 2d or 3d position embeddings.

    Args:
        embedding_dimension: numbers per patch.
        grid_size: patches along one side of an image or frame.
        num_time_steps: tubelets along time. 1 means an image.

    Returns:
        Embeddings of shape [N, embedding_dimension].
    """

    if num_time_steps == 1:
        return get_2d_pos_embed(embedding_dimension, grid_size)

    return get_3d_pos_embed(embedding_dimension, grid_size, num_time_steps)

def gather_tokens (x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select a subset of tokens for each sample in a batch.
    
    Args:
        x: Tokens. (shape [B, N, D])
        idx: index of token to keep. shape [B,K]. Each row can pick different patches.

    Returns:
        tensor of shape [B, K, D] containing the selected tokens.
    """

    # [B, K, D] - gather neex idx to have same num of dims as x.
    idx = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
    return torch.gather(x, dim=1, index=idx) # [B, K, D]

class PatchEmbedding(nn.Module):
    """Samples and image into patches and embeds each patch into a vector."""

    def __init__(self, patch_size: int, input_channels: int, embedding_dimension: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch tokens of shape [B, N, D] for images of shape [B, C, H, W]."""

        x = self.projection(x) # [B, D, H/patch_size, W/patch_size]
        return x.flatten(2).transpose(1,2) # [B, N, D]

class PatchEmbedding3D(nn.Module):
    """Sample video into tubes then embeds each to a vector."""

    def __init__(self, patch_size: int, tubelet_size: int, input_channels: int, embedding_dimension: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels=input_channels,
            out_channels=embedding_dimension,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tube tokens of shape [B, N, D] for videos of shape [B, C, T, H, W]."""

        x = self.projection(x) # [B, D, T/tubelet_size, H/patch_size, W/patch_size]
        return x.flatten(2).transpose(1,2) # [B, N, D]

class Attention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, embedding_dimension: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.qkv = nn.Linear(embedding_dimension, embedding_dimension * 3) # queries, key and values
        self.project = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns atteneded tokens with same shape as input tokens – [B, N, D]."""

        batch, tokens, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dimension // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4) # [B, heads, K, D/heads]
        x = F.scaled_dot_product_attention(q,k,v) # softmax(q @ K^T / sqrt(d)) @ v
        x = x.transpose(1,2).reshape(batch, tokens, dimension) # [B, N, D]

        return self.project(x)

class Block(nn.Module):
    """transformer block with attention and MLP layers, both with residual connections."""

    def __init__(self, embedding_dimension: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.attention = Attention(embedding_dimension, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        hidden = int(embedding_dimension * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embedding_dimension, hidden), nn.GELU(), 
                                 nn.Linear(hidden, embedding_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns tokens with the same shape as x, [B, N, D]."""

        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

def init_weights(module: nn.Module) -> None:
    """Apply standard ViT initialisation to a module.
    
    Args:
        module: a nn.Module.
    """

    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class VisionTransformer(nn.Module):
    """Vision Transformer – turns an image of patches into vectors."""

    def __init__(
            self,
            image_size: int = 32,
            patch_size: int = 4,
            input_channels: int = 3,
            embedding_dimension: int = 192,
            depth: int = 6,
            num_heads: int = 3,
            num_frames: int = 1,
            tubelet_size: int = 2,
    ) -> None:
        
        super().__init__()
        self.grid_size = image_size // patch_size
        self.num_time_steps = num_frames // tubelet_size if num_frames > 1 else 1
        self.num_patches = self.num_time_steps * self.grid_size ** 2
        
        self.patch_embedding: nn.Module
        if num_frames > 1:
            self.patch_embedding = PatchEmbedding3D(patch_size, tubelet_size, input_channels, embedding_dimension)
        else:
            self.patch_embedding = PatchEmbedding(patch_size, input_channels, embedding_dimension)
        
        position_embedding = get_pos_embed(embedding_dimension, self.grid_size, self.num_time_steps)
        self.register_buffer("position_embedding", position_embedding.unsqueeze(0)) # [1, N, D]

        self.blocks = nn.ModuleList([Block(embedding_dimension, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embedding_dimension)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through ViT. Encodes images.
        
        Args:
            x: Images of shape [B, C, H, W].
            keep_idx: Optional patch indices to keep. if None, all N patches encoded.

        Returns:
            Patch embeddings of shape [B, K, D] (or [B, N, D] if keep_idx is None).
        """

        x = self.patch_embedding(x) + self.position_embedding # [B, N, D]

        if keep_idx is not None:
            x = gather_tokens(x, keep_idx) # [B, K, D]

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


