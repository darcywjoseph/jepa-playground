"""3D masking ('tubes') for V-JEPA."""

import math

import torch
from torch.utils.data import default_collate


class TubeMaskCollator:
    """batch video, label paris with 3D masks."""

    def __init__(
        self,
        grid_size: int,
        num_time_steps: int,
        num_blocks: tuple[int, ...] = (8, 2),
        spatial_scales: tuple[float, ...] = (0.15, 0.7),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 4,
    ) -> None:

        if len(num_blocks) != len(spatial_scales):
            raise ValueError ("one scale per mask type required.")

        self.grid_size = grid_size
        self.num_time_steps = num_time_steps
        self.num_blocks = num_blocks
        self.spatial_scales = spatial_scales
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep

    def _sample_uniform(self, low: float, high: float) -> float:
        """sample random float from the range [low, high]."""
        return low + torch.rand(1).item() * (high - low)

    def _sample_block_size(self, scale: float) -> tuple[int, int]:
        """sample a (height,width) in patches covering 'scale of a frame."""

        area = scale * self.grid_size * self.grid_size
        ratio = self._sample_uniform(*self.aspect_ratio)

        height = round(math.sqrt(area * ratio))
        width = round(math.sqrt(area / ratio))

        height = min(max(height, 1), self.grid_size)
        width = min(max(width, 1), self.grid_size)

        return height, width

    def _sample_tube(self, height: int, width: int) -> torch.Tensor:
        """sample [time, grid, grid] dimensioned boolean mask."""

        top = torch.randint(0, self.grid_size - height + 1, (1,)).item()
        left = torch.randint(0, self.grid_size - width + 1, (1,)).item()

        mask = torch.zeros(self.num_time_steps, self.grid_size, self.grid_size, dtype=torch.bool)
        mask[:, top : top + height, left : left + width] = True
        
        return mask

    def _sample_mask(self, batch_size: int, num_blocks: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one mask type on a batch.
        
        Args:
            batch_size: number of videos.
            num_blocks: how many tubes merged into the target.
            scale: fraction of a frame each tube will covers.

        Returns:
            (context_idx [B, K], target_idx [B, M]) token indices.
        """

        height, width = self._sample_block_size(scale)

        context_masks, target_masks = [], []

        for _ in range(batch_size):

            # resample if the tubes accidently cover whole video
            for _ in range(20):
                target = torch.stack([self._sample_tube(height, width) for _ in range(num_blocks)]).any(dim=0)
                if (~target).sum() >= self.min_keep:
                    break
            else:
                raise RuntimeError(f"tubes of size {height}x{width} leave fewer than {self.min_keep} context tokens")

            context_masks.append(~target.flatten())
            target_masks.append(target.flatten())

        # keep the smallest mask size in the batch so they stack.
        min_context = min(int(mask.sum()) for mask in context_masks)
        min_target = min(int(mask.sum()) for mask in target_masks)

        context_idx = torch.stack([mask.nonzero().squeeze(-1)[:min_context] for mask in context_masks])
        target_idx = torch.stack([mask.nonzero().squeeze(-1)[:min_target] for mask in target_masks])

        return context_idx, target_idx

    def __call__(self, batch: list) -> tuple:
        """Collate a batch of videos + labels, attach a mask per mask type.

        Args:
            batch: list of (video, label) pairs, each video [C, T, H, W].

        Returns:
            (videos, labels, context_idxs, target_idxs).
        """

        videos, labels = default_collate(batch)

        context_idxs, target_idxs = [], []

        for num_blocks, scale in zip(self.num_blocks, self.spatial_scales, strict=True):
            context_idx, target_idx = self._sample_mask(len(videos), num_blocks, scale)
            context_idxs.append(context_idx)
            target_idxs.append(target_idx)

        return videos, labels, context_idxs, target_idxs