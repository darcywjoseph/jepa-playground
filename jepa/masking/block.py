"""Multi-block masking for I-JEPA.

For each image, we pick:
    - target blocks which the model must predict
    - a context block with targets cut out, this is provided to the model as input.
"""

import math

import torch
from torch.utils.data import default_collate


class MultiBlockMaskCollator:
    """Batches image and label pairs with multi-block masks.
    
    Passed to dataloader to collate a batch of images into a single batch with multi-block masks. 
    Each batch is [images, labels, context_idx, target_idxs].
    """

    def __init__(
        self,
        grid_size: int,
        num_target_blocks: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
        context_scale: tuple[float, float] = (0.85, 1.0),
        min_keep: int = 4,
    ) -> None:

        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_targets = num_target_blocks
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
        self.min_keep = min_keep    

    def _sample_uniform(self, low: float, high: float) -> float:
        """Get a random float drawn uniformly from given range [low, high]."""
        return low + torch.rand(1).item() * (high - low)

    def _sample_target_block_size(
            self, scale: tuple[float, float], aspect_ratio: tuple[float, float]) -> tuple[int, int]:
        """sample a (height, width) in patches covering approx 'scale' of image."""

        area = self._sample_uniform(*scale) * self.num_patches
        ratio = self._sample_uniform(*aspect_ratio)

        height = round(math.sqrt(area * ratio))
        width = round(math.sqrt(area/ratio))

        height = min(max(height,1), self.grid_size)
        width = min(max(width,1), self.grid_size)

        return height, width

    def _sample_target_block(self, height: int, width: int) -> torch.Tensor:
        """sample a [grid, grid] boolean mask with one block at a random positions."""

        top = torch.randint(0, self.grid_size - height + 1, (1,)).item()
        left = torch.randint(0, self.grid_size - width + 1, (1,)).item()

        mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
        mask[top : top + height, left : left + width] = True
        return mask

    def __call__(self, batch: list) -> tuple:
        """Collate a batch of images and labels and attach masks.
        
        Args:
            batch: list of (image, label) pairs.
        
        Returns:
        (images, labels, context_idx, target_idxs).
        """

        images, labels = default_collate(batch)

        # block sizes are shared across the whole batch, 
        #   but positions are samples independently for each image, label pair.

        target_height, target_width = self._sample_target_block_size(self.target_scale, self.target_aspect_ratio)
        context_height, context_width = self._sample_target_block_size(self.context_scale, (1.0,1.0))

        context_masks = []
        target_masks: list[list[torch.Tensor]] = [[] for _ in range(self.num_targets)]

        for _ in range(len(images)):
            targets = [self._sample_target_block(target_height, target_width) 
                       for _ in range(self.num_targets)]
            covered = torch.stack(targets).any(dim=0)

            # remove target patches from the context so model cannot see them.
            # 20 is from the I-JEPA implementation, 
            # used to resample up to 20 times if the context block 
            #   is too small (less than min_keep patches).            
            for _ in range(20):
                context = self._sample_target_block(context_height, context_width) & ~covered
                if context.sum() >= self.min_keep:
                    break
            else:
                # Give up on random sampling and jus use everything that is not convered by targets.
                context = ~covered

            context_masks.append(context.flatten())
            for i, target in enumerate(targets):
                target_masks[i].append(target.flatten())

        # Context differ in size per image. Ensure they are kept at the smallest size in the batch.
        min_context_size = min(mask.sum().item() for mask in context_masks)
        context_idx = torch.stack([mask.nonzero().squeeze(-1)[:min_context_size] 
                                   for mask in context_masks])
        target_idxs = [
            torch.stack([m.nonzero().squeeze(-1) for m in masks]) for masks in target_masks
            ]

        return images, labels, context_idx, target_idxs

            