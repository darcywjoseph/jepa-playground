"""Train V-JEPA"""

import argparse
import pathlib
from typing import Any

import torch
from torchvision.transforms import v2

from jepa import ema
from jepa.masking import tube
from jepa.utils import training_utils, video_utils
from variants.vjepa.model import VJEPA


def train(config: dict[str, Any]) -> VJEPA:
    """Train V-JEPA and save a checkpoint after each epoch.

    Args:
        config: config dict loaded from a yaml file.

    Returns:
        The trained model.
    """

    data_config, mask_config, optimiser_config, logs_config = (
        config["data"], config["mask"], config["optimization"], config["logging"])
    model_config = config["model"]

    device = training_utils.pick_device()
    model = VJEPA(**model_config).to(device)

    collator = tube.TubeMaskCollator(
        model.context_encoder.grid_size,
        model.context_encoder.num_time_steps,
        num_blocks=tuple(mask_config["num_blocks"]),
        spatial_scales=tuple(mask_config["spatial_scales"]),
        aspect_ratio=tuple(mask_config["aspect_ratio"]),
        min_keep=mask_config["min_keep"],
    )

    augmentations = [
        v2.RandomResizedCrop(
            model_config["image_size"],
            scale=tuple(data_config["crop_scale"]),
            ratio=tuple(data_config["crop_aspect_ratio"]),
        ),
        v2.RandomHorizontalFlip(),
    ]

    dataset = video_utils.load_kinetics(
        data_config["data_dir"],
        num_frames=model_config["num_frames"],
        frame_stride=data_config["frame_stride"],
        size=model_config["image_size"],
        augmentations=augmentations,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=data_config["num_workers"],
        persistent_workers=data_config["num_workers"] > 0,
    )

    # context encoder and predictor are trained by gradients; the target encoder by EMA.
    optimizer = torch.optim.AdamW(training_utils.sort_parameter_groups(model.context_encoder, model.predictor))

    total_steps = optimiser_config["epochs"] * len(loader)
    if optimiser_config["max_steps"] is not None:
        total_steps = min(total_steps, optimiser_config["max_steps"])

    warmup_steps = max(1, round(total_steps * optimiser_config["warmup_fraction"]))
    # schedules run as if training were schedule_scale times longer then stop.
    schedule_steps = round(total_steps * optimiser_config["schedule_scale"])

    checkpoint = pathlib.Path(logs_config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"device {device}, {len(dataset)} videos, {len(loader)} steps per epoch, "
          f"{total_steps} steps, {warmup_steps} warmup steps")

    step = 0
    for epoch in range(optimiser_config["epochs"]):

        for videos, _, context_idxs, target_idxs in loader:

            if step >= total_steps:
                break

            learning_rate = training_utils.warmup_cosine_schedule(
                step, schedule_steps, warmup_steps, optimiser_config["start_lr"],
                optimiser_config["lr"], optimiser_config["final_lr"],
            )
            weight_decay = training_utils.linear_schedule(
                step, schedule_steps, optimiser_config["weight_decay"], optimiser_config["final_weight_decay"],
            )

            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                if group["decay"]:
                    group["weight_decay"] = weight_decay

            # forward pass
            videos = videos.to(device)
            context_idxs = [context_idx.to(device) for context_idx in context_idxs]
            target_idxs = [target_idx.to(device) for target_idx in target_idxs]
            loss = model(videos, context_idxs, target_idxs)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ema update of target encoder
            model.update_target_encoder(ema.momentum_at(step, schedule_steps, *optimiser_config["ema"]))

            if step % logs_config["log_every"] == 0:
                # check for collapse.
                with torch.no_grad():
                    clip_embeddings = model.target_encoder(videos).mean(dim=1)  # [B, D]
                    emb_std = clip_embeddings.std(dim=0).mean().item()

                print(
                    f"epoch {epoch} step {step} loss {loss.item():.2f} lr {learning_rate:.2e} "
                    f"wd {weight_decay:.3f} emb_std {emb_std:.4f}"
                )

            step += 1

        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, checkpoint)

        if step >= total_steps:
            break

    return model


def main() -> None:
    """Reads the config and launches training."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a YAML config, e.g. variants/vjepa/config/kinetics400_VJEPA.yaml")

    config = training_utils.load_config(parser.parse_args().config)

    train(config)
    print(f"saved {config['logging']['checkpoint']}")


if __name__ == "__main__":
    main()