"""Train I-JEPA on CIFAR-10.

Note:
    - Original I-JEPA trains on ImageNet-1K (1.28M images) not CIFAR-10 (50k Images)
    - warmup 5% of training run rather than fixed at 15 epochs. 
    - no weight decay on biases or LayerNorm weights - 
        this is done in the official code base but not referenced in the paper.
"""

import argparse
import pathlib
from typing import Any

import torch
from torchvision import transforms

from jepa import ema
from jepa.masking import block
from jepa.utils import training_utils
from variants.ijepa.model import IJEPA


def train(config: dict[str, Any]) -> IJEPA:
    """train I-JEPA + save checkpoints at each epoch.
    
    Args:
        config: config dict loaded from yaml file.

    Returns:
        Final trained model.
    """

    data_config, mask_config, optimiser_config, logs_config = (config["data"], 
                    config["mask"], config["optimization"], config["logging"])

    device = training_utils.pick_device()
    model = IJEPA(**config["model"]).to(device)

    collator = block.MultiBlockMaskCollator(
        model.context_encoder.grid_size,
        num_target_blocks=mask_config["num_target_blocks"],
        target_scale=tuple(mask_config["target_scale"]),
        target_aspect_ratio=tuple(mask_config["target_aspect_ratio"]),
        context_scale=tuple(mask_config["context_scale"]),
        min_keep=mask_config["min_keep"],
    )   
    crop = transforms.RandomResizedCrop(config["model"]["image_size"], scale=tuple(data_config["crop_scale"]))
    loader = torch.utils.data.DataLoader(
        training_utils.load_cifar(data_config["data_dir"], train=True, augmentations=[crop], 
                                  fake=data_config["fake_data"]),
        batch_size=data_config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=data_config["num_workers"],
        persistent_workers=data_config["num_workers"] > 0,
        )

    # context_encoder and predictor trained via gradients. target encoder isnt.
    optimizer = torch.optim.AdamW(training_utils.sort_parameter_groups(model.context_encoder, model.predictor))

    total_steps = optimiser_config["epochs"] * len(loader)

    if optimiser_config["max_steps"] is not None:
        total_steps = min(total_steps, optimiser_config["max_steps"])

    warmup_steps = max(1, round(total_steps * optimiser_config["warmup_fraction"]))

    checkpoint = pathlib.Path(logs_config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"device {device}, {len(loader)} steps per epoch, {total_steps} steps, {warmup_steps} warmup steps")

    step = 0
    for epoch in range(optimiser_config["epochs"]):

        for images, _, context_idx, target_idxs in loader:

            if step >= total_steps:
                break

            learning_rate = training_utils.warmup_cosine_schedule(
                step, total_steps, warmup_steps, optimiser_config["start_lr"], 
                optimiser_config["lr"], optimiser_config["final_lr"]
            )

            weight_decay = training_utils.linear_schedule(
                step, total_steps, optimiser_config["weight_decay"], optimiser_config["final_weight_decay"]
            )

            for group in optimizer.param_groups:
            
                group["lr"] = learning_rate
            
                if group["decay"]:
                    group["weight_decay"] = weight_decay

            # forward pass
            images = images.to(device)
            context_idx = context_idx.to(device)
            target_idxs = [target_idx.to(device) for target_idx in target_idxs]
            loss = model(images, context_idx, target_idxs)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ema update of target encoder
            model.update_target_encoder(ema.momentum_at(step, total_steps, *optimiser_config["ema"]))

            if step % logs_config["log_every"] == 0:
                # check for training collapse
                with torch.no_grad():
                    image_embeddings = model.target_encoder(images).mean(dim=1)  # [B, D]
                    emb_std = image_embeddings.std(dim=0).mean().item()

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
    """Reads config and launch training."""
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a YAML config, e.g. variants/ijepa/config/cifar10_IJEPA.yaml")
    
    config = training_utils.load_config(parser.parse_args().config)
    
    train(config)
    print(f"saved {config['logging']['checkpoint']}")


if __name__ == "__main__":
    main()