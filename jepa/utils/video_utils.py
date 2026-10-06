"""Video-based utility functios

Note:
    - we use a subset of the dataset used in the official paper
    - we use 64x64 frames with 8x8 patches rather than 224x224 frames with 
        16x16 patches as in the official implementation
    - we use a smaller ViT network
    - set warmup and schedule length as a fraction of the run not hardcoded in.
    - no weight decay on biases and layernorm weights.

"""

import csv
import multiprocessing
import pathlib
import tarfile
from collections.abc import Sequence

import av
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def decode_video(path: str, frame_stride: int, size: int) -> torch.Tensor | None:
    """Decode every frame_stride-th frame + resize & crop to size.

    Args:
        path: path to a video file.
        frame_stride: which n-th frames to keep.
        size: output height and width in pixels.

    Returns:
        frames of shape [F, size, size, 3] in uint8.
    """

    try:
        with av.open(path) as container:

            stream = container.streams.video[0]
            frames = []

            for i, frame in enumerate(container.decode(stream)):
                if i % frame_stride:
                    continue

                # resize the short side to `size`, then crop the centre square.
                scale = size / min(frame.width, frame.height)
                width, height = round(frame.width * scale), round(frame.height * scale)
                image = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
                top, left = (height - size) // 2, (width - size) // 2

                frames.append(image[top : top + size, left : left + size])

    except (av.FFmpegError, IndexError):
        return None

    if not frames:
        return None
    
    return torch.from_numpy(np.stack(frames))


def _decode_job(job: tuple[str, int, int]) -> torch.Tensor | None:
    """Unpacks a (path, frame_stride, size) for decoding."""
    return decode_video(*job)


def build_kinetics_cache(data_dir: str, frame_stride: int, size: int, min_frames: int) -> pathlib.Path:
    """Extract Kinetics-400 parts and decode every video into a cache.

    Expects data_dir/train.csv and data_dir/targz/*.tar.gz (from the CVDF mirror).
    skips if cache exists.

    Args:
        data_dir: folder with Kinetics-400.
        frame_stride: which n-th frames to keep.
        size: frame height and width in pixels.
        min_frames: minimum frames a video must have ot be kept.

    Returns:
        Path of the cache file.
    """

    root = pathlib.Path(data_dir)
    cache = root / f"cache_{size}px_stride{frame_stride}.pt"

    if cache.exists():
        return cache

    video_dir = root / "videos"
    if not video_dir.exists():
        video_dir.mkdir()

        for part in sorted((root / "targz").glob("*.tar.gz")):
            print(f"extracting {part.name}")
            with tarfile.open(part) as tar:
                tar.extractall(video_dir, filter="data")

    # video files are named <youtube_id>_<start:06d>_<end:06d>.mp4
    labels = {}
    with open(root / "train.csv") as file:
        for row in csv.DictReader(file):
            key = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
            labels[key] = row["label"]

    classes = sorted(set(labels.values()))

    paths = sorted(p for p in video_dir.rglob("*.mp4") if p.stem in labels)
    print(f"decoding {len(paths)} videos")

    with multiprocessing.Pool() as pool:
        decoded = pool.map(_decode_job, [(str(p), frame_stride, size) for p in paths], chunksize=8)

    kept = [(frames, classes.index(labels[p.stem])) for p, frames in zip(paths, decoded, strict=True)
            if frames is not None and len(frames) >= min_frames]
    
    print(f"kept {len(kept)} of {len(paths)} videos")

    torch.save({
        "frames": [frames for frames, _ in kept],
        "labels": torch.tensor([label for _, label in kept]),
        "classes": classes,
    }, cache)

    return cache


class VideoClips(Dataset):
    """num_frames long clips from a video cache.

    Each __getitem__ gets a random start frame meaning every epoch sees different clips.

    Augmentations are applied with the same random parameters to every frame of a clip.
    """

    def __init__(self, cache: pathlib.Path, num_frames: int, augmentations: Sequence[nn.Module] = ()) -> None:

        data = torch.load(cache)
        self.frames: list[torch.Tensor] = data["frames"]
        self.labels: torch.Tensor = data["labels"]
        self.classes: list[str] = data["classes"]
        self.num_frames = num_frames
        self.transform = v2.Compose([
            *augmentations,
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Returns (video [C, T, H, W], label) for one random clip of video `index`."""

        frames = self.frames[index]
        start = torch.randint(0, len(frames) - self.num_frames + 1, (1,)).item()
        clip = frames[start : start + self.num_frames].permute(0, 3, 1, 2)  # [T, C, H, W]

        clip = self.transform(clip)

        return clip.permute(1, 0, 2, 3), int(self.labels[index])  # [C, T, H, W]


def load_kinetics(
    data_dir: str,
    num_frames: int,
    frame_stride: int,
    size: int,
    augmentations: Sequence[nn.Module] = (),
) -> VideoClips:
    """Load the Kinetics-400 subset as random clips.

    If not already done, build the cache

    Args:
        data_dir: Kinetics-400 folder (train.csv + targz/).
        num_frames: frames per clip.
        frame_stride: which n-th frames to keep.
        size: frame height and width in pixels.
        augmentations: random transforms applied identically to every frame of a clip.

    Returns:
        A dataset of (video [3, num_frames, size, size], label) pairs.
    """

    cache = build_kinetics_cache(data_dir, frame_stride, size, min_frames=num_frames)

    return VideoClips(cache, num_frames, augmentations)
