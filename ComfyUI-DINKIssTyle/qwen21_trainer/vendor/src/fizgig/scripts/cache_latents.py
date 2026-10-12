"""CLI entry point for caching VAE latents.

Usage:
    python src/fizgig/scripts/cache_latents.py --dataset_config config.toml --vae path/to/ae --model_version klein-base-9b
"""

import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# Ensure fizgig package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint, load_user_config
from fizgig.dataset.image_dataset import ItemInfo, save_latent_cache
from fizgig.training.metadata import ARCHITECTURE_KLEIN_9B_FULL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def preprocess_images(batch: List[ItemInfo]) -> tuple:
    """Convert image batch to normalized tensor for VAE encoding."""
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    # Handle control images
    controls = []
    for item in batch:
        if item.control_content is not None and len(item.control_content) > 0:
            controls.append([torch.from_numpy(cc[..., :3]) for cc in item.control_content])

    if controls:
        controls = [[c.permute(2, 0, 1) / 127.5 - 1.0 for c in cl] for cl in controls]
    else:
        controls = None

    return contents, controls


def encode_and_save_batch(ae, batch: List[ItemInfo]):
    """Encode images through VAE and save cached latents."""
    contents, controls = preprocess_images(batch)

    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W
        if controls is not None:
            control_latents = [[ae.encode(c.to(ae.device, dtype=ae.dtype).unsqueeze(0))[0] for c in cl] for cl in controls]
        else:
            control_latents = None

    for b, item in enumerate(batch):
        target_latent = latents[b]
        control_latent = control_latents[b] if control_latents is not None else None
        logger.info(f"Saving cache: {item.item_key} -> {item.latent_cache_path}, shape: {target_latent.shape}")
        save_latent_cache(item, target_latent, control_latent)


def encode_datasets(datasets, encode_fn, args):
    """Iterate through datasets, encode images, save caches."""
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_cache_paths = []

        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            # Ensure RGB (strip alpha)
            for item in batch:
                if isinstance(item.content, np.ndarray) and item.content.shape[-1] == 4:
                    item.content = item.content[..., :3]

            all_cache_paths.extend([item.latent_cache_path for item in batch])

            if args.skip_existing:
                # "Existing" must mean existing AT THIS RESOLUTION: the filename encodes the
                # original image size, so a cache from a different Target Megapixels looks
                # identical by name. Re-encode (overwrite) whenever the contained latent
                # doesn't match the current bucket — that is the wipe-on-resolution-change.
                from fizgig.dataset.image_dataset import ImageDataset as _ID
                # An optional per-architecture hook (args.needs_reencode(path) -> bool) can
                # veto the skip for a file that exists and matches — MiniMax uses it for a
                # clip cached without the still that --clip_still asks for.
                _needs = getattr(args, "needs_reencode", None)
                batch = [item for item in batch
                         if not os.path.exists(item.latent_cache_path)
                         or _ID.latent_cache_matches_reso(item.latent_cache_path,
                                                          item.bucket_size,
                                                          dataset.architecture) is not True
                         or (_needs is not None and _needs(item.latent_cache_path))]
                if not batch:
                    continue

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for j in range(0, len(batch), bs):
                encode_fn(batch[j : j + bs])

        # Remove stale cache files
        all_cache_paths = set(os.path.normpath(p) for p in all_cache_paths)
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keeping stale cache: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed stale cache: {cache_file}")


