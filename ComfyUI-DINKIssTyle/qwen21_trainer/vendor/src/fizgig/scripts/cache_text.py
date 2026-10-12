"""CLI entry point for caching text encoder outputs.

Usage:
    python src/fizgig/scripts/cache_text.py --dataset_config config.toml --text_encoder path/to/qwen3 --model_version klein-base-9b
"""

import argparse
import logging
import os
import sys
from typing import List

import torch
from tqdm import tqdm

# Ensure fizgig package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint, load_user_config
from fizgig.dataset.image_dataset import ItemInfo, save_text_encoder_output_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_embedder, batch: List[ItemInfo], device: torch.device):
    """Encode captions and save cached text embeddings."""
    prompts = [item.caption for item in batch]
    autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype

    with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
        ctx_vec = text_embedder(prompts)
        ctx_vec = ctx_vec.cpu()  # (B, 512, 12288)

    for item, embed in zip(batch, ctx_vec):
        save_text_encoder_output_cache(item, embed)


def prepare_cache_files_and_paths(datasets):
    """Gather existing cache files and all expected cache paths."""
    all_files = []
    all_paths = []
    for dataset in datasets:
        files = set(os.path.normpath(f) for f in dataset.get_all_text_encoder_output_cache_files())
        all_files.append(files)
        all_paths.append(set())
    return all_files, all_paths


def process_batches(args, datasets, all_files, all_paths, encode_fn, index_offset=0):
    """Process text encoder batches with skip-existing and batching support.

    index_offset only affects the log line: the reference pass calls this once per dataset with
    a ONE-element list, so without it every dataset reports itself as "[0]"."""
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i + index_offset}]")
        batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)

        bs_want = args.batch_size if args.batch_size is not None else 16
        for batch in tqdm(_regroup(batches, bs_want)):
            all_paths[i].update(os.path.normpath(item.text_encoder_output_cache_path) for item in batch)

            if args.skip_existing:
                batch = [item for item in batch if os.path.normpath(item.text_encoder_output_cache_path) not in all_files[i]]
                if not batch:
                    continue

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for j in range(0, len(batch), bs):
                encode_fn(batch[j : j + bs])


def _regroup(batches, size):
    """Re-chunk the dataset's batches to `size`.

    The dataset yields groups of its TRAINING batch_size — 1 for MiniMax — which has nothing to
    do with how many captions a text-encoder forward should take. Without this the encoder is
    called once per caption no matter what --batch_size says."""
    buf = []
    for b in batches:
        buf.extend(b)
        while len(buf) >= size:
            yield buf[:size]
            buf = buf[size:]
    if buf:
        yield buf


def post_process(datasets, all_files, all_paths, keep_cache):
    """Remove stale cache files not in the dataset."""
    for i, dataset in enumerate(datasets):
        for cache_file in all_files[i]:
            if cache_file not in all_paths[i]:
                if keep_cache:
                    logger.info(f"Keeping stale cache: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed stale cache: {cache_file}")


