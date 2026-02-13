"""Tokenize FineWeb-Edu into .bin memmap files for training.

Adapted from nanoGPT's data/openwebtext/prepare.py with:
  - Parameterized tokenizer (--tokenizer CLI arg, HF model name)
  - uint32 dtype (vocab sizes > 65535)
  - FineWeb-Edu sample-10BT split
  - Deterministic train/val split (seed=2357)
  - Saves token_bytes.pt for BPB evaluation
  - Saves meta.pkl with vocab_size, eot_token_id, tokenizer_name

Usage:
  uv run python data/fineweb_edu/prepare.py --tokenizer allenai/Olmo-3-1025-7B
  uv run python data/fineweb_edu/prepare.py --tokenizer Qwen/Qwen3-8B --out_dir data/fineweb_edu_qwen
"""

import argparse
import os
import pickle

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

num_proc = 12


def compute_token_bytes(tokenizer):
    """Build a (vocab_size,) int tensor: token_id -> UTF-8 byte length.

    Special tokens get 0 bytes so they are excluded from BPB calculation.
    """
    vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "get_added_vocab"):
        vocab_size = len(tokenizer)  # includes added tokens

    token_bytes = torch.zeros(vocab_size, dtype=torch.int32)
    special_ids = set(tokenizer.all_special_ids)

    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes[token_id] = 0
            continue
        try:
            text = tokenizer.decode([token_id])
            token_bytes[token_id] = len(text.encode("utf-8"))
        except Exception:
            token_bytes[token_id] = 0

    return token_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace model name to load tokenizer from",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as this script)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eot_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}, EOT token ID: {eot_token_id}")

    # Load FineWeb-Edu sample-10BT
    print("Loading FineWeb-Edu sample-10BT...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        num_proc=num_proc,
    )

    # Deterministic train/val split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    print(f"Train documents: {len(split_dataset['train']):,}")
    print(f"Val documents: {len(split_dataset['val']):,}")

    # Tokenize
    def process(example):
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        ids.append(eot_token_id)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # Write to memmap .bin files (uint32)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(out_dir, f"{split}.bin")
        dtype = np.uint32  # vocab sizes > 65535
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{split}.bin: {arr_len:,} tokens ({arr_len * 4 / 1e9:.2f} GB)")

    # Save token_bytes.pt
    print("Computing token_bytes...")
    token_bytes = compute_token_bytes(tokenizer)
    torch.save(token_bytes, os.path.join(out_dir, "token_bytes.pt"))
    print(f"Saved token_bytes.pt (shape: {token_bytes.shape})")

    # Save meta.pkl
    meta = {
        "vocab_size": vocab_size,
        "eot_token_id": eot_token_id,
        "tokenizer_name": args.tokenizer,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl: {meta}")


if __name__ == "__main__":
    main()
