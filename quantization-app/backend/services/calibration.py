"""
Calibration data loading and perplexity evaluation using wikitext-2.
Provides real calibration data for GPTQ and standard perplexity measurement.
"""
import torch
import numpy as np
import time
import os
from typing import List, Optional


CACHE_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "quant_cache")


def get_calibration_data(num_samples: int = 128, seq_length: int = 512, tokenizer=None) -> List[torch.Tensor]:
    """
    Load wikitext-2 calibration data. Downloads once, caches locally.
    Returns list of tokenized sequences (input_ids tensors).

    Args:
        num_samples: Number of calibration sequences to return
        seq_length: Length of each sequence in tokens
        tokenizer: HuggingFace tokenizer to use for encoding

    Returns:
        List of input_ids tensors, each of shape (1, seq_length)
    """
    from datasets import load_dataset

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load wikitext-2 train split
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
        cache_dir=CACHE_DIR
    )

    # Concatenate all text into one long string
    all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize the full text
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]

    # Split into chunks of seq_length
    total_tokens = tokens.shape[0]
    num_chunks = total_tokens // seq_length

    if num_chunks < num_samples:
        # If not enough sequential chunks, use overlapping windows
        stride = max(1, (total_tokens - seq_length) // num_samples)
        chunks = []
        for i in range(num_samples):
            start = i * stride
            if start + seq_length > total_tokens:
                break
            chunk = tokens[start:start + seq_length].unsqueeze(0)
            chunks.append(chunk)
    else:
        # Randomly sample from available chunks
        all_chunks = []
        for i in range(num_chunks):
            start = i * seq_length
            all_chunks.append(tokens[start:start + seq_length].unsqueeze(0))

        # Deterministic random selection
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_chunks), size=min(num_samples, len(all_chunks)), replace=False)
        chunks = [all_chunks[i] for i in sorted(indices)]

    return chunks


def evaluate_perplexity_wikitext(model, tokenizer, max_samples: int = 50, seq_length: int = 512) -> float:
    """
    Compute perplexity on wikitext-2 test split (industry standard metric).
    Uses sliding window approach.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        max_samples: Maximum number of chunks to evaluate (for speed)
        seq_length: Sequence length for each chunk

    Returns:
        Float perplexity value
    """
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load wikitext-2 test split
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
        cache_dir=CACHE_DIR
    )

    # Concatenate all text
    all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
    total_tokens = tokens.shape[0]

    # Split into chunks
    num_chunks = min(max_samples, total_tokens // seq_length)
    if num_chunks == 0:
        raise ValueError("Not enough tokens in test set for evaluation")

    # Compute perplexity using cross-entropy loss
    total_loss = 0.0
    total_count = 0
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * seq_length
            end = start + seq_length
            input_ids = tokens[start:end].unsqueeze(0).to(device)

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # loss is average cross-entropy over all positions
            total_loss += loss.item() * (seq_length - 1)  # -1 because first token has no prediction
            total_count += (seq_length - 1)

    avg_loss = total_loss / total_count
    perplexity = float(np.exp(avg_loss))

    return perplexity
