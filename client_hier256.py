#!/usr/bin/env python3
"""Experimental full-PMF RWKV codec using a 256 x 256 hierarchical alphabet.

The first categorical symbol is the high byte of the token id.  The second is the low
byte conditioned on the decoded high byte.  Their ideal joint information is exactly the
full 65,536-way model surprisal, while each CDF has only 256 symbols.  This avoids UID112's
top-K escape approximation and removes the expensive GPU top-K plus rank transfer.
"""

from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

import client as base


SCHEME = 0xFFFE
RADIX = 256
# DockerRunner does not forward GLYPH_B. Keep the final, timing-validated lane
# geometry in this submitted entrypoint instead of inheriting the base client's
# UID112 B=256 default or an ambient diagnostic environment variable.
B_FIXED = 96
# The pinned UID114 base image's /app/client.py has no TEMPERATURE attribute, and
# DockerRunner does not forward GLYPH_TEMPERATURE.  The promoted benchmark used 1.0, so
# bind that exact coding parameter in this entrypoint just like the lane geometry above.
TEMPERATURE = 1.0


def _chunks(token_count: int, batch: int) -> list[int]:
    quotient, remainder = divmod(token_count, batch)
    return [quotient + 1 if lane < remainder else quotient for lane in range(batch)]


def _probabilities(inp, state):
    with torch.no_grad():
        probabilities = torch.softmax(
            base.step(inp, *state).float() / TEMPERATURE,
            dim=-1,
        )
        blocks = probabilities.view(inp.shape[0], RADIX, RADIX).sum(dim=-1)
    # Constriction accepts float32, and its fixed-point conversion defines the exact coding
    # semantics.  Avoid the original codec's unnecessary float64 expansion and transfer.
    return probabilities, blocks.cpu().numpy()


def _within_block(probabilities, rows: np.ndarray, high: np.ndarray) -> np.ndarray:
    row_index = torch.as_tensor(rows, dtype=torch.long, device=base.DEV)
    high_index = torch.as_tensor(high, dtype=torch.long, device=base.DEV)
    reshaped = probabilities.view(probabilities.shape[0], RADIX, RADIX)
    selected = reshaped[row_index, high_index]
    return selected.cpu().numpy()


def compress(data: bytes) -> bytes:
    tokens = base.tok.encode(data)
    token_count = len(tokens)
    batch = min(B_FIXED, max(1, token_count))
    if token_count == 0:
        # Constriction's range decoder requires an actual coded word stream.  Keep the
        # canonical empty representation header-only and bypass both CUDA state allocation
        # and range-coder construction on the matching decode path.
        return struct.pack("<QHH", 0, batch, SCHEME)
    lengths = _chunks(token_count, batch)
    starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    longest = max(lengths)
    matrix = np.zeros((batch, longest), dtype=np.int64)
    for lane, (start, length) in enumerate(zip(starts, lengths)):
        matrix[lane, :length] = tokens[int(start) : int(start) + length]

    state = base.new_state(batch)
    encoder = base.constriction.stream.queue.RangeEncoder()
    previous = torch.zeros(batch, dtype=torch.long, device=base.DEV)
    length_array = np.asarray(lengths)
    for position in range(longest):
        probabilities, block_probabilities = _probabilities(previous, state)
        active = np.flatnonzero(position < length_array)
        truth = matrix[active, position]
        high = (truth >> 8).astype(np.int32)
        low = (truth & 255).astype(np.int32)
        encoder.encode(high, base.fam, block_probabilities[active])
        within = _within_block(probabilities, active, high)
        encoder.encode(low, base.fam, within)
        previous = torch.as_tensor(matrix[:, position], device=base.DEV)

    compressed = encoder.get_compressed()
    return struct.pack("<QHH", token_count, batch, SCHEME) + compressed.tobytes()


def decompress(blob: bytes) -> bytes:
    token_count, batch, scheme = struct.unpack("<QHH", blob[:12])
    if scheme != SCHEME:
        raise ValueError(f"not a hierarchical-256 blob: scheme={scheme:#x}")
    if token_count == 0:
        if batch != 1:
            raise ValueError(f"empty hierarchical-256 blob has invalid batch: {batch}")
        if len(blob) != 12:
            raise ValueError("empty hierarchical-256 blob has an unexpected range-coded body")
        return base.tok.decode([])
    compressed = np.frombuffer(blob[12:], dtype=np.uint32)
    lengths = _chunks(token_count, batch)
    longest = max(lengths)
    length_array = np.asarray(lengths)
    state = base.new_state(batch)
    decoder = base.constriction.stream.queue.RangeDecoder(compressed)
    recovered = np.zeros((batch, longest), dtype=np.int64)
    previous = torch.zeros(batch, dtype=torch.long, device=base.DEV)
    for position in range(longest):
        probabilities, block_probabilities = _probabilities(previous, state)
        active = np.flatnonzero(position < length_array)
        high = np.asarray(
            decoder.decode(base.fam, block_probabilities[active]),
            dtype=np.int64,
        )
        within = _within_block(probabilities, active, high)
        low = np.asarray(decoder.decode(base.fam, within), dtype=np.int64)
        recovered[active, position] = (high << 8) | low
        previous = torch.as_tensor(recovered[:, position], device=base.DEV)

    tokens: list[int] = []
    for lane, length in enumerate(lengths):
        tokens.extend(recovered[lane, :length].tolist())
    return base.tok.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compress", "decompress"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, "rb") as source:
        payload = source.read()
    output = compress(payload) if args.mode == "compress" else decompress(payload)
    with open(args.output, "wb") as destination:
        destination.write(output)


if __name__ == "__main__":
    main()
