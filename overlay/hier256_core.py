"""CPU-only format and probability primitives for the UID95 hier256 candidate.

This module deliberately has no torch or constriction dependency.  The production
adapter in :mod:`coder_hier256` and the CPU reference tests therefore share the exact
same header validation and zero-probability policy.

Inner blob layout (the daemon retains the existing outer method byte ``0x10``)::

    <QHH token_count, lanes, scheme=0xD256>
    <II  range_word_count, crc32(range_words)>
    <range_word_count little-endian uint32 words>

The canonical empty stream is the 12-byte header ``(0, 1, 0xD256)`` with no body.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import struct
import zlib
from typing import Collection, Iterable

import numpy as np


RADIX = 256
VOCAB_SIZE = RADIX * RADIX
SCHEME_HIER256_D256 = 0xD256
HEADER = struct.Struct("<QHH")
BODY_META = struct.Struct("<II")

# The validator mailbox accepts at most 64 MiB input.  A byte-lossless tokenizer
# cannot emit more tokens than input bytes, so larger token counts are malformed.
MAX_TOKEN_COUNT = 64 * 2**20
MAX_RANGE_BODY_BYTES = 128 * 2**20

# Predeclared deterministic zero policy.  Every 256-way categorical is first
# normalized in float64, then receives one 2^-24 pseudo-count per symbol and is
# normalized again.  An all-zero selected low-byte block becomes uniform.  NaN,
# infinity, and negative mass always fail closed.
ZERO_FLOOR = float(2.0**-24)

# Two fixed uniform-coded footer symbols. They are stream framing, not RWKV
# vocabulary tokens, and therefore never trigger a model step.
EOS_SYMBOLS = np.asarray([0xD2, 0x56], dtype=np.int32)


def runtime_batch_default(directory: str, fallback: int = RADIX) -> int:
    """Resolve the activation-pinned batch when the launcher sets no override.

    Screen runs normally set ``GLYPH_B_FIXED`` explicitly.  An activated fork also
    carries ``HIER256_ENVIRONMENT.txt`` so its measured lane count must remain the
    default after packaging rather than silently reverting to B256.
    """
    path = os.path.join(directory, "HIER256_ENVIRONMENT.txt")
    if not os.path.exists(path):
        value = int(fallback)
    else:
        fields: dict[str, str] = {}
        try:
            with open(path, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    key, separator, raw_value = line.partition("=")
                    if separator != "=" or key in fields:
                        raise CodecFormatError(
                            f"malformed or duplicate activation environment line: {line!r}"
                        )
                    fields[key] = raw_value
        except OSError as exc:
            raise CodecFormatError(
                f"cannot read activation environment {path}: {exc}"
            ) from exc
        if set(fields) != {"GLYPH_B_FIXED", "GLYPH_HIER256_ALLOWED_B"}:
            raise CodecFormatError(
                "activation environment must contain only GLYPH_B_FIXED and "
                "GLYPH_HIER256_ALLOWED_B"
            )
        if fields["GLYPH_HIER256_ALLOWED_B"] != fields["GLYPH_B_FIXED"]:
            raise CodecFormatError(
                "activation environment fixed and allowed batch values differ"
            )
        try:
            value = int(fields["GLYPH_B_FIXED"])
        except ValueError as exc:
            raise CodecFormatError("activation batch is not an integer") from exc
    if value < 1 or value > RADIX:
        raise CodecFormatError(f"activation batch out of range: {value}")
    return value


class CodecFormatError(ValueError):
    """A blob, token, or categorical surface violates the codec contract."""


class CodecProbabilityError(ValueError):
    """A model probability surface is non-finite, negative, or malformed."""


@dataclass(frozen=True)
class ParsedBlob:
    token_count: int
    lanes: int
    words: np.ndarray


def chunks(token_count: int, lanes: int) -> list[int]:
    """Return the canonical contiguous lane partition."""
    validate_header_fields(token_count, lanes, SCHEME_HIER256_D256)
    if token_count == 0:
        return [0]
    quotient, remainder = divmod(token_count, lanes)
    return [quotient + 1 if lane < remainder else quotient for lane in range(lanes)]


def validate_header_fields(
    token_count: int,
    lanes: int,
    scheme: int,
    allowed_lanes: Collection[int] | None = None,
) -> None:
    """Validate all non-body header invariants without allocating from them."""
    if not isinstance(token_count, (int, np.integer)):
        raise CodecFormatError("token_count is not an integer")
    if not isinstance(lanes, (int, np.integer)):
        raise CodecFormatError("lanes is not an integer")
    token_count = int(token_count)
    lanes = int(lanes)
    if scheme != SCHEME_HIER256_D256:
        raise CodecFormatError(
            f"wrong hier256 scheme 0x{scheme:04x}; expected 0x{SCHEME_HIER256_D256:04x}"
        )
    if token_count < 0 or token_count > MAX_TOKEN_COUNT:
        raise CodecFormatError(f"token_count out of range: {token_count}")
    if token_count == 0:
        if lanes != 1:
            raise CodecFormatError("canonical empty stream requires lanes=1")
        return
    if lanes < 1 or lanes > RADIX:
        raise CodecFormatError(f"lanes out of range: {lanes}")
    if allowed_lanes is not None and lanes not in {int(x) for x in allowed_lanes}:
        raise CodecFormatError(
            f"lanes={lanes} is not in the explicitly captured set {sorted(allowed_lanes)}"
        )


def validate_tokens(tokens: Iterable[int]) -> np.ndarray:
    """Return a contiguous int64 token vector, rejecting non-integral/out-of-vocab IDs."""
    try:
        raw = list(tokens)
    except TypeError as exc:
        raise CodecFormatError("tokens are not iterable") from exc
    if len(raw) > MAX_TOKEN_COUNT:
        raise CodecFormatError(f"too many tokens: {len(raw)}")
    for position, token in enumerate(raw):
        if not isinstance(token, (int, np.integer)):
            raise CodecFormatError(f"token[{position}] is not an integer: {token!r}")
        if int(token) < 0 or int(token) >= VOCAB_SIZE:
            raise CodecFormatError(f"token[{position}] out of range: {token}")
    return np.ascontiguousarray(raw, dtype=np.int64)


def split_tokens(tokens: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    values = validate_tokens(tokens)
    return (
        np.ascontiguousarray(values >> 8, dtype=np.int32),
        np.ascontiguousarray(values & 0xFF, dtype=np.int32),
    )


def join_tokens(high: Iterable[int], low: Iterable[int]) -> np.ndarray:
    high_values = np.asarray(list(high))
    low_values = np.asarray(list(low))
    if (
        high_values.ndim != 1
        or low_values.ndim != 1
        or high_values.shape != low_values.shape
    ):
        raise CodecFormatError(
            "high/low token arrays must be matching one-dimensional arrays"
        )
    for name, values in (("high", high_values), ("low", low_values)):
        if not np.issubdtype(values.dtype, np.integer):
            raise CodecFormatError(f"{name} token components are not integral")
        if np.any(values < 0) or np.any(values >= RADIX):
            raise CodecFormatError(f"{name} token component out of [0,255]")
    return np.ascontiguousarray(
        (high_values.astype(np.int64) << 8) | low_values.astype(np.int64),
        dtype=np.int64,
    )


def canonicalize_categorical(rows: np.ndarray) -> np.ndarray:
    """Normalize 256-way rows with the frozen deterministic zero policy.

    Individual zero entries receive ``ZERO_FLOOR``.  A completely zero row is
    interpreted as underflow and becomes uniform.  Invalid numeric values fail.
    The float64, C-contiguous result is the sole surface passed to constriction.
    """
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != RADIX:
        raise CodecProbabilityError(
            f"categorical surface must have shape [N,{RADIX}], got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise CodecProbabilityError("categorical surface contains NaN or infinity")
    if np.any(values < 0.0):
        raise CodecProbabilityError("categorical surface contains negative mass")

    output = values.copy()
    totals = output.sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(totals)):
        raise CodecProbabilityError("categorical row sum overflowed")
    zero_rows = totals == 0.0
    if np.any(zero_rows):
        output[zero_rows, :] = 1.0
        totals[zero_rows] = float(RADIX)
    output /= totals[:, None]
    output += ZERO_FLOOR
    output /= output.sum(axis=1, dtype=np.float64)[:, None]
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise CodecProbabilityError(
            "zero-policy construction did not produce positive PMFs"
        )
    return np.ascontiguousarray(output, dtype=np.float64)


def eos_categorical() -> np.ndarray:
    """Return the frozen two-row uniform PMF for the D2/56 range footer."""
    return canonicalize_categorical(
        np.ones((len(EOS_SYMBOLS), RADIX), dtype=np.float64)
    )


def high_byte_categorical(full_probabilities: np.ndarray) -> np.ndarray:
    probabilities = _validated_full_probabilities(full_probabilities)
    blocks = probabilities.reshape(probabilities.shape[0], RADIX, RADIX).sum(
        axis=2, dtype=np.float64
    )
    return canonicalize_categorical(blocks)


def conditional_low_categorical(
    full_probabilities: np.ndarray,
    high: Iterable[int],
    rows: Iterable[int] | None = None,
) -> np.ndarray:
    probabilities = _validated_full_probabilities(full_probabilities)
    high_values = np.asarray(list(high))
    if high_values.ndim != 1 or not np.issubdtype(high_values.dtype, np.integer):
        raise CodecFormatError(
            "high-byte selectors must be a one-dimensional integer array"
        )
    if np.any(high_values < 0) or np.any(high_values >= RADIX):
        raise CodecFormatError("high-byte selector out of [0,255]")
    if rows is None:
        row_values = np.arange(len(high_values), dtype=np.int64)
    else:
        row_values = np.asarray(list(rows))
        if row_values.ndim != 1 or not np.issubdtype(row_values.dtype, np.integer):
            raise CodecFormatError(
                "row selectors must be a one-dimensional integer array"
            )
    if len(row_values) != len(high_values):
        raise CodecFormatError("row and high-byte selectors differ in length")
    if np.any(row_values < 0) or np.any(row_values >= probabilities.shape[0]):
        raise CodecFormatError("row selector out of range")
    selected = probabilities.reshape(-1, RADIX, RADIX)[
        row_values.astype(np.int64), high_values.astype(np.int64)
    ]
    return canonicalize_categorical(selected)


def _validated_full_probabilities(full_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(full_probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != VOCAB_SIZE:
        raise CodecProbabilityError(
            f"full PMF must have shape [N,{VOCAB_SIZE}], got {probabilities.shape}"
        )
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise CodecProbabilityError("full PMF is not floating point")
    if not np.all(np.isfinite(probabilities)):
        raise CodecProbabilityError("full PMF contains NaN or infinity")
    if np.any(probabilities < 0.0):
        raise CodecProbabilityError("full PMF contains negative mass")
    return probabilities


def pack_empty_blob() -> bytes:
    return HEADER.pack(0, 1, SCHEME_HIER256_D256)


def pack_blob(token_count: int, lanes: int, words: np.ndarray) -> bytes:
    validate_header_fields(token_count, lanes, SCHEME_HIER256_D256)
    if token_count == 0:
        array = np.asarray(words)
        if array.size:
            raise CodecFormatError("empty stream cannot carry a range-coded body")
        return pack_empty_blob()
    array = np.asarray(words)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise CodecFormatError(
            "range-coded body must be a one-dimensional integer word array"
        )
    if array.size == 0:
        raise CodecFormatError("nonempty stream requires at least one range word")
    if np.any(array < 0) or np.any(array > 0xFFFFFFFF):
        raise CodecFormatError("range word out of uint32 range")
    little = np.ascontiguousarray(array, dtype="<u4")
    raw = little.tobytes(order="C")
    if len(raw) > MAX_RANGE_BODY_BYTES:
        raise CodecFormatError("range-coded body exceeds mailbox response limit")
    checksum = zlib.crc32(raw) & 0xFFFFFFFF
    return (
        HEADER.pack(int(token_count), int(lanes), SCHEME_HIER256_D256)
        + BODY_META.pack(int(little.size), checksum)
        + raw
    )


def unpack_blob(
    blob: bytes, allowed_lanes: Collection[int] | None = None
) -> ParsedBlob:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise CodecFormatError("blob must be bytes-like")
    raw_blob = bytes(blob)
    if len(raw_blob) < HEADER.size:
        raise CodecFormatError(f"truncated header: {len(raw_blob)} bytes")
    token_count, lanes, scheme = HEADER.unpack_from(raw_blob, 0)
    validate_header_fields(token_count, lanes, scheme, allowed_lanes)
    if token_count == 0:
        if len(raw_blob) != HEADER.size:
            raise CodecFormatError("canonical empty stream has an unexpected body")
        return ParsedBlob(0, 1, np.empty(0, dtype=np.uint32))
    if len(raw_blob) < HEADER.size + BODY_META.size:
        raise CodecFormatError("truncated range-body metadata")
    word_count, expected_checksum = BODY_META.unpack_from(raw_blob, HEADER.size)
    if word_count == 0:
        raise CodecFormatError("nonempty stream declares zero range words")
    body_bytes = int(word_count) * 4
    if body_bytes > MAX_RANGE_BODY_BYTES:
        raise CodecFormatError("declared range body exceeds mailbox response limit")
    expected_size = HEADER.size + BODY_META.size + body_bytes
    if len(raw_blob) != expected_size:
        raise CodecFormatError(
            f"range body length mismatch: declared {body_bytes}, actual "
            f"{len(raw_blob) - HEADER.size - BODY_META.size}"
        )
    body = raw_blob[HEADER.size + BODY_META.size :]
    actual_checksum = zlib.crc32(body) & 0xFFFFFFFF
    if actual_checksum != expected_checksum:
        raise CodecFormatError(
            f"range body crc32 mismatch: expected {expected_checksum:08x}, "
            f"got {actual_checksum:08x}"
        )
    words = np.frombuffer(body, dtype="<u4").astype(np.uint32, copy=True)
    return ParsedBlob(int(token_count), int(lanes), words)
