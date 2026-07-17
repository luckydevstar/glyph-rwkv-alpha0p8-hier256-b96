"""Isolated full-PMF hier256 codec for Charlie UID95's RWKV-7 runtime.

The extracted UID95 files remain untouched.  A gated deployment copies that tree to a
new directory and overlays this module.  The recurrent model advances exactly once per
position.  Its existing FP32 softmax remains resident on GPU; compression transfers the
256 high-byte masses and the truth-selected 256-low-byte block in one D2H operation,
while decompression transfers high masses first and the decoded block second.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _path in (_PARENT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = (
        "/scratch/.triton"
        if os.path.isdir("/scratch")
        else os.path.join(_HERE, ".triton_cache")
    )

import constriction  # noqa: E402 - CUDA env and image paths must be fixed first
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hier256_core import (  # noqa: E402
    CodecFormatError,
    CodecProbabilityError,
    EOS_SYMBOLS,
    RADIX,
    SCHEME_HIER256_D256,
    VOCAB_SIZE,
    canonicalize_categorical,
    chunks,
    eos_categorical,
    pack_blob,
    pack_empty_blob,
    runtime_batch_default,
    unpack_blob,
    validate_header_fields,
    validate_tokens,
)


# Engine still receives its original top-K width while loading/capturing.  This width
# does not participate in hier256 coding and never appears in the D256 header.
ENGINE_COMPAT_K = int(os.environ.get("GLYPH_K", "4096"))
B_FIXED = (
    int(os.environ["GLYPH_B_FIXED"])
    if "GLYPH_B_FIXED" in os.environ
    else runtime_batch_default(_HERE, RADIX)
)


def _parse_allowed_batches() -> frozenset[int]:
    raw = os.environ.get("GLYPH_HIER256_ALLOWED_B", str(B_FIXED))
    try:
        values = frozenset(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid GLYPH_HIER256_ALLOWED_B={raw!r}") from exc
    if not values or any(value < 1 or value > RADIX for value in values):
        raise RuntimeError(f"invalid GLYPH_HIER256_ALLOWED_B={raw!r}")
    if B_FIXED not in values:
        raise RuntimeError(
            f"GLYPH_B_FIXED={B_FIXED} is absent from GLYPH_HIER256_ALLOWED_B={sorted(values)}"
        )
    return values


ALLOWED_BATCHES = _parse_allowed_batches()


def _find_model() -> str:
    if os.environ.get("GLYPH_MODEL"):
        return os.environ["GLYPH_MODEL"]
    for candidate in (
        os.path.join(_HERE, "model_pack.pth"),
        os.path.join(_PARENT, "model_pack.pth"),
        "/opt/codec/model/model_pack.pth",
        "/opt/codec/model/model_int8.safetensors",
        "/opt/model/model_pack.pth",
    ):
        if os.path.exists(candidate):
            return candidate
    development = "/root/glyph-lab/rwkv_models/rwkv7-g1h-13.3b-20260710-ctx10240.pth"
    if os.path.exists(development):
        return development
    raise FileNotFoundError("no model: set GLYPH_MODEL or retain the UID95 image model")


def _find_vocab() -> str:
    if os.environ.get("GLYPH_VOCAB"):
        return os.environ["GLYPH_VOCAB"]
    for candidate in (
        os.path.join(_HERE, "rwkv_vocab_v20230424.txt"),
        os.path.join(_PARENT, "rwkv_vocab_v20230424.txt"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("no vocab: set GLYPH_VOCAB or retain UID95's bundled vocab")


class Codec:
    """Tokenizer, cached UID95 engine, and fail-closed D256 coder."""

    def __init__(self, model_path: str | None = None, vocab_path: str | None = None):
        from rwkv_tokenizer import RWKVTokenizer

        self.model_path = model_path or _find_model()
        self.vocab_path = vocab_path or _find_vocab()
        self.tok = RWKVTokenizer(self.vocab_path)
        self.fam = constriction.stream.model.Categorical(perfect=False)
        self._eng = None
        self.V: int | None = None

    def engine(self, lanes: int):
        lanes = int(lanes)
        if lanes not in ALLOWED_BATCHES:
            raise CodecFormatError(
                f"lanes={lanes} has no approved/captured engine; allowed={sorted(ALLOWED_BATCHES)}"
            )
        if self._eng is None or self._eng.B != lanes:
            if self._eng is not None:
                self._eng = None
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            from engine import Engine

            engine = Engine(self.model_path, lanes, ENGINE_COMPAT_K)
            engine.capture()
            self._eng = engine
        self.V = int(self._eng.emb.shape[0])
        if self.V != VOCAB_SIZE:
            raise CodecProbabilityError(
                f"hier256 requires exactly {VOCAB_SIZE} logits, engine exposes {self.V}"
            )
        self._eng.reset()
        return self._eng

    def reset(self) -> None:
        if self._eng is not None:
            self._eng.reset()

    @staticmethod
    def _step_full_probabilities(engine, tokens: torch.Tensor):
        """One and only one recurrent step, followed by UID95's FP32 softmax."""
        engine.inp.copy_(tokens.to(engine.inp.device, non_blocking=True))
        if engine.graph is not None:
            engine.graph.replay()
        else:
            with torch.no_grad():
                engine._step()
        with torch.no_grad():
            probabilities = F.softmax(engine.logits.float(), dim=-1)
            if probabilities.ndim != 2 or probabilities.shape[1] != VOCAB_SIZE:
                raise CodecProbabilityError(
                    f"full softmax has shape {tuple(probabilities.shape)}, expected [B,{VOCAB_SIZE}]"
                )
            high = probabilities.view(probabilities.shape[0], RADIX, RADIX).sum(dim=-1)
        return probabilities, high

    @staticmethod
    def _compression_surfaces(
        probabilities: torch.Tensor,
        high_probabilities: torch.Tensor,
        active: np.ndarray,
        truth_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather high and truth-selected low rows into one 512-float D2H copy."""
        active_device = torch.as_tensor(
            active, dtype=torch.long, device=probabilities.device
        )
        high_device = torch.as_tensor(
            truth_high, dtype=torch.long, device=probabilities.device
        )
        selected_low = probabilities.view(-1, RADIX, RADIX)[active_device, high_device]
        joined = torch.cat((high_probabilities[active_device], selected_low), dim=1)
        host = joined.cpu().numpy()
        return (
            canonicalize_categorical(host[:, :RADIX]),
            canonicalize_categorical(host[:, RADIX:]),
        )

    @staticmethod
    def _decode_high_surface(
        high_probabilities: torch.Tensor, active: np.ndarray
    ) -> np.ndarray:
        active_device = torch.as_tensor(
            active, dtype=torch.long, device=high_probabilities.device
        )
        return canonicalize_categorical(high_probabilities[active_device].cpu().numpy())

    @staticmethod
    def _decode_low_surface(
        probabilities: torch.Tensor, active: np.ndarray, decoded_high: np.ndarray
    ) -> np.ndarray:
        active_device = torch.as_tensor(
            active, dtype=torch.long, device=probabilities.device
        )
        high_device = torch.as_tensor(
            decoded_high, dtype=torch.long, device=probabilities.device
        )
        selected = probabilities.view(-1, RADIX, RADIX)[active_device, high_device]
        return canonicalize_categorical(selected.cpu().numpy())

    def compress_bytes(self, data: bytes, force_B: int | None = None) -> bytes:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise CodecFormatError("compress input must be bytes-like")
        tokens = validate_tokens(self.tok.encode(bytes(data)))
        token_count = int(tokens.size)
        if token_count == 0:
            return pack_empty_blob()
        lanes = B_FIXED if force_B is None else int(force_B)
        validate_header_fields(
            token_count,
            lanes,
            SCHEME_HIER256_D256,
            allowed_lanes=ALLOWED_BATCHES,
        )
        lengths = chunks(token_count, lanes)
        starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
        longest = max(lengths)
        matrix = np.zeros((lanes, longest), dtype=np.int64)
        for lane, (start, length) in enumerate(zip(starts, lengths)):
            matrix[lane, :length] = tokens[int(start) : int(start) + length]

        engine = self.engine(lanes)
        encoder = constriction.stream.queue.RangeEncoder()
        previous = torch.zeros(lanes, dtype=torch.long)
        length_array = np.asarray(lengths)
        for position in range(longest):
            probabilities, high_probabilities = self._step_full_probabilities(
                engine, previous
            )
            active = np.flatnonzero(position < length_array)
            truth = matrix[active, position]
            truth_high = np.ascontiguousarray(truth >> 8, dtype=np.int32)
            truth_low = np.ascontiguousarray(truth & 0xFF, dtype=np.int32)
            high_surface, low_surface = self._compression_surfaces(
                probabilities, high_probabilities, active, truth_high
            )
            encoder.encode(truth_high, self.fam, high_surface)
            encoder.encode(truth_low, self.fam, low_surface)
            previous = torch.from_numpy(matrix[:, position])
        encoder.encode(EOS_SYMBOLS, self.fam, eos_categorical())
        return pack_blob(token_count, lanes, encoder.get_compressed())

    def decompress_bytes(self, blob: bytes) -> bytes:
        parsed = unpack_blob(blob, allowed_lanes=ALLOWED_BATCHES)
        if parsed.token_count == 0:
            return self.tok.decode([])
        if self._eng is not None and parsed.lanes != self._eng.B:
            raise CodecFormatError(
                f"blob lanes={parsed.lanes} differ from captured engine B={self._eng.B}"
            )
        lengths = chunks(parsed.token_count, parsed.lanes)
        longest = max(lengths)
        length_array = np.asarray(lengths)
        recovered = np.zeros((parsed.lanes, longest), dtype=np.int64)
        previous = torch.zeros(parsed.lanes, dtype=torch.long)
        engine = self.engine(parsed.lanes)
        try:
            decoder = constriction.stream.queue.RangeDecoder(parsed.words)
            # Re-encode each decoded symbol against the identical PMF. This adds no
            # model step and rejects a body that is valid only as a prefix, has trailing
            # words, or is otherwise not constriction's canonical queue stream.
            canonical_encoder = constriction.stream.queue.RangeEncoder()
            for position in range(longest):
                probabilities, high_probabilities = self._step_full_probabilities(
                    engine, previous
                )
                active = np.flatnonzero(position < length_array)
                high_surface = self._decode_high_surface(high_probabilities, active)
                high = np.asarray(
                    decoder.decode(self.fam, high_surface), dtype=np.int64
                ).reshape(-1)
                if (
                    len(high) != len(active)
                    or np.any(high < 0)
                    or np.any(high >= RADIX)
                ):
                    raise CodecFormatError(
                        "range decoder returned malformed high-byte symbols"
                    )
                canonical_encoder.encode(
                    np.ascontiguousarray(high, dtype=np.int32),
                    self.fam,
                    high_surface,
                )
                low_surface = self._decode_low_surface(probabilities, active, high)
                low = np.asarray(
                    decoder.decode(self.fam, low_surface), dtype=np.int64
                ).reshape(-1)
                if len(low) != len(active) or np.any(low < 0) or np.any(low >= RADIX):
                    raise CodecFormatError(
                        "range decoder returned malformed low-byte symbols"
                    )
                canonical_encoder.encode(
                    np.ascontiguousarray(low, dtype=np.int32), self.fam, low_surface
                )
                recovered[active, position] = (high << 8) | low
                previous = torch.from_numpy(recovered[:, position])
            footer_surface = eos_categorical()
            footer = np.asarray(
                decoder.decode(self.fam, footer_surface), dtype=np.int64
            ).reshape(-1)
            if not np.array_equal(footer, EOS_SYMBOLS.astype(np.int64)):
                raise CodecFormatError(
                    f"range footer mismatch: got {footer.tolist()}, "
                    f"expected {EOS_SYMBOLS.tolist()}"
                )
            canonical_encoder.encode(EOS_SYMBOLS, self.fam, footer_surface)
            canonical_words = np.asarray(
                canonical_encoder.get_compressed(), dtype=np.uint32
            ).reshape(-1)
            if not np.array_equal(canonical_words, parsed.words):
                raise CodecFormatError(
                    "range body is noncanonical, has trailing words, or was not fully consumed"
                )
        except CodecFormatError:
            raise
        except Exception as exc:
            raise CodecFormatError(
                f"range body decode failed: {type(exc).__name__}: {exc}"
            ) from exc

        output: list[int] = []
        for lane, length in enumerate(lengths):
            output.extend(recovered[lane, :length].tolist())
        canonical_tokens = validate_tokens(output)
        try:
            decoded = self.tok.decode(canonical_tokens.tolist())
        except Exception as exc:
            raise CodecFormatError(f"tokenizer rejected decoded tokens: {exc}") from exc
        if not isinstance(decoded, bytes):
            raise CodecFormatError("tokenizer decode did not return bytes")
        return decoded


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="development D256 file roundtrip driver"
    )
    parser.add_argument("mode", choices=("compress", "decompress"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=B_FIXED)
    args = parser.parse_args()
    codec = Codec()
    with open(args.input, "rb") as source:
        payload = source.read()
    output = (
        codec.compress_bytes(payload, force_B=args.batch)
        if args.mode == "compress"
        else codec.decompress_bytes(payload)
    )
    temporary = f"{args.output}.tmp-{os.getpid()}"
    with open(temporary, "wb") as destination:
        destination.write(output)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    _main()
