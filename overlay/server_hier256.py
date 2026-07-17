"""Gated UID95 daemon fork that retains outer neural method byte 0x10."""

from __future__ import annotations

import importlib.util
import os
import secrets
import sys
import time


_TOKEN = None
for _index, _argument in enumerate(sys.argv[1:]):
    if _argument == "--token" and _index + 2 <= len(sys.argv[1:]):
        _TOKEN = sys.argv[1:][_index + 1]
    elif _argument.startswith("--token="):
        _TOKEN = _argument.split("=", 1)[1]

_LOG_PATH = os.environ.get("GLYPH_DAEMON_LOG", "/scratch/rwkv-hier256-daemon.log")
if _TOKEN is not None:
    try:
        _fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(_fd, 1)
        os.dup2(_fd, 2)
        if _fd > 2:
            os.close(_fd)
    except OSError:
        pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TRITON_CACHE_DIR", "/scratch/.triton")


def _log(message: str) -> None:
    print(f"[rwkv13-v2-hier256 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _seed_triton_cache() -> None:
    source = os.environ.get("GLYPH_TRITON_SEED", "/opt/codec/triton_seed")
    destination = os.environ["TRITON_CACHE_DIR"]
    try:
        if os.path.isdir(source):
            import shutil

            os.makedirs(destination, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
            _log(f"triton seed cache merged {source} -> {destination}")
    except Exception as exc:
        _log(f"triton seed copy failed (non-fatal): {exc!r}")


_seed_triton_cache()
_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_mailbox_path = os.path.join(_HERE, "mailbox.py")
_mailbox_spec = importlib.util.spec_from_file_location("glyph_mailbox", _mailbox_path)
if _mailbox_spec is None or _mailbox_spec.loader is None:
    raise RuntimeError(f"mailbox transport missing: {_mailbox_path}")
mailbox = importlib.util.module_from_spec(_mailbox_spec)
_mailbox_spec.loader.exec_module(mailbox)
for _required in ("prepare_mailbox", "publish_ready", "serve_forever"):
    if not hasattr(mailbox, _required):
        raise RuntimeError(f"mailbox.py lacks required function {_required!r}")

import coder_hier256 as coder  # noqa: E402 - mailbox/path bootstrap precedes CUDA import


CONTRACT_ID = "rwkv13-v2-hier256-d256"
PROTOCOL = 1
MAILBOX_DIR = os.environ.get("GLYPH_MAILBOX_DIR", "/scratch/rwkv-mailbox-v1")
IDLE_TIMEOUT_SECS = float(os.environ.get("GLYPH_IDLE_TIMEOUT_SECS", "1800"))
METHOD_NEURAL = b"\x10"
VERSION_BODY = b"rwkv13-v2-hier256-d256"

_SELF_CHECK = (
    (
        "glyph rwkv13-v2 hier256-d256 daemon self-check. The quick brown fox jumps "
        "over the lazy dog; pack my box with five dozen liquor jugs. 0123456789 "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua.\n"
    )
    * 10
).encode("utf-8")


def _make_handler(codec: "coder.Codec", fixed_batch: int):
    def handler(operation: bytes, payload: bytes) -> bytes:
        if operation == b"V":
            return VERSION_BODY
        if operation == b"C":
            # Codec.engine() performs the one required state reset.  An outer reset
            # would traverse the multi-GiB recurrent state twice per request.
            return METHOD_NEURAL + codec.compress_bytes(payload, force_B=fixed_batch)
        if operation == b"D":
            if len(payload) < 1 or payload[:1] != METHOD_NEURAL:
                raise ValueError(
                    f"decompress expected outer neural method byte 0x10, got {payload[:1]!r}"
                )
            return codec.decompress_bytes(payload[1:])
        raise ValueError(f"unknown op byte {operation!r}")

    return handler


def main() -> int:
    token = (_TOKEN if _TOKEN is not None else secrets.token_hex(16)).lower()
    try:
        if len(bytes.fromhex(token)) != 16:
            raise ValueError
    except ValueError:
        _log("bad --token (need 32 hex chars); refusing to start")
        return 2

    _log(f"starting pid={os.getpid()} mailbox={MAILBOX_DIR} log={_LOG_PATH}")
    start = time.time()
    codec = coder.Codec()
    _log(f"tokenizer ready vocab={codec.vocab_path} in {time.time() - start:.1f}s")
    fixed_batch = int(os.environ.get("GLYPH_B_FIXED", str(coder.B_FIXED)))
    start = time.time()
    codec.engine(fixed_batch)
    _log(
        f"engine ready B={fixed_batch} scheme=0xD256 V={codec.V} "
        f"in {time.time() - start:.1f}s"
    )

    start = time.time()
    blob = codec.compress_bytes(_SELF_CHECK, force_B=fixed_batch)
    recovered = codec.decompress_bytes(blob)
    if recovered != _SELF_CHECK:
        raise RuntimeError(
            f"hier256 self-check mismatch: {len(_SELF_CHECK)} bytes in, "
            f"{len(recovered)} bytes out"
        )
    _log(
        f"self-check ok: {len(_SELF_CHECK)} -> {len(blob)} -> {len(recovered)} "
        f"bytes in {time.time() - start:.1f}s"
    )
    if coder.torch.cuda.is_available():
        coder.torch.cuda.synchronize()

    handler = _make_handler(codec, fixed_batch)
    mailbox.prepare_mailbox(MAILBOX_DIR)
    mailbox.publish_ready(
        MAILBOX_DIR,
        {
            "token": token,
            "limits": {
                "max_request_bytes": 64 * 2**20,
                "max_response_bytes": 128 * 2**20,
            },
        },
    )
    _log(
        f"ready contract_id={CONTRACT_ID} protocol={PROTOCOL}; "
        f"idle timeout={IDLE_TIMEOUT_SECS:.0f}s"
    )
    mailbox.serve_forever(MAILBOX_DIR, token, handler, IDLE_TIMEOUT_SECS)
    _log("serve loop exited cleanly")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1)
