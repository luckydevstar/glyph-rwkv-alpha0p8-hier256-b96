"""Timed compress entrypoint for the rwkv13-v2 codec.

argv: `compress.py <input> <output>` — also accepts the v1 style
`compress.py compress --input X --output Y` (flexible parse).

Thin stdlib-only client (precheck-scanned; importing this file never pulls
torch/triton into the timed window).  If the warmup-launched daemon is ready,
one mailbox request does the work and the server returns the blob ALREADY
prefixed with the 0x10 neural method byte.  Otherwise: 2 reconnect probes 1 s
apart, then a classical fallback blob that decompress.py can decode without
the model.

Method-byte container (first byte of every blob):
  0x10 neural | 0x02 lzma | 0x00 raw-stored.

Output is published via same-dir temp + os.replace — the validator only
checks exit code + output existence and must never see a partial file.
NOTHING is printed on stdout; stderr is free for diagnostics.
"""

import lzma
import os
import secrets
import subprocess
import sys
import time

_ENTRY = time.monotonic()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbclient

TOTAL_BUDGET_SECS = 440.0  # inside the validator's 450 s per-op exec timeout
REQUEST_TIMEOUT_SECS = 420.0
RECONNECT_PROBES = 2
DEFAULT_SERVER_PATH = "/opt/codec/server.py"
METHOD_NEURAL = b"\x10"
METHOD_LZMA = b"\x02"
METHOD_RAW = b"\x00"


def _remaining():
    return TOTAL_BUDGET_SECS - (time.monotonic() - _ENTRY)


def _log_default():
    if os.path.isdir("/scratch") and os.access("/scratch", os.W_OK):
        return "/scratch/rwkv-daemon.log"
    import tempfile
    return os.path.join(tempfile.gettempdir(), "rwkv-daemon.log")


def _launch_daemon():
    """Launch the daemon ourselves when no warm one exists (e.g. local
    glyph-miner check runs no warmup). Returns the validated ready dict or None.
    On the real validator, warmup already started it, so this never runs."""
    server = _server_path()
    if not os.path.isfile(server):
        return None
    token = secrets.token_hex(16)
    log_fd = os.open(_log_default(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, server, "--token", token],
            stdin=subprocess.DEVNULL, stdout=log_fd, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True, env=env,
        )
    except OSError:
        return None
    finally:
        os.close(log_fd)
    budget = min(300.0, _remaining() - 30.0)
    if budget <= 0.0:
        return None
    print("compress: no warm daemon; launched pid=%d, waiting up to %.0fs"
          % (proc.pid, budget), file=sys.stderr)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        r = mbclient.read_ready()
        if r is not None and r.get("pid") == proc.pid:
            return r
        time.sleep(0.5)
    return None


def _server_path():
    p = os.environ.get("GLYPH_SERVER_PATH")
    if p:
        return p
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    if os.path.isfile(here):          # self-contained: our own server sits alongside
        return here
    return DEFAULT_SERVER_PATH         # image deploy: server is baked at /opt/codec


def _parse_argv(argv):
    """Accept `<input> <output>`, `compress --input X --output Y`, and mixes."""
    input_path = None
    output_path = None
    positional = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--input" and index + 1 < len(argv):
            input_path = argv[index + 1]
            index += 2
            continue
        if arg == "--output" and index + 1 < len(argv):
            output_path = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--input="):
            input_path = arg.split("=", 1)[1]
            index += 1
            continue
        if arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
            index += 1
            continue
        if arg in ("compress", "decompress"):
            index += 1
            continue
        positional.append(arg)
        index += 1
    if input_path is None and positional:
        input_path = positional.pop(0)
    if output_path is None and positional:
        output_path = positional.pop(0)
    return input_path, output_path


def _daemon_compress(data):
    """Neural 0x10 blob from the warm daemon, or None -> classical fallback."""
    if len(data) > mbclient.MAX_REQUEST_BYTES:
        return None  # cannot frame it; classical fallback still round-trips
    ready = mbclient.read_ready()  # validated + zombie-aware pid liveness
    if ready is None:
        # No warm daemon. On the real validator warmup started one; in a local
        # glyph-miner check (no warmup) we launch it ourselves so the neural
        # path is exercised instead of the classical fallback.
        ready = _launch_daemon()
    if ready is None:
        return None  # no daemon and none launchable -> classical fallback
    for attempt in range(1 + RECONNECT_PROBES):
        if attempt:
            time.sleep(1.0)  # reconnect probes, 1 s apart
            ready = mbclient.read_ready()
        if ready is None:
            continue
        timeout_s = min(REQUEST_TIMEOUT_SECS, _remaining() - 10.0)
        if timeout_s <= 5.0:
            return None  # keep enough budget to write a fallback blob
        try:
            blob = mbclient.send_request(
                None,
                ready["token"],
                mbclient.OP_COMPRESS,
                data,
                timeout_s=timeout_s,
            )
        except mbclient.MailboxError as error:
            # DaemonDiedError / CodecError / protocol error / timeout
            print(
                "compress: daemon attempt %d failed: %s" % (attempt + 1, error),
                file=sys.stderr,
            )
            continue
        if blob[:1] == METHOD_NEURAL:
            return blob  # server already prefixed the 0x10 method byte
        print(
            "compress: daemon returned a non-neural blob; using fallback",
            file=sys.stderr,
        )
        return None
    return None


def _classical(data):
    """Classical fallback blob, decodable without the model."""
    compressed = lzma.compress(data, preset=6)
    if len(compressed) < len(data):
        return METHOD_LZMA + compressed
    return METHOD_RAW + data  # raw-stored: never larger than input + 1


def main():
    input_path, output_path = _parse_argv(sys.argv[1:])
    if not input_path or not output_path:
        print("usage: compress.py <input> <output>", file=sys.stderr)
        return 1
    try:
        with open(input_path, "rb") as handle:
            data = handle.read()
    except OSError as error:
        print("compress: cannot read input: %s" % error, file=sys.stderr)
        return 1

    blob = None
    try:
        blob = _daemon_compress(data)
    except Exception as error:  # the daemon path must never kill the phase
        print("compress: daemon path error: %s" % error, file=sys.stderr)
        blob = None
    if blob is None:
        blob = _classical(data)

    try:
        mbclient.atomic_write(output_path, blob)
    except (OSError, mbclient.MailboxError) as error:
        print("compress: cannot write output: %s" % error, file=sys.stderr)
        return 1
    try:
        os.chmod(output_path, 0o644)  # readable by the host-side validator
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
