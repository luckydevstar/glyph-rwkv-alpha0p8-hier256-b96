"""Timed decompress entrypoint for the rwkv13-v2 codec.

argv: `decompress.py <input> <output>` — also accepts the v1 style
`decompress.py decompress --input X --output Y` (flexible parse).

Thin stdlib-only client (precheck-scanned).  Dispatches on the method byte
(first byte of the blob):
  0x02 lzma  -> stdlib decode, no daemon needed
  0x00 raw   -> stored bytes, no daemon needed
  0x10 neural-> mailbox request; the FULL blob including the 0x10 byte is
                sent and the server returns the raw stream bytes.

If the daemon is dead for a neural blob, the LAST RESORT is relaunching
/opt/codec/server.py exactly like warmup does (spec fallback ladder #3) and
eating the model load inside the exec window: strictly better than a
guaranteed failed stream.  All waits are budgeted against a deadline started
at process entry (total 440 s, inside the validator's 450 s exec timeout).

Output is published via same-dir temp + os.replace.  NOTHING on stdout.
Exit 0 on success, 1 on failure.
"""

import lzma
import zlib
import os
import secrets
import subprocess
import sys
import time

_ENTRY = time.monotonic()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbclient

TOTAL_BUDGET_SECS = 440.0
REQUEST_TIMEOUT_SECS = 420.0
RELAUNCH_READY_SECS = 120.0
POLL_INTERVAL_SECS = 0.02
DEFAULT_SERVER_PATH = "/opt/codec/server.py"
def _log_default():
    if os.path.isdir("/scratch") and os.access("/scratch", os.W_OK):
        return "/scratch/rwkv-daemon.log"
    import tempfile
    return os.path.join(tempfile.gettempdir(), "rwkv-daemon.log")

DEFAULT_LOG_PATH = _log_default()
METHOD_NEURAL = 0x10
METHOD_ZLIB = 0x01
METHOD_LZMA = 0x02
METHOD_RAW = 0x00


def _remaining():
    return TOTAL_BUDGET_SECS - (time.monotonic() - _ENTRY)


def _server_path():
    p = os.environ.get("GLYPH_SERVER_PATH")
    if p:
        return p
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    if os.path.isfile(here):          # self-contained: our own server sits alongside
        return here
    return DEFAULT_SERVER_PATH         # image deploy: server is baked at /opt/codec


def _log_path():
    return os.environ.get("GLYPH_DAEMON_LOG", DEFAULT_LOG_PATH)


def _parse_argv(argv):
    """Accept `<input> <output>`, `decompress --input X --output Y`, and mixes."""
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


def _log_tail(max_bytes=4096):
    try:
        descriptor = os.open(_log_path(), os.O_RDONLY)
    except OSError:
        return ""
    try:
        size = os.lseek(descriptor, 0, os.SEEK_END)
        os.lseek(descriptor, max(0, size - max_bytes), os.SEEK_SET)
        data = os.read(descriptor, max_bytes)
    except OSError:
        return ""
    finally:
        os.close(descriptor)
    return data.decode("utf-8", errors="replace").strip()


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            pass


def _relaunch_daemon():
    """LAST RESORT: relaunch the daemon exactly like warmup does and wait for
    readiness (the server prepares its own mailbox before publishing).
    Returns the validated ready.json dict; raises MailboxError on failure."""
    server = _server_path()
    if not os.path.isfile(server):
        raise mbclient.MailboxError(
            "neural blob but daemon is dead and no server exists at %s" % server
        )
    token = secrets.token_hex(16)
    log_fd = os.open(_log_path(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, server, "--token", token],
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=environment,
        )
    except OSError as error:
        raise mbclient.MailboxError("cannot relaunch daemon: %s" % error)
    finally:
        os.close(log_fd)

    ready_budget = min(RELAUNCH_READY_SECS, _remaining() - 30.0)
    if ready_budget <= 0.0:
        _stop_process(process)
        raise mbclient.MailboxError("no time budget left to relaunch the daemon")
    print(
        "decompress: daemon dead; relaunched pid=%d, waiting up to %.0fs"
        % (process.pid, ready_budget),
        file=sys.stderr,
    )
    deadline = time.monotonic() + ready_budget
    while True:
        code = process.poll()
        if code is not None:
            tail = _log_tail()
            detail = ("; log tail:\n" + tail) if tail else ""
            raise mbclient.DaemonDiedError(
                "relaunched daemon exited with status %s before readiness%s"
                % (code, detail)
            )
        ready = mbclient.read_ready()
        if ready is not None:
            if ready["pid"] != process.pid or ready["token"] != token:
                _stop_process(process)
                raise mbclient.MailboxError(
                    "relaunched daemon readiness identity mismatch (pid/token)"
                )
            return ready
        if time.monotonic() >= deadline:
            _stop_process(process)
            raise mbclient.MailboxError(
                "relaunched daemon not ready within %.0fs" % ready_budget
            )
        time.sleep(POLL_INTERVAL_SECS)


def _neural_decompress(blob):
    """Send the FULL blob (incl. its 0x10 byte); the server returns raw bytes."""
    if len(blob) > mbclient.MAX_REQUEST_BYTES:
        raise mbclient.MailboxError("neural blob exceeds the 64 MiB request cap")
    ready = mbclient.read_ready()  # validated + zombie-aware pid liveness
    relaunched = False
    if ready is None:
        ready = _relaunch_daemon()
        relaunched = True
    timeout_s = min(REQUEST_TIMEOUT_SECS, _remaining() - 5.0)
    if timeout_s <= 0.0:
        raise mbclient.MailboxError("no time budget left for the daemon request")
    try:
        return mbclient.send_request(
            None, ready["token"], mbclient.OP_DECOMPRESS, blob, timeout_s=timeout_s
        )
    except mbclient.DaemonDiedError:
        # Daemon died mid-request: one relaunch if we have not already paid it
        # and enough budget remains for a load + request.
        if relaunched or _remaining() < 60.0:
            raise
        ready = _relaunch_daemon()
        timeout_s = min(REQUEST_TIMEOUT_SECS, _remaining() - 5.0)
        if timeout_s <= 0.0:
            raise
        return mbclient.send_request(
            None, ready["token"], mbclient.OP_DECOMPRESS, blob, timeout_s=timeout_s
        )


def main():
    input_path, output_path = _parse_argv(sys.argv[1:])
    if not input_path or not output_path:
        print("usage: decompress.py <input> <output>", file=sys.stderr)
        return 1
    try:
        with open(input_path, "rb") as handle:
            blob = handle.read()
    except OSError as error:
        print("decompress: cannot read input: %s" % error, file=sys.stderr)
        return 1
    if not blob:
        print("decompress: empty blob (missing method byte)", file=sys.stderr)
        return 1

    method = blob[0]
    try:
        if method == METHOD_LZMA:
            data = lzma.decompress(blob[1:])
        elif method == METHOD_ZLIB:
            data = zlib.decompress(blob[1:])
        elif method == METHOD_RAW:
            data = blob[1:]
        elif method == METHOD_NEURAL:
            data = _neural_decompress(blob)
        else:
            print("decompress: unknown method byte 0x%02x" % method, file=sys.stderr)
            return 1
    except (mbclient.MailboxError, lzma.LZMAError, zlib.error, OSError) as error:
        print("decompress: %s" % error, file=sys.stderr)
        return 1

    try:
        mbclient.atomic_write(output_path, data)
    except (OSError, mbclient.MailboxError) as error:
        print("decompress: cannot write output: %s" % error, file=sys.stderr)
        return 1
    try:
        os.chmod(output_path, 0o644)  # readable by the host-side validator
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
