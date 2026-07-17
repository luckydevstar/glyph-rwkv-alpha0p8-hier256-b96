"""Artifact-side client for the rwkv13-v2 file mailbox (SN117 glyph).

STDLIB-ONLY by contract: the validator precheck AST-scans every artifact .py
file (precheck.py:58-127).  This module imports nothing but os, struct, time,
secrets, json.  No sockets, no URLs, no subprocess, no torch.  It is a
top-level module with no package __init__ so importing it costs ~0 inside the
timed window.

Interface contract (fixed, shared with /opt/codec image side):
  - mailbox dir  : /scratch/rwkv-mailbox-v1  (env GLYPH_MAILBOX_DIR overrides)
  - request frame: struct '<8s16scQQ' = magic b'RWKVMB01', 16-byte token,
                   op byte (b'C'/b'D'/b'V'), max_response echo, payload_len;
                   then payload bytes (cap 64 MiB)
  - response     : struct '<8s16sBQ' = magic, token, status (0 OK / 1 ERROR),
                   body_len; then body (cap 128 MiB; ERROR body = utf-8
                   message <= 4096 bytes)
  - ready.json   : {"contract_id": "rwkv13-v2", "pid": int, "token": hex,
                   "protocol": 1}

Every publish is an atomic same-directory temp + os.replace; every read is
O_NOFOLLOW + regular-file-checked + byte-capped, so a torn or foreign file is
a detectable protocol error, never silent corruption (king_qwen
kv4_mailbox.py pattern).
"""

import json
import os
import secrets
import struct
import time

CONTRACT_ID = "rwkv13-v2"
PROTOCOL_VERSION = 1
DEFAULT_MAILBOX_DIR = "/scratch/rwkv-mailbox-v1"
MAILBOX_DIR_ENV = "GLYPH_MAILBOX_DIR"

MAGIC = b"RWKVMB01"
REQUEST_HEADER = struct.Struct("<8s16scQQ")
RESPONSE_HEADER = struct.Struct("<8s16sBQ")
MAX_REQUEST_BYTES = 64 * 2**20
MAX_RESPONSE_BYTES = 128 * 2**20
MAX_READY_BYTES = 4096
MAX_ERROR_BYTES = 4096
STATUS_OK = 0
STATUS_ERROR = 1
POLL_INTERVAL_SECS = 0.02
OP_COMPRESS = b"C"
OP_DECOMPRESS = b"D"
OP_VERSION = b"V"
VALID_OPS = (OP_COMPRESS, OP_DECOMPRESS, OP_VERSION)

READY_NAME = "ready.json"
REQUEST_NAME = "request.ready"
CLAIMED_NAME = "request.claimed"
CLIENT_CLAIM_NAME = "client.claim"
RESPONSE_NAME = "response.ready"

_S_IFMT = 0o170000
_S_IFREG = 0o100000
_S_IFDIR = 0o040000
_O_EXTRA = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class MailboxError(RuntimeError):
    """The mailbox lifecycle or local file contract failed."""


class MailboxProtocolError(MailboxError):
    """A published mailbox frame does not match the pinned protocol."""


class DaemonDiedError(MailboxError):
    """The daemon is absent, a zombie, or exited without responding."""


class CodecError(MailboxError):
    """The daemon answered with a STATUS_ERROR frame."""


def default_mailbox_dir():
    env = os.environ.get(MAILBOX_DIR_ENV)
    if env:
        return env
    parent = os.path.dirname(DEFAULT_MAILBOX_DIR)
    if os.path.isdir(parent) and os.access(parent, os.W_OK):
        return DEFAULT_MAILBOX_DIR
    import tempfile
    return os.path.join(tempfile.gettempdir(), "rwkv-mailbox-v1")


def _token_bytes(token):
    """Normalize a launch token (32-hex str or 16 raw bytes) to 16 bytes."""
    if isinstance(token, (bytes, bytearray)):
        if len(token) != 16:
            raise ValueError("token bytes must be exactly 16 bytes")
        return bytes(token)
    if isinstance(token, str) and len(token) == 32:
        try:
            return bytes.fromhex(token)
        except ValueError:
            pass
    raise ValueError("token must be 32 hex characters or 16 raw bytes")


def atomic_write(path, data):
    """Atomically publish ``path``: O_EXCL random same-dir temp, write+fsync,
    os.replace, fsync directory."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    data = bytes(data)
    directory = os.path.dirname(os.path.abspath(path))
    temporary = os.path.join(directory, ".tmp-" + secrets.token_hex(8))
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_EXTRA, 0o600
        )
    except OSError as error:
        raise MailboxError(f"cannot create temp for {path}: {error}") from error
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError as error:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise MailboxError(f"cannot atomically publish {path}: {error}") from error
    _fsync_directory(directory)


def _fsync_directory(directory):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def read_bounded(path, cap):
    """Read one non-symlink regular file of at most ``cap`` bytes (exact)."""
    if type(cap) is not int or cap < 0:
        raise ValueError("cap must be a non-negative integer")
    try:
        descriptor = os.open(path, os.O_RDONLY | _O_EXTRA)
    except OSError as error:
        raise MailboxError(f"cannot open {path}: {error}") from error
    try:
        mode = os.fstat(descriptor).st_mode
        if (mode & _S_IFMT) != _S_IFREG:
            raise MailboxError(f"path is not a regular file: {path}")
        chunks = []
        total = 0
        while total <= cap:
            chunk = os.read(descriptor, min(1 << 20, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > cap:
            raise MailboxError(f"{path} exceeds its {cap}-byte limit")
        return b"".join(chunks)
    except OSError as error:
        raise MailboxError(f"cannot read {path}: {error}") from error
    finally:
        os.close(descriptor)


def _read_proc_stat(pid):
    """Read /proc/<pid>/stat; module-level so tests can monkeypatch it."""
    return read_bounded("/proc/%d/stat" % pid, 4096)


def pid_alive(pid):
    """True if ``pid`` is a live, non-zombie process.

    The container's PID 1 (``sleep infinity``) never reaps reparented
    children, so a dead daemon lingers as a zombie and kill(pid, 0) alone
    lies: /proc/<pid>/stat state 'Z' means DEAD.
    """
    if type(pid) is not int or pid <= 0:
        return False
    try:
        status = _read_proc_stat(pid)
    except MailboxError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    closing = status.rfind(b")")
    if closing >= 0:
        fields = status[closing + 1 :].split()
        if fields and fields[0] == b"Z":
            return False
    return True


def read_ready(mailbox_dir=None):
    """Validated ready.json as a dict, or None (absent/torn/foreign/dead pid)."""
    directory = mailbox_dir or default_mailbox_dir()
    try:
        raw = read_bounded(os.path.join(directory, READY_NAME), MAX_READY_BYTES)
        document = json.loads(raw)
    except (MailboxError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("contract_id") != CONTRACT_ID:
        return None
    if document.get("protocol") != PROTOCOL_VERSION:
        return None
    pid = document.get("pid")
    if type(pid) is not int or pid <= 0:
        return None
    token = document.get("token")
    try:
        _token_bytes(token)
    except (TypeError, ValueError):
        return None
    if not pid_alive(pid):
        return None
    return document


def _claim_client(directory):
    claim = os.path.join(directory, CLIENT_CLAIM_NAME)
    for attempt in (0, 1):
        try:
            descriptor = os.open(
                claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_EXTRA, 0o600
            )
            break
        except FileExistsError as error:
            # A claim whose owner is dead (client SIGKILLed mid-request, exec
            # timed out) must be breakable, or it silently poisons every later
            # request into the classical fallback. The claim records its
            # owner's pid; break it iff that pid is gone.
            if attempt:
                raise MailboxError("mailbox already has an active client claim") from error
            owner = 0
            try:
                raw = read_bounded(claim, 64)
                owner = int(raw.decode("ascii", "replace").strip() or "0")
            except (MailboxError, OSError, ValueError):
                owner = 0
            if owner > 0 and owner != os.getpid() and pid_alive(owner):
                raise MailboxError(
                    f"mailbox already has an active client claim (pid={owner})"
                ) from error
            try:
                os.unlink(claim)
            except OSError:
                pass
        except OSError as error:
            raise MailboxError(f"cannot claim mailbox client: {error}") from error
    try:
        os.write(descriptor, b"%d\n" % os.getpid())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(directory)


def _decode_response(encoded, token_bytes):
    if len(encoded) < RESPONSE_HEADER.size:
        raise MailboxProtocolError("response is shorter than its fixed header")
    magic, frame_token, status_code, body_size = RESPONSE_HEADER.unpack_from(encoded)
    if magic != MAGIC or frame_token != token_bytes:
        raise MailboxProtocolError("response identity mismatch")
    if len(encoded) != RESPONSE_HEADER.size + body_size:
        raise MailboxProtocolError("response body length mismatch")
    body = encoded[RESPONSE_HEADER.size :]
    if status_code == STATUS_OK:
        if body_size > MAX_RESPONSE_BYTES:
            raise MailboxProtocolError("response exceeds its exact output limit")
        return body
    if status_code == STATUS_ERROR:
        if body_size > MAX_ERROR_BYTES:
            raise MailboxProtocolError("daemon error message exceeds its limit")
        raise CodecError(body.decode("utf-8", errors="replace"))
    raise MailboxProtocolError("response status is invalid")


def send_request(mailbox_dir, token, op, payload, timeout_s=420.0):
    """Publish one bounded request and await its bounded response.

    Claims ``client.claim`` (O_CREAT|O_EXCL), publishes ``request.ready``
    atomically, polls ``response.ready`` at 20 ms with a per-iteration daemon
    liveness recheck (zombie-aware) so mid-request death raises
    DaemonDiedError immediately instead of burning the timeout.  Raises
    CodecError on a STATUS_ERROR frame.  Cleans up claim/response files.
    """
    if not isinstance(op, bytes) or op not in VALID_OPS:
        raise ValueError("op must be one of b'C', b'D', b'V'")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    payload = bytes(payload)
    if len(payload) > MAX_REQUEST_BYTES:
        raise MailboxError(
            f"request is {len(payload)} bytes; maximum is {MAX_REQUEST_BYTES} bytes"
        )
    directory = mailbox_dir or default_mailbox_dir()
    token_bytes = _token_bytes(token)

    ready = read_ready(directory)
    if ready is None:
        raise DaemonDiedError("no live daemon (ready.json absent, invalid, or dead pid)")
    if ready["token"] != token_bytes.hex():
        raise MailboxError("provided token does not match the published ready.json")
    pid = ready["pid"]

    frame = (
        REQUEST_HEADER.pack(MAGIC, token_bytes, op, MAX_RESPONSE_BYTES, len(payload))
        + payload
    )
    request_path = os.path.join(directory, REQUEST_NAME)
    response_path = os.path.join(directory, RESPONSE_NAME)
    claim_path = os.path.join(directory, CLIENT_CLAIM_NAME)

    _claim_client(directory)
    try:
        try:  # never read a stale response left by a crashed predecessor
            os.unlink(response_path)
        except OSError:
            pass
        atomic_write(request_path, frame)
        deadline = time.monotonic() + timeout_s
        response_cap = RESPONSE_HEADER.size + MAX_RESPONSE_BYTES
        while True:
            try:
                os.lstat(response_path)
            except FileNotFoundError:
                if not pid_alive(pid):
                    # one last look: the daemon may have published then exited
                    try:
                        os.lstat(response_path)
                    except FileNotFoundError:
                        raise DaemonDiedError(
                            f"daemon pid {pid} exited without a response"
                        ) from None
                elif time.monotonic() >= deadline:
                    raise MailboxError(
                        f"daemon did not respond within {timeout_s:.1f} seconds"
                    )
                else:
                    time.sleep(POLL_INTERVAL_SECS)
                    continue
            encoded = read_bounded(response_path, response_cap)
            return _decode_response(encoded, token_bytes)
    finally:
        for leftover in (claim_path, response_path, request_path):
            try:
                os.unlink(leftover)
            except OSError:
                pass


__all__ = [
    "CONTRACT_ID",
    "PROTOCOL_VERSION",
    "DEFAULT_MAILBOX_DIR",
    "MAILBOX_DIR_ENV",
    "MAGIC",
    "REQUEST_HEADER",
    "RESPONSE_HEADER",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MAX_ERROR_BYTES",
    "STATUS_OK",
    "STATUS_ERROR",
    "OP_COMPRESS",
    "OP_DECOMPRESS",
    "OP_VERSION",
    "MailboxError",
    "MailboxProtocolError",
    "DaemonDiedError",
    "CodecError",
    "default_mailbox_dir",
    "atomic_write",
    "read_bounded",
    "pid_alive",
    "read_ready",
    "send_request",
]
