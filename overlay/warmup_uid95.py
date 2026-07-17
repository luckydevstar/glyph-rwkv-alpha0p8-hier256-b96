"""Warmup entrypoint (untimed) for the rwkv13-v2 codec — spec section 2.

Runs once per fresh container, before the single timed exec of that phase:
  (a) GPU drain-wait: the previous phase's container teardown can lag while
      holding VRAM and would OOM our model load (king_qwen compress.py:8-20).
  (b) Idempotency probe: a live healthy daemon (valid ready.json + live pid +
      version-hello round-trip) is reused, never doubled (two 13.3B models in
      VRAM would OOM the 24 GB gate).
  (c) Launch the image-side daemon detached (start_new_session + DEVNULL stdin
      + log-redirected stdout/stderr, close_fds): `docker exec` does not
      return until the exec's pipes hit EOF, so an inherited fd would hang
      warmup until its timeout (runner_docker.py:373-376).
  (d) Poll ready.json every 20 ms up to 1500 s (strictly inside the 1800 s
      manifest warmup timeout so failure is our clean error, not a validator
      kill), checking the child each iteration; on death surface the last
      4 KB of the daemon log.
  (e) Verify readiness identity: ready.json pid == Popen pid AND token ==
      our launch token, so a stale/foreign daemon can never be mistaken for
      the fresh one.

STDLIB-ONLY (precheck-scanned artifact file).  Local `glyph-miner check` runs
this on the bare host where /opt/codec does not exist: print a note, exit 0.
"""

import os
import secrets
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbclient

DEFAULT_SERVER_PATH = "/opt/codec/server.py"
def _log_default():
    if os.path.isdir("/scratch") and os.access("/scratch", os.W_OK):
        return "/scratch/rwkv-daemon.log"
    import tempfile
    return os.path.join(tempfile.gettempdir(), "rwkv-daemon.log")

DEFAULT_LOG_PATH = _log_default()
STARTUP_DEADLINE_SECS = 1500.0
HELLO_TIMEOUT_SECS = 10.0
POLL_INTERVAL_SECS = 0.02
GPU_DRAIN_LIMIT_MB = int(os.environ.get("GLYPH_GPU_DRAIN_LIMIT_MB", "2000"))
GPU_DRAIN_TIMEOUT_SECS = float(os.environ.get("GLYPH_GPU_DRAIN_TIMEOUT_SECS", "300"))


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


def _wait_gpu_drain(limit_mb=None, timeout_secs=None):
    """Busy-wait until GPU used memory < limit_mb.  Non-fatal on any error
    (nvidia-smi missing, no GPU, unparsable output): warmup must not flake."""
    limit_mb = GPU_DRAIN_LIMIT_MB if limit_mb is None else limit_mb
    timeout_secs = GPU_DRAIN_TIMEOUT_SECS if timeout_secs is None else timeout_secs
    deadline = time.monotonic() + timeout_secs
    while True:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30.0,
            )
        except (OSError, subprocess.SubprocessError):
            return
        if probe.returncode != 0:
            return
        used = []
        for line in probe.stdout.decode("ascii", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                used.append(int(line))
            except ValueError:
                pass
        if not used:
            return
        if max(used) < limit_mb:
            return
        if time.monotonic() >= deadline:
            print(
                "warmup: GPU still holds %d MB after %.0fs drain wait; proceeding"
                % (max(used), timeout_secs),
                file=sys.stderr,
            )
            return
        time.sleep(2.0)


def _log_tail(max_bytes=4096):
    """Last max_bytes of the daemon log for error surfacing."""
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


def _launch_server(token):
    """Spec section 1 launch recipe: detached daemon, fds never inherited."""
    log_fd = os.open(
        _log_path(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    try:
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        return subprocess.Popen(
            [sys.executable, _server_path(), "--token", token],
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=environment,
        )
    finally:
        os.close(log_fd)


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


def main():
    started = time.monotonic()

    # (a) idempotency probe FIRST — before the GPU drain-wait. A healthy live
    # daemon is what HOLDS the GPU memory; probing after the drain-wait would
    # burn the full 300 s timeout against our own daemon before discovering
    # nothing needs launching.
    ready = mbclient.read_ready()  # validated + zombie-aware pid liveness
    if ready is not None:
        try:
            mbclient.send_request(
                None,
                ready["token"],
                mbclient.OP_VERSION,
                b"",
                timeout_s=HELLO_TIMEOUT_SECS,
            )
        except mbclient.MailboxError:
            pass  # stale or unhealthy: fall through to a fresh launch
        else:
            print("warmup: reusing live daemon pid=%d (hello ok)" % ready["pid"])
            return 0

    # (b) GPU drain-wait: only relevant before an actual launch (previous
    # phase's container teardown can lag holding VRAM).
    _wait_gpu_drain()

    server = _server_path()
    if not os.path.isfile(server):
        # glyph-miner check --local-path: bare host, no image, no daemon.
        print(
            "warmup: no daemon server at %s (local check mode); nothing to do"
            % server
        )
        return 0

    # (c) pre-launch guard (spec §2 step 2 / failure mode #7): a live daemon
    # whose hello FAILED is unhealthy but still owns 13.3 GB of VRAM — launching
    # a second server next to it would stack two models (26.6 GB > 24 GB) and
    # OOM both. Refuse loudly instead; also clear stale mailbox litter so the
    # new daemon's own prepare_mailbox cannot trip on it minutes into its load.
    if ready is not None and mbclient.pid_alive(ready["pid"]):
        print(
            "warmup: refusing to launch: live daemon pid=%d holds the mailbox "
            "but failed the version hello; kill it or wait for idle-exit"
            % ready["pid"],
            file=sys.stderr,
        )
        return 1
    mbdir = mbclient.default_mailbox_dir()
    for name in ("ready.json", "request.ready", "request.claimed",
                 "client.claim", "response.ready"):
        try:
            os.unlink(os.path.join(mbdir, name))
        except OSError:
            pass
    if os.path.isdir(mbdir):
        for name in os.listdir(mbdir):
            if name.startswith(".tmp-"):
                try:
                    os.unlink(os.path.join(mbdir, name))
                except OSError:
                    pass

    # launch the daemon; the server re-runs its own prepare_mailbox (allowlist
    # cleanup, loud refusal of a live pid) before publishing readiness.
    token = secrets.token_hex(16)
    try:
        process = _launch_server(token)
    except OSError as error:
        print("warmup: cannot launch daemon: %s" % error, file=sys.stderr)
        return 1

    # (d) poll ready.json every 20 ms up to the startup deadline.
    deadline = time.monotonic() + STARTUP_DEADLINE_SECS
    while True:
        code = process.poll()
        if code is not None:
            print(
                "warmup: daemon exited with status %s before readiness" % code,
                file=sys.stderr,
            )
            tail = _log_tail()
            if tail:
                print("warmup: last 4KB of daemon log:\n%s" % tail, file=sys.stderr)
            return 1
        ready = mbclient.read_ready()
        if ready is not None:
            # (e) readiness identity: pid and token must both match.
            if ready["pid"] != process.pid or ready["token"] != token:
                print(
                    "warmup: readiness identity mismatch (pid/token); "
                    "stopping launched daemon",
                    file=sys.stderr,
                )
                _stop_process(process)
                return 1
            print(
                "warmup: daemon ready pid=%d in %.1fs"
                % (ready["pid"], time.monotonic() - started)
            )
            return 0
        if time.monotonic() >= deadline:
            print(
                "warmup: daemon not ready within %.0fs; stopping it"
                % STARTUP_DEADLINE_SECS,
                file=sys.stderr,
            )
            tail = _log_tail()
            if tail:
                print("warmup: last 4KB of daemon log:\n%s" % tail, file=sys.stderr)
            _stop_process(process)
            return 1
        time.sleep(POLL_INTERVAL_SECS)


if __name__ == "__main__":
    raise SystemExit(main())
