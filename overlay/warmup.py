"""Untimed, fail-closed D256 primer layered over the sealed UID95 warmup.

The sealed warmup remains in ``warmup_uid95.py``.  On a deployed codec host,
this wrapper first launches or reuses that daemon and then submits the fixed
4 KiB primer ``bytes(range(256)) * 16``.  It accepts only an outer neural
method byte and an internally consistent D256 blob at the explicitly selected
lane.  A bare-host ``glyph-miner check`` has no ``/opt/codec`` runtime and
retains the sealed warmup's successful local-check behavior.

This artifact is stdlib-only apart from its two local artifact modules.
"""

import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mbclient
import warmup_uid95


PRIMER_PAYLOAD = bytes(range(256)) * 16
PRIMER_TIMEOUT_SECS = 180.0
METHOD_NEURAL = 0x10
SCHEME_D256 = 0xD256
ALLOWED_FINAL_LANES = (208, 192)
HEADER = struct.Struct("<QHH")
BODY_META = struct.Struct("<II")


class PrimerError(RuntimeError):
    """The deployed daemon did not satisfy the fixed primer contract."""


def _is_real_codec_host(server_path):
    """Distinguish deployed image execution from the bare local precheck."""
    return os.path.isdir("/opt/codec") or os.path.isfile(server_path)


def _selected_lane():
    fixed = os.environ.get("GLYPH_B_FIXED", "")
    allowed = os.environ.get("GLYPH_HIER256_ALLOWED_B", "")
    try:
        lane = int(fixed)
    except ValueError as error:
        raise PrimerError("GLYPH_B_FIXED is not a resolved integer lane") from error
    if lane not in ALLOWED_FINAL_LANES:
        raise PrimerError(
            "resolved lane %r is outside the final allowlist %r"
            % (lane, ALLOWED_FINAL_LANES)
        )
    if allowed != fixed:
        raise PrimerError(
            "GLYPH_HIER256_ALLOWED_B must exactly match GLYPH_B_FIXED"
        )
    return lane


def _validate_primer_blob(blob, expected_lane):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise PrimerError("primer response is not bytes-like")
    blob = bytes(blob)
    minimum = 1 + HEADER.size + BODY_META.size
    if len(blob) < minimum:
        raise PrimerError("primer response is shorter than the D256 header")
    if blob[0] != METHOD_NEURAL:
        raise PrimerError(
            "primer outer method is 0x%02x, expected neural 0x10" % blob[0]
        )
    token_count, lanes, scheme = HEADER.unpack_from(blob, 1)
    if token_count <= 0:
        raise PrimerError("primer response has no encoded tokens")
    if lanes != expected_lane:
        raise PrimerError(
            "primer D256 lane is %d, expected fixed lane %d"
            % (lanes, expected_lane)
        )
    if scheme != SCHEME_D256:
        raise PrimerError(
            "primer inner scheme is 0x%04x, expected 0xD256" % scheme
        )
    meta_offset = 1 + HEADER.size
    range_words, expected_crc32 = BODY_META.unpack_from(blob, meta_offset)
    if range_words <= 0:
        raise PrimerError("primer D256 body has no range words")
    body = blob[meta_offset + BODY_META.size :]
    if len(body) != range_words * 4:
        raise PrimerError("primer D256 body length does not match its word count")
    actual_crc32 = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc32 != expected_crc32:
        raise PrimerError("primer D256 body CRC32 mismatch")
    return {
        "token_count": token_count,
        "lanes": lanes,
        "scheme": scheme,
        "range_words": range_words,
    }


def main():
    server_path = warmup_uid95._server_path()
    real_host = _is_real_codec_host(server_path)

    # On the real image, reject an unresolved or unintended lane before paying
    # model-load cost.  The bare local check deliberately has no lane yet.
    if real_host:
        try:
            selected_lane = _selected_lane()
        except PrimerError as error:
            print("primed warmup: %s" % error, file=sys.stderr)
            return 1
    else:
        selected_lane = None

    status = warmup_uid95.main()
    if status not in (0, None):
        return int(status)

    if not real_host:
        print("primed warmup: local check mode; primer intentionally skipped")
        return 0

    ready = mbclient.read_ready()
    if ready is None:
        print(
            "primed warmup: deployed daemon has no validated readiness record",
            file=sys.stderr,
        )
        return 1
    try:
        response = mbclient.send_request(
            None,
            ready["token"],
            mbclient.OP_COMPRESS,
            PRIMER_PAYLOAD,
            timeout_s=PRIMER_TIMEOUT_SECS,
        )
        parsed = _validate_primer_blob(response, selected_lane)
    except (KeyError, mbclient.MailboxError, PrimerError) as error:
        print("primed warmup: primer failed closed: %s" % error, file=sys.stderr)
        return 1

    print(
        "primed warmup: neural D256 primer passed at B=%d (%d tokens, %d words)"
        % (parsed["lanes"], parsed["token_count"], parsed["range_words"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
