"""Untimed warmup for the submitted B96 hierarchical RWKV codec."""

import sys

sys.path.insert(0, "/app")
import client_hier256 as client  # noqa: E402


if __name__ == "__main__":
    payload = b"the quick brown fox warms the hierarchical codec. " * 4000
    compressed = client.compress(payload)
    recovered = client.decompress(compressed)
    if recovered != payload:
        raise RuntimeError("hierarchical codec warmup did not round-trip")
    print("hierarchical B96 warmup complete", flush=True)

