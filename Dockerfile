# syntax=docker/dockerfile:1.7
FROM mongo1001/glyph-rwkv-mem-v2@sha256:15518bf00777d22337b3e478407239fa37a0ad459644f455294d00af72d0338d

ARG SOURCE_DATE_EPOCH
ARG VCS_REVISION

LABEL org.opencontainers.image.title="Glyph RWKV alpha-0.8 hierarchical-256 B96" \
      org.opencontainers.image.description="SN117 lossless compression candidate" \
      org.opencontainers.image.source="https://github.com/luckydevstar/glyph-rwkv-alpha0p8-hier256-b96" \
      org.opencontainers.image.revision="$VCS_REVISION" \
      org.opencontainers.image.licenses="MIT"

# Vast's SSH validation launcher opens a reverse tunnel from inside the rented
# container.  The accepted UID114 base has sshd support injected by Vast but no
# ssh client binary, so include the minimal client package in the release image.
# Glyph evaluation still seals all network access before scored entrypoints run.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=0444 selected-model.pth /opt/model/rwkv7-g1g-1.5b-20260526-ctx8192.pth
COPY --chmod=0555 client_hier256.py /app/client_hier256.py
COPY --chmod=0555 warmup_hier256.py /app/warmup_hier256.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONHASHSEED=117
