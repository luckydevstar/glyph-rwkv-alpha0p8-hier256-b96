#!/usr/bin/env bash
set -euo pipefail

image_name="${1:?usage: verify-anonymous-pull.sh IMAGE_NAME sha256:DIGEST}"
digest="${2:?usage: verify-anonymous-pull.sh IMAGE_NAME sha256:DIGEST}"

if [[ ! "$image_name" =~ ^ghcr\.io/[a-z0-9._-]+/[a-z0-9._/-]+$ ]]; then
  echo "Refusing unexpected GHCR image name: $image_name" >&2
  exit 2
fi
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest must be sha256 followed by 64 lowercase hex characters" >&2
  exit 2
fi

immutable_ref="${image_name}@${digest}"
anonymous_config="$(mktemp -d)"
trap 'rm -rf "$anonymous_config"' EXIT
printf '{"auths":{}}\n' > "$anonymous_config/config.json"

# DOCKER_CONFIG contains no auth entry and this script never runs docker login.
DOCKER_CONFIG="$anonymous_config" docker pull "$immutable_ref"

repo_digests="$(docker image inspect --format '{{join .RepoDigests "\n"}}' "$immutable_ref")"
if ! grep -Fxq "$immutable_ref" <<< "$repo_digests"; then
  echo "Pulled image did not report the requested immutable RepoDigest" >&2
  printf '%s\n' "$repo_digests" >&2
  exit 1
fi

echo "Anonymous digest pull verified: $immutable_ref"

