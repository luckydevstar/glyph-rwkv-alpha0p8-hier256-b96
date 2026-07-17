#!/usr/bin/env bash
set -euo pipefail

echo "Disk before cleanup:"
df -h /

# The build context, base layers, and output layer coexist during the build.
# Remove hosted-runner SDKs that this workflow does not use.
sudo rm -rf \
  /usr/local/lib/android \
  /usr/share/dotnet \
  /opt/ghc \
  /usr/local/.ghcup \
  /usr/local/share/boost \
  /opt/hostedtoolcache/CodeQL
sudo apt-get clean
docker system prune --all --force --volumes || true

echo "Disk after cleanup:"
df -h /

available_kib="$(df --output=avail / | tail -1 | tr -d ' ')"
minimum_kib=$((20 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
  echo "Need at least 20 GiB free for a reliable build; found $((available_kib / 1024 / 1024)) GiB" >&2
  echo "Use a larger GitHub-hosted runner rather than risking an incomplete push." >&2
  exit 1
fi
