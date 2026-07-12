#!/usr/bin/env bash
set -euo pipefail

: "${HF_MODEL_REPOSITORY:?HF_MODEL_REPOSITORY is required}"
: "${HF_MODEL_REVISION:?HF_MODEL_REVISION is required}"
: "${HF_MODEL_FILENAME:?HF_MODEL_FILENAME is required}"
: "${HF_MODEL_SHA256:?HF_MODEL_SHA256 is required}"

if [[ ! "$HF_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MODEL_REVISION must be a 40-character immutable Hugging Face commit" >&2
  exit 2
fi
if [[ ! "$HF_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "MODEL_SHA256 must be a lowercase SHA-256" >&2
  exit 2
fi
if [[ ! "$HF_MODEL_REPOSITORY" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "MODEL_REPOSITORY must be a Hugging Face owner/repository pair" >&2
  exit 2
fi
if [[ "$HF_MODEL_FILENAME" == */* || "$HF_MODEL_FILENAME" == .* ]]; then
  echo "MODEL_FILENAME must be a plain repository-root filename" >&2
  exit 2
fi

url="https://huggingface.co/${HF_MODEL_REPOSITORY}/resolve/${HF_MODEL_REVISION}/${HF_MODEL_FILENAME}?download=true"
partial="${HF_MODEL_FILENAME}.partial"

rm -f "$partial" "$HF_MODEL_FILENAME"
curl --fail --location --silent --show-error \
  --proto '=https' --tlsv1.2 \
  --retry 5 --retry-delay 5 --retry-all-errors \
  --output "$partial" "$url"

printf '%s  %s\n' "$HF_MODEL_SHA256" "$partial" | sha256sum --check --strict
mv "$partial" "$HF_MODEL_FILENAME"
chmod 0444 "$HF_MODEL_FILENAME"

actual_size="$(stat --format='%s' "$HF_MODEL_FILENAME")"
if [[ "$actual_size" != "3055596935" ]]; then
  echo "Unexpected model size: $actual_size" >&2
  exit 1
fi
