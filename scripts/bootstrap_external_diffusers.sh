#!/usr/bin/env bash
set -euo pipefail

DIFFUSERS_REPO="${DIFFUSERS_REPO:-https://github.com/huggingface/diffusers.git}"
DIFFUSERS_REV="${DIFFUSERS_REV:-07a63e197e10860a470576cf4f610381b31a4dd7}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-external/diffusers}"

mkdir -p "$(dirname "${DIFFUSERS_DIR}")"

if [[ -d "${DIFFUSERS_DIR}/.git" ]]; then
  git -C "${DIFFUSERS_DIR}" fetch --all --tags
elif [[ -e "${DIFFUSERS_DIR}" ]]; then
  echo "${DIFFUSERS_DIR} exists but is not a git checkout." >&2
  echo "Move it away or set DIFFUSERS_DIR to another path." >&2
  exit 1
else
  git clone "${DIFFUSERS_REPO}" "${DIFFUSERS_DIR}"
fi

git -C "${DIFFUSERS_DIR}" checkout "${DIFFUSERS_REV}"
git -C "${DIFFUSERS_DIR}" status -sb
