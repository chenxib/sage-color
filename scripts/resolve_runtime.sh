#!/usr/bin/env bash

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="python"
fi

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  ACCELERATE_BIN="accelerate"
fi
