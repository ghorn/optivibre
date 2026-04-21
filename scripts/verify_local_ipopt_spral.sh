#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/use_local_ipopt_spral_env.sh" >/dev/null

pkg-config --modversion ipopt
pkg-config --cflags --libs ipopt
ipopt --print-options | rg 'linear_solver|spral'
