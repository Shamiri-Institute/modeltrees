#!/usr/bin/env bash
set -euo pipefail
mkdir -p reference
cd reference
clone() { [ -d "$1" ] || git clone --depth 1 --branch "$2" "https://github.com/cran/$1" "$1"; }
clone partykit 1.3-0
clone strucchange 1.6-0
clone party 1.0-1
