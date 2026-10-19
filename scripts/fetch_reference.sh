#!/usr/bin/env bash
set -euo pipefail
mkdir -p reference
cd reference
clone() { [ -d "$1" ] || git clone --depth 1 --branch "$2" "https://github.com/cran/$1" "$1"; }
clone partykit 1.2-26
clone strucchange 1.5-4
clone party 1.0-1
