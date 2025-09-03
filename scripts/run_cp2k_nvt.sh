#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 COORD_PDB [CP2K_EXE]" >&2
  exit 1
fi

COORD_PDB=$1
CP2K_EXE=${2:-cp2k.popt}

cd /workspace/aimd/cp2k

sed "s#__COORD__#${COORD_PDB}#g" water_nvt.inp > water_nvt.run.inp

"${CP2K_EXE}" -i water_nvt.run.inp -o water_nvt.out | cat

echo "NVT finished. Outputs in /workspace/aimd/cp2k"

