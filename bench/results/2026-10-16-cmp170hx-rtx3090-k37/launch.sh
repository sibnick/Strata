#!/bin/sh
# Exact successful-run engine invocation. No polling guard: see guarded-launch.py.
set -eu
unset CUDA_VISIBLE_DEVICES
export STRATA_UNBUFFERED_LOAD=1
export STRATA_RESIDENT_HEADROOM_GIB=4
export STRATA_PIN_RESERVE_GIB=4
export STRATA_TRACE=1
export STRATA_SPLIT_RING=96
export STRATA_STAGE_PIN=0
exec build/strata \
  --pack build/pack/ud-q4-k-xl \
  --native /home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --gpu 1,0 --layer-split 37 --trim-stage-weights \
  --max-context 262144 --kv int8 \
  --mtp /home/nikolay/projects/Strata-data/mtp/rt \
  --spec 2 --prefill 256 --no-prefill-borrow \
  --resident-budget-gib 0.001 --expert-profile data/expert-profile.bin \
  --ple-io direct --ple-row-cache 1024 \
  --vram-reserve-mib 1024 --vram-reserve-later-mib 1024 \
  --pool-workers 1 --serve --prompt-cache 0 \
  --conversation-cache-min-free-mib 4096 <<'INPUT'
GEN 4 1,2,3
QUIT
INPUT
