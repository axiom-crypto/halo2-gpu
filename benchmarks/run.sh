#!/bin/bash
set -e
NSYS=0
SAMPLY=0

REPO_ROOT=$(git rev-parse --show-toplevel)

while [[ $# -gt 0 ]]; do
  case $1 in
  --nsys)
    NSYS=1
    shift 1
    ;;
  --samply)
    SAMPLY=1
    shift 1
    ;;
  *)
    echo "Unknown argument: $1"
    exit 1
    ;;
  esac
done

export JEMALLOC_SYS_WITH_MALLOC_CONF="retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:10000,muzzy_decay_ms:10000,abort_conf:true"

RUSTFLAGS="-C force-frame-pointers=yes -Ctarget-cpu=native"
TOOLCHAIN="+nightly-2026-01-18"
echo "running: CUDA_LINEINFO=1 RUSTFLAGS=\"$RUSTFLAGS\" cargo \"$TOOLCHAIN\" build -p benchmarks --profile=profiling"

CUDA_LINEINFO=1 RUSTFLAGS="$RUSTFLAGS" cargo "$TOOLCHAIN" build -p benchmarks --profile=profiling
BIN="$REPO_ROOT/target/profiling/benchmarks"

# increase preallocation size to avoid expensive VPMM cuda api calls
MAX_MEM_SIZE=$((16 << 30))         # 16 GB
export VPMM_PAGE_SIZE=$((4 << 20)) # 4 MB
export VPMM_PAGES=$(($MAX_MEM_SIZE / $VPMM_PAGE_SIZE))

export OUTPUT_PATH="$REPO_ROOT/metrics.json"
if [ $NSYS -eq 1 ]; then
  nsys profile -f true -o halo-gpu.nsys-rep --gpu-metrics-devices=cuda-visible --cuda-memory-usage=true $BIN
elif [ $SAMPLY -eq 1 ]; then
  perf record -F 300 --call-graph=fp -g -o perf.data -- $BIN

  SAMPLY_PROFILE_PATH="$REPO_ROOT/samply_profile"
  mkdir -p $SAMPLY_PROFILE_PATH
  samply import perf.data --presymbolicate --save-only -o $SAMPLY_PROFILE_PATH/profile.json.gz
  echo "Saved profile: $SAMPLY_PROFILE_PATH/profile.json.gz"
else
  $BIN
fi
