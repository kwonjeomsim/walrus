#!/bin/bash
# Retrain the v6 hybrid JIT model and write src/jit/JITModelData.h (or $1).
# The runtime binary no longer dumps features, so this needs a research build
# from tier-up-v6 (WALRUS_BIN). The feature order below is the training column
# order; changing it breaks ties differently and gives a different tree.
set -e
cd "$(dirname "$0")/../.."
OUT=${1:-src/jit/JITModelData.h}
: "${WALRUS_BIN:=out/release/x64/walrus}"
export WALRUS_BIN
FS=local_count,branch_count,call_indirect_count,is_leaf_function,max_own_loop_depth,call_site_count,caller_in_loop_count,max_caller_loop_depth,call_graph_depth,call_site_work
.venv/bin/python tools/jit-model/export_levels.py \
    --level-dir O1=profile/train6_t4.0 \
    --level-dir O2=profile/train6_t2.0 \
    --level-dir O3=profile/train6_t1.0 \
    --features "$FS" --bench-weight equal --max-depth 6 --min-samples-leaf 20 \
    --ccp-alpha 0.0003 --hot-weight 2 --out "$OUT"
