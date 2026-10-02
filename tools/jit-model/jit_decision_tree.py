"""Static feature extraction for JIT-priority training.

The sixteen features the decision tree may see, in the order it indexes them:

    - call_site_work                 : how many call sites target this function,
                                       times its opcode count
    - local_count                    : number of local variables
    - call_indirect_count            : `call_indirect` occurrences in the body
    - call_graph_depth               : BFS distance from an entry point (-1 if
                                       unreachable)
    - branch_count                   : `br` + `br_if` + `br_table` in the body
    - caller_in_loop_count           : direct call sites that sit inside a loop
    - max_caller_loop_depth          : deepest loop nesting among those sites
    - caller_count                   : number of distinct direct callers
    - is_leaf_function               : 1 if the body has no call at all
    - max_own_loop_depth             : deepest loop nesting inside this function
    - exec_count_sweep_log2          : estimated execution count, one sweep in
                                       call-graph-depth order
    - exec_work_rounds_log2          : estimated execution count from iterated
                                       propagation, times opcode count
    - indirect_caller_count          : `call_indirect` sites whose signature
                                       admits this function
    - max_indirect_caller_loop_depth : deepest loop nesting among those sites
    - call_site_count                : direct call sites that target this function
    - exec_count_rounds_log2         : exec_work_rounds_log2 without the opcode count

All *_log2 values are round(16 * log2(1 + x)). required_stack_size and walrus_bytecode_size come from
the runtime rather than the binary and are likewise not fed to the tree.

The values are not computed here. `walrus --dump-jit-features` runs the same
extractor the runtime uses at load time, and this module reads its output, so
training and inference cannot drift apart.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURE_NAMES = [
    "call_site_work",
    "local_count",
    "call_indirect_count",
    "call_graph_depth",
    "branch_count",
    "caller_in_loop_count",
    "max_caller_loop_depth",
    "caller_count",
    "is_leaf_function",
    "max_own_loop_depth",
    "exec_count_sweep_log2",
    "exec_work_rounds_log2",
    "indirect_caller_count",
    "max_indirect_caller_loop_depth",
    "call_site_count",
    "exec_count_rounds_log2",
]

@dataclass
class FuncFeature:
    index: int
    call_site_work: int
    local_count: int
    call_indirect_count: int
    call_graph_depth: int
    branch_count: int
    caller_in_loop_count: int
    max_caller_loop_depth: int
    caller_count: int
    is_leaf_function: int
    max_own_loop_depth: int
    exec_count_sweep_log2: int
    exec_work_rounds_log2: int
    indirect_caller_count: int
    max_indirect_caller_loop_depth: int
    call_site_count: int
    exec_count_rounds_log2: int

    def to_vector(self) -> List[int]:
        return [
            self.call_site_work,
            self.local_count,
            self.call_indirect_count,
            self.call_graph_depth,
            self.branch_count,
            self.caller_in_loop_count,
            self.max_caller_loop_depth,
            self.caller_count,
            self.is_leaf_function,
            self.max_own_loop_depth,
            self.exec_count_sweep_log2,
            self.exec_work_rounds_log2,
            self.indirect_caller_count,
            self.max_indirect_caller_loop_depth,
            self.call_site_count,
            self.exec_count_rounds_log2,
        ]

def _walrus_binary() -> str:
    path = os.environ.get("WALRUS_BIN") or str(_REPO_ROOT / "out/walrus")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"walrus binary not found at {path}; build it or set WALRUS_BIN"
        )
    return path


def extract_features_from_wasm(path: str) -> List[FuncFeature]:
    """Read the feature table straight out of walrus's own extractor."""
    with tempfile.NamedTemporaryFile("r", suffix=".tsv", delete=False) as tmp:
        tsv = tmp.name
    try:
        subprocess.run([_walrus_binary(), "--dump-jit-features", tsv, path],
                       check=True, capture_output=True)
        with open(tsv) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            assert [f.name for f in dataclasses.fields(FuncFeature)][1:] == FEATURE_NAMES
            if header[1:] != FEATURE_NAMES:
                raise RuntimeError(
                    f"feature table from walrus does not match FEATURE_NAMES:\n"
                    f"  walrus: {header[1:]}\n  python: {FEATURE_NAMES}"
                )
            funcs = []
            for line in fh:
                cols = [int(c) for c in line.rstrip("\n").split("\t")]
                funcs.append(FuncFeature(cols[0], *cols[1:]))
    finally:
        os.unlink(tsv)
    return funcs
