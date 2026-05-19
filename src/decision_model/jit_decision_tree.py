"""Static feature extraction for JIT-priority training.

This module is a pure library — `export_model.py` imports `FEATURE_NAMES`
and `extract_features_from_wasm()` from here to build the training
dataset. The trained tree is then emitted to `src/jit/JITModelData.h`.

Features (per WebAssembly function, ordered as in `FEATURE_NAMES`).
Originally 12 features; `loop_count` and `is_in_elem_table` were dropped
after ablation (the trained tree never split on them under the current
data + cv-selected hyperparameters), leaving 10:

    - call_frequency         : 정적 직접 호출 사이트 수
    - call_freq_x_body       : call_frequency * body_size
    - local_count            : 로컬 변수 개수
    - call_indirect_count    : 함수 본문 내 `call_indirect` 출현 수
    - call_graph_depth       : exported entry 기준 BFS 거리 (도달 불가는 -1)
    - branch_count           : 함수 본문의 `br` / `br_if` / `br_table` 합

    -- Caller-side loop signal (microbench coverage) --
    - caller_in_loop_count   : 이 함수를 호출하는 직접 사이트들 중 호출자
                               함수 안의 loop 내부에 있는 사이트의 수.
                               "이 함수가 핫 루프에서 불리는가"를 직접 표현.
    - max_caller_loop_depth  : 직접 호출 사이트들 중 최대 loop 중첩 깊이.
    - caller_count           : 서로 다른 직접 호출자(함수 인덱스) 수.
    - is_leaf_function       : 함수 본문에 `call`/`call_indirect`가 전혀
                               없으면 1, 있으면 0. atomic/math wrapper류 검출용.

Note: `mem_access_count` and `body_size` were dropped after ablation
showed they were either redundant (`mem_access_count`) or too aggressive
in pruning hot mibench functions (`body_size`). `body_size` is still
computed internally to feed `call_freq_x_body`, but is not exposed to
the tree.

The order MUST match `FuncFeature::feature(int idx)` in
`src/jit/JITPredictor.cpp`; the trained tree references features by
index, so any reordering here will silently corrupt predictions.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List

FEATURE_NAMES = [
    "call_frequency",
    "call_freq_x_body",
    "local_count",
    "call_indirect_count",
    "call_graph_depth",
    "branch_count",
    # Caller-side loop signal for microbench coverage.
    "caller_in_loop_count",
    "max_caller_loop_depth",
    "caller_count",
    "is_leaf_function",
]


FUNC_HEADER_RE = re.compile(
    r"\s*\(func \(;(\d+);\) \(type (\d+)\)(.*)", re.DOTALL
)
EXPORT_FUNC_RE = re.compile(r'\(export "[^"]+" \(func (\d+)\)\)')
CALL_RE = re.compile(r"\bcall (\d+)\b")
CALL_INDIRECT_RE = re.compile(r"\bcall_indirect\b")
LOCAL_LINE_RE = re.compile(r"\(local\s+([^\)]*)\)")
BRANCH_RE = re.compile(r"\b(?:br_if|br_table|br)\b")


@dataclass
class FuncFeature:
    index: int
    call_frequency: int
    call_freq_x_body: int
    local_count: int
    call_indirect_count: int
    call_graph_depth: int
    branch_count: int
    caller_in_loop_count: int
    max_caller_loop_depth: int
    caller_count: int
    is_leaf_function: int

    def to_vector(self) -> List[int]:
        return [
            self.call_frequency,
            self.call_freq_x_body,
            self.local_count,
            self.call_indirect_count,
            self.call_graph_depth,
            self.branch_count,
            self.caller_in_loop_count,
            self.max_caller_loop_depth,
            self.caller_count,
            self.is_leaf_function,
        ]


def _iter_func_blocks(text: str):
    marker = "\n  (func (;"
    i = 0
    while True:
        idx = text.find(marker, i)
        if idx == -1:
            return
        start = idx + 1
        depth = 0
        j = start
        while j < len(text):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield start, j + 1
        i = j + 1


def _compute_depths(exports: Iterable[int], adj: dict, all_indices: Iterable[int]) -> dict:
    """BFS from exported entry points. Unreachable nodes get depth = -1."""
    depth = {i: -1 for i in all_indices}
    q = deque()
    for e in exports:
        if e in depth and depth[e] == -1:
            depth[e] = 0
            q.append(e)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if depth.get(v, -1) == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


# Token-level scanner for one function body. Yields (kind, payload, loop_depth)
# at each interesting token, where `kind` is one of:
#   "call"          : direct call,    payload = callee_idx
#   "call_indirect" : indirect call,  payload = None
# loop_depth is the count of currently-open `loop` scopes at the token.
# Handles both stacked WAT control form (`loop` ... `end`) and folded form
# (`(loop ...)`); wasm2wat default output uses stacked form for instructions
# and folded form for the wrapping `(func ...)` itself.
_BODY_TOKEN_RE = re.compile(
    r"\((loop|block|if)\b"            # 1: folded control opener
    r"|\((\w+)"                        # 2: any other folded paren opener
    r"|\)"                             # close paren
    r"|\bloop\b|\bblock\b|\bif\b|\bend\b|\belse\b"  # stacked control
    r"|\bcall\s+(\d+)\b"               # 3: direct call (callee)
    r"|\bcall_indirect\b"
)


def _scan_body(body_text: str):
    paren_stack: List[str] = []     # "loop" or "other" per open paren
    stacked_stack: List[str] = []   # "loop"/"block"/"if" per stacked control
    loop_depth = 0
    for m in _BODY_TOKEN_RE.finditer(body_text):
        full = m.group(0)
        ctrl_open = m.group(1)
        other_open = m.group(2)
        callee = m.group(3)

        if ctrl_open is not None:
            paren_stack.append(ctrl_open)
            if ctrl_open == "loop":
                loop_depth += 1
        elif other_open is not None:
            paren_stack.append("other")
        elif full == ")":
            if paren_stack:
                kind = paren_stack.pop()
                if kind == "loop":
                    loop_depth -= 1
        elif full == "loop":
            stacked_stack.append("loop")
            loop_depth += 1
        elif full in ("block", "if"):
            stacked_stack.append(full)
        elif full == "end":
            if stacked_stack:
                kind = stacked_stack.pop()
                if kind == "loop":
                    loop_depth -= 1
        elif full == "else":
            pass
        elif callee is not None:
            yield "call", int(callee), loop_depth
        elif full == "call_indirect":
            yield "call_indirect", None, loop_depth


def _parse_wat(text: str) -> List[FuncFeature]:
    exports = [int(m.group(1)) for m in EXPORT_FUNC_RE.finditer(text)]

    parsed = []
    inbound_by_idx: dict = {}
    adj: dict = {}
    all_indices: list = []

    # Aggregates filled by the body scanner.
    caller_in_loop_by_idx: dict = {}    # callee -> count of in-loop direct sites
    max_caller_loop_by_idx: dict = {}   # callee -> max loop_depth across sites
    direct_callers_by_idx: dict = {}    # callee -> set of distinct caller indices
    is_leaf_by_idx: dict = {}           # caller -> 1 if no call/call_indirect

    for s, e in _iter_func_blocks(text):
        block = text[s:e]
        header = FUNC_HEADER_RE.match(block)
        if not header:
            continue
        func_idx = int(header.group(1))
        rest = header.group(3)

        locals_count = sum(
            len(m.group(1).split()) for m in LOCAL_LINE_RE.finditer(rest)
        )

        body_size = sum(1 for ln in rest.splitlines()[1:] if ln.strip())
        call_indirects = len(CALL_INDIRECT_RE.findall(rest))
        branches = len(BRANCH_RE.findall(rest))

        # Walk the body once to compute caller-side loop signal and adjacency.
        callees: List[int] = []
        has_any_call = (call_indirects > 0)
        for kind, payload, loop_depth in _scan_body(rest):
            if kind == "call":
                callees.append(payload)
                has_any_call = True
                if loop_depth > 0:
                    caller_in_loop_by_idx[payload] = caller_in_loop_by_idx.get(payload, 0) + 1
                if loop_depth > max_caller_loop_by_idx.get(payload, 0):
                    max_caller_loop_by_idx[payload] = loop_depth
                direct_callers_by_idx.setdefault(payload, set()).add(func_idx)
            else:  # call_indirect
                has_any_call = True

        for c in callees:
            inbound_by_idx[c] = inbound_by_idx.get(c, 0) + 1
        adj.setdefault(func_idx, set()).update(callees)
        all_indices.append(func_idx)
        is_leaf_by_idx[func_idx] = 0 if has_any_call else 1

        parsed.append(
            dict(
                index=func_idx,
                local_count=locals_count,
                _body_size=body_size,
                call_indirect_count=call_indirects,
                branch_count=branches,
            )
        )

    # Include callees that have no body (imports) so the BFS terminates cleanly.
    for idx in set(inbound_by_idx).union(*[adj.get(i, set()) for i in adj]):
        if idx not in all_indices:
            all_indices.append(idx)

    depths = _compute_depths(exports, adj, all_indices)

    funcs: List[FuncFeature] = []
    for p in parsed:
        call_freq = inbound_by_idx.get(p["index"], 0)
        funcs.append(
            FuncFeature(
                index=p["index"],
                call_frequency=call_freq,
                call_freq_x_body=call_freq * p["_body_size"],
                local_count=p["local_count"],
                call_indirect_count=p["call_indirect_count"],
                call_graph_depth=depths.get(p["index"], -1),
                branch_count=p["branch_count"],
                caller_in_loop_count=caller_in_loop_by_idx.get(p["index"], 0),
                max_caller_loop_depth=max_caller_loop_by_idx.get(p["index"], 0),
                caller_count=len(direct_callers_by_idx.get(p["index"], set())),
                is_leaf_function=is_leaf_by_idx.get(p["index"], 1),
            )
        )
    return funcs


def extract_features_from_wasm(path: str) -> List[FuncFeature]:
    """Run wasm2wat on a .wasm (or read directly if .wat) and extract features."""
    if path.endswith(".wat"):
        with open(path) as f:
            text = f.read()
    else:
        with tempfile.NamedTemporaryFile("r", suffix=".wat", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # --no-debug-names: emit numeric `(func (;N;) ...)` even when the
            # module carries a name section (Rust/clang-with-names wasm).
            # Without it the WAT uses `(func $sym ...)` and the index regex
            # below matches nothing → 0 features. C/wasi-sdk wasm is already
            # numeric, so this flag is a no-op there.
            # --enable-all: accept every wasm proposal (e.g. the legacy
            # exception-handling / Tag section emitted by clang's setjmp
            # lowering, as in the Lua benchmark) so feature extraction never
            # fails to parse. Benches that don't use a feature are unaffected.
            subprocess.run(
                ["wasm2wat", "--enable-all", "--no-debug-names", path, "-o", tmp_path],
                check=True, capture_output=True,
            )
            with open(tmp_path) as f:
                text = f.read()
        finally:
            os.unlink(tmp_path)
    return _parse_wat(text)
