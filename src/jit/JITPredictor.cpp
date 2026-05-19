/*
 * Copyright (c) 2022-present Samsung Electronics Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "JITPredictor.h"
#include "JITModelData.h"

#include "wabt/binary-reader.h"
#include "wabt/binary-reader-nop.h"
#include "wabt/feature.h"
#include "wabt/opcode.h"
#include "wabt/result.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace Walrus {

namespace {

// Feature indices below MUST match FEATURE_NAMES in
// src/decision_model/jit_decision_tree.py exactly. The trained tree refers
// to features by index, so any reordering here will silently corrupt
// predictions.
struct FuncFeature {
    int32_t index = 0;
    int32_t call_frequency = 0;
    int32_t body_size = 0;
    int32_t call_freq_x_body = 0;
    int32_t local_count = 0;
    int32_t call_indirect_count = 0;
    int32_t call_graph_depth = -1;
    int32_t branch_count = 0;
    // Caller-side loop signal — see FEATURE_NAMES docstring in
    // src/decision_model/jit_decision_tree.py for motivation.
    int32_t caller_in_loop_count = 0;
    int32_t max_caller_loop_depth = 0;
    int32_t caller_count = 0;
    int32_t is_leaf_function = 1;   // 1 until a call/call_indirect proves otherwise

    int32_t feature(int idx) const
    {
        switch (idx) {
        case 0:
            return call_frequency;
        case 1:
            return call_freq_x_body;
        case 2:
            return local_count;
        case 3:
            return call_indirect_count;
        case 4:
            return call_graph_depth;
        case 5:
            return branch_count;
        case 6:
            return caller_in_loop_count;
        case 7:
            return max_caller_loop_depth;
        case 8:
            return caller_count;
        case 9:
            return is_leaf_function;
        default:
            return 0;
        }
    }
};

class FeatureCollector : public wabt::BinaryReaderNop {
public:
    wabt::Result OnImportFunc(wabt::Index import_index,
                              nonstd::string_view module_name,
                              nonstd::string_view field_name,
                              wabt::Index func_index, wabt::Index sig_index) override
    {
        ensureFunc(func_index);
        if (m_isImport.size() <= func_index) {
            m_isImport.resize(func_index + 1, false);
        }
        m_isImport[func_index] = true;
        return wabt::Result::Ok;
    }

    wabt::Result OnFunction(wabt::Index index, wabt::Index sig_index) override
    {
        ensureFunc(index);
        return wabt::Result::Ok;
    }

    wabt::Result OnExport(wabt::Index index, wabt::ExternalKind kind,
                          wabt::Index item_index, nonstd::string_view name) override
    {
        if (kind == wabt::ExternalKind::Func) {
            m_exportedFuncs.push_back(static_cast<uint32_t>(item_index));
        }
        return wabt::Result::Ok;
    }

    wabt::Result BeginFunctionBody(wabt::Index index, wabt::Offset size) override
    {
        ensureFunc(index);
        m_curFunc = static_cast<int32_t>(index);
        // Reset per-body block tracking. The loop-depth stack uses a single
        // byte per nesting level: 1 = loop scope, 0 = block/if scope. Both
        // are popped on EndExpr, but only loops change loop_depth.
        m_blockStack.clear();
        m_curLoopDepth = 0;
        return wabt::Result::Ok;
    }

    wabt::Result OnLocalDecl(wabt::Index decl_index, wabt::Index count,
                             wabt::Type type) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].local_count += static_cast<int32_t>(count);
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnOpcode(wabt::Opcode opcode) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].body_size++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnLoopExpr(wabt::Type sig_type) override
    {
        // m_curLoopDepth is still needed to populate caller_in_loop_count /
        // max_caller_loop_depth on each OnCallExpr, so we keep tracking it.
        if (m_curFunc >= 0) {
            m_blockStack.push_back(1); // 1 marks a loop scope
            m_curLoopDepth++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBlockExpr(wabt::Type sig_type) override
    {
        if (m_curFunc >= 0) {
            m_blockStack.push_back(0);
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnIfExpr(wabt::Type sig_type) override
    {
        if (m_curFunc >= 0) {
            m_blockStack.push_back(0);
        }
        return wabt::Result::Ok;
    }

    // `else` re-enters the same structured scope opened by an earlier `if`;
    // no stack change. The matching `end` pops the if/else as one frame.

    wabt::Result OnEndExpr() override
    {
        if (m_curFunc >= 0 && !m_blockStack.empty()) {
            uint8_t kind = m_blockStack.back();
            m_blockStack.pop_back();
            if (kind == 1 && m_curLoopDepth > 0) {
                m_curLoopDepth--;
            }
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrExpr(wabt::Index depth) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].branch_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrIfExpr(wabt::Index depth) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].branch_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrTableExpr(wabt::Index num_targets,
                               wabt::Index* target_depths,
                               wabt::Index default_target_depth) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].branch_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnCallExpr(wabt::Index func_index) override
    {
        if (m_curFunc < 0) {
            return wabt::Result::Ok;
        }
        // Record the edge for later call_frequency / call_graph_depth /
        // caller_count aggregation, plus the loop_depth at this call site for
        // caller_in_loop_count / max_caller_loop_depth.
        m_callEdges.emplace_back(static_cast<uint32_t>(m_curFunc),
                                 static_cast<uint32_t>(func_index));
        m_callSiteDepths.emplace_back(static_cast<uint32_t>(func_index),
                                      m_curLoopDepth);
        m_funcs[m_curFunc].is_leaf_function = 0;
        return wabt::Result::Ok;
    }

    wabt::Result OnCallIndirectExpr(wabt::Index sig_index, wabt::Index table_index) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].call_indirect_count++;
            m_funcs[m_curFunc].is_leaf_function = 0;
        }
        return wabt::Result::Ok;
    }

    wabt::Result EndFunctionBody(wabt::Index index) override
    {
        m_curFunc = -1;
        m_blockStack.clear();
        m_curLoopDepth = 0;
        return wabt::Result::Ok;
    }

    void recordRestFeatures()
    {
        const size_t N = m_funcs.size();
        for (size_t i = 0; i < N; ++i) {
            m_funcs[i].index = static_cast<int32_t>(i);
        }
        // Direct-edge aggregates: call_frequency and the per-callee set of
        // distinct callers (caller_count). m_distinctCallers[i] holds the
        // set so we don't double-count multi-site callers.
        std::vector<std::unordered_set<uint32_t>> distinctCallers(N);
        for (const auto& e : m_callEdges) {
            if (e.second < N) {
                m_funcs[e.second].call_frequency++;
                distinctCallers[e.second].insert(e.first);
            }
            m_adj[e.first].push_back(e.second);
        }
        std::vector<std::pair<uint32_t, uint32_t>>().swap(m_callEdges);
        for (size_t i = 0; i < N; ++i) {
            m_funcs[i].caller_count = static_cast<int32_t>(distinctCallers[i].size());
        }

        // Per-callee in-loop call-site aggregates. m_callSiteDepths holds
        // (callee, loop_depth) for every direct call seen during body
        // traversal; here we tally how many of those sites were inside any
        // `loop` scope and track the deepest nesting.
        for (const auto& site : m_callSiteDepths) {
            uint32_t callee = site.first;
            int32_t depth = site.second;
            if (callee >= N) {
                continue;
            }
            if (depth > 0) {
                m_funcs[callee].caller_in_loop_count++;
            }
            if (depth > m_funcs[callee].max_caller_loop_depth) {
                m_funcs[callee].max_caller_loop_depth = depth;
            }
        }
        std::vector<std::pair<uint32_t, int32_t>>().swap(m_callSiteDepths);

        setCallGraphDepths(N);
        std::unordered_map<uint32_t, std::vector<uint32_t>>().swap(m_adj);
        for (auto& f : m_funcs) {
            f.call_freq_x_body = f.call_frequency * f.body_size;
        }
    }

    const std::vector<FuncFeature>& funcs() const { return m_funcs; }
    bool isImport(uint32_t idx) const
    {
        return idx < m_isImport.size() && m_isImport[idx];
    }

private:
    void ensureFunc(wabt::Index idx)
    {
        if (m_funcs.size() <= idx) {
            m_funcs.resize(idx + 1);
        }
    }

    void setCallGraphDepths(size_t N)
    {
        std::queue<uint32_t> q;
        for (uint32_t e : m_exportedFuncs) {
            if (e < N && m_funcs[e].call_graph_depth == -1) {
                m_funcs[e].call_graph_depth = 0;
                q.push(e);
            }
        }
        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();
            auto list = m_adj.find(u);
            if (list == m_adj.end()) {
                continue;
            }

            for (uint32_t v : list->second) {
                if (v < N && m_funcs[v].call_graph_depth == -1) {
                    m_funcs[v].call_graph_depth = m_funcs[u].call_graph_depth + 1;
                    q.push(v);
                }
            }
        }
    }

    std::vector<FuncFeature> m_funcs;
    std::vector<bool> m_isImport;
    std::vector<uint32_t> m_exportedFuncs;
    std::vector<std::pair<uint32_t, uint32_t>> m_callEdges;
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_adj;
    int32_t m_curFunc = -1;
    // Per-body block-scope stack: 1 = loop, 0 = block/if. The depth shadow
    // m_curLoopDepth is kept in sync so OnCallExpr can read it in O(1).
    std::vector<uint8_t> m_blockStack;
    int32_t m_curLoopDepth = 0;
    // (callee, loop_depth at the call site) for every direct call seen.
    std::vector<std::pair<uint32_t, int32_t>> m_callSiteDepths;
};

int evaluateTree(const FuncFeature& f)
{
    using namespace JITPredictorModel;
    int n = 0;
    for (int i = 0; i <= kNodeCount; ++i) {
        if (kNodeFeature[n] < 0) {
            return kNodeLeft[n]; // leaf node
        }
        int32_t v = f.feature(kNodeFeature[n]);
        n = (v <= kNodeThreshold[n]) ? kNodeLeft[n] : kNodeRight[n];
    }
    return 0;
}

} // namespace

bool predictJITCandidates(const uint8_t* wasm, size_t size,
                          std::vector<uint32_t>& outIndices)
{
    outIndices.clear();
    FeatureCollector reader;
    wabt::Features features;
    features.EnableAll();
    wabt::ReadBinaryOptions options(features, nullptr, false, false, false);

    if (wabt::Failed(wabt::ReadBinary(wasm, size, &reader, options))) {
        return false;
    }
    reader.recordRestFeatures();

    for (const auto& f : reader.funcs()) {
        const uint32_t idx = static_cast<uint32_t>(f.index);
        if (reader.isImport(idx)) {
            continue;
        }
        if (f.body_size == 0) {
            continue;
        }
        if (evaluateTree(f) == 1) {
            outIndices.push_back(idx);
        }
    }
    return true;
}

} // namespace Walrus
