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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace Walrus {

namespace {

// Feature indices below MUST match FEATURE_NAMES in
// tools/jit-model/jit_decision_tree.py exactly.
struct FuncFeature {
    int32_t index = 0;
    int32_t call_site_count = 0;
    int32_t body_size = 0;
    int32_t call_site_work = 0;
    int32_t local_count = 0;
    int32_t call_indirect_count = 0;
    int32_t call_graph_depth = -1;
    int32_t branch_count = 0;
    int32_t caller_in_loop_count = 0;
    int32_t max_caller_loop_depth = 0;
    int32_t caller_count = 0;
    int32_t is_leaf_function = 1;
    int32_t max_own_loop_depth = 0;
    int32_t exec_count_sweep_log2 = 0;
    int32_t exec_work_rounds_log2 = 0;
    int32_t indirect_caller_count = 0;
    int32_t max_indirect_caller_loop_depth = 0;
    int32_t exec_count_rounds_log2 = 0;

    int32_t feature(int idx) const
    {
        switch (idx) {
        case 0:
            return call_site_work;
        case 1:
            return local_count;
        case 2:
            return call_indirect_count;
        case 3:
            return call_graph_depth;
        case 4:
            return branch_count;
        case 5:
            return caller_in_loop_count;
        case 6:
            return max_caller_loop_depth;
        case 7:
            return caller_count;
        case 8:
            return is_leaf_function;
        case 9:
            return max_own_loop_depth;
        case 10:
            return exec_count_sweep_log2;
        case 11:
            return exec_work_rounds_log2;
        case 12:
            return indirect_caller_count;
        case 13:
            return max_indirect_caller_loop_depth;
        case 14:
            return call_site_count;
        case 15:
            return exec_count_rounds_log2;
        default:
            return 0;
        }
    }
};

constexpr const char* kFeatureNames[] = {
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
};
constexpr int kFeatureNameCount = sizeof(kFeatureNames) / sizeof(kFeatureNames[0]);

// Every loop is assumed to run kDefaultTrip times, so exec_count_sweep_log2
// measures call-site loop-nest depth rather than real trip counts.
// Two estimates of how often a function runs. exec_count_sweep_log2 makes one
// pass in call-graph-depth order; exec_work_rounds_log2 iterates the
// propagation so cycles compound, then multiplies by the function's opcode
// count; exec_count_rounds_log2 is the same estimate without that product.
// All three are reported as fixedLog2.
constexpr int64_t kDefaultTrip = 10;

// Loop weight compounds as trip^depth, so a call seven loops deep is credited
// ten million executions. These let the constant and the depth it saturates at
// be swept against measured execution counts instead of assumed.
static int64_t defaultTrip()
{
    static const int64_t v = [] {
        const char* e = getenv("WALRUS_TRIP");
        long t = (e != nullptr) ? strtol(e, nullptr, 10) : kDefaultTrip;
        return static_cast<int64_t>(t > 0 ? t : kDefaultTrip);
    }();
    return v;
}

static int32_t maxLoopWeightDepth()
{
    static const int32_t v = [] {
        const char* e = getenv("WALRUS_TRIP_MAXDEPTH");
        long t = (e != nullptr) ? strtol(e, nullptr, 10) : 64;
        return static_cast<int32_t>(t > 0 ? t : 64);
    }();
    return v;
}
// Propagation rounds and the fixed-point scale shared by every *_log2 feature.
constexpr size_t kEstRounds = 24;
constexpr double kMaxEst = 1e18;
constexpr double kLog2Scale = 16.0;

inline int32_t fixedLog2(double v)
{
    // This if statement is for negative and NaN values.
    if (!(v > 0.0)) {
        return 0;
    }
    else if (!std::isfinite(v)) {
        return 30000;
    }
    const double l = std::log2(1.0 + v) * kLog2Scale;
    return static_cast<int32_t>(l > 30000.0 ? 30000.0 : (l < 0.0 ? 0.0 : l + 0.5));
}
constexpr int64_t kMaxStaticIters = 1000000000000000000LL;
inline int64_t satMul(int64_t a, int64_t b)
{
    if (a <= 0 || b <= 0) {
        return 0;
    }
    if (a > kMaxStaticIters / b) {
        return kMaxStaticIters;
    }
    return a * b;
}

static bool modelUses(int feature)
{
    for (int n = 0; n < JITPredictorModel::kNodeCount; ++n) {
        if (JITPredictorModel::kNodeFeature[n] == feature) {
            return true;
        }
    }
    return false;
}

// At load only the features the model tests are computed; dumping for training computes all.
struct FeatureNeeds {
    bool indirect, callerCount, sweep, rounds;
    static FeatureNeeds all() { return { true, true, true, true }; }
    static FeatureNeeds model()
    {
        return { modelUses(12) || modelUses(13), modelUses(7), modelUses(10),
                 modelUses(11) || modelUses(15) };
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
        setSig(func_index, sig_index);
        return wabt::Result::Ok;
    }

    wabt::Result OnFunction(wabt::Index index, wabt::Index sig_index) override
    {
        ensureFunc(index);
        setSig(index, sig_index);
        return wabt::Result::Ok;
    }

    wabt::Result OnExport(wabt::Index index, wabt::ExternalKind kind,
                          wabt::Index item_index, nonstd::string_view name) override
    {
        if (kind == wabt::ExternalKind::Func) {
            m_rootFuncs.push_back(static_cast<uint32_t>(item_index));
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnStartFunction(wabt::Index func_index) override
    {
        m_rootFuncs.push_back(static_cast<uint32_t>(func_index));
        return wabt::Result::Ok;
    }

    wabt::Result BeginElemSegment(wabt::Index index, wabt::Index table_index,
                                  uint8_t flags) override
    {
        m_inElemSegment = true;
        return wabt::Result::Ok;
    }

    wabt::Result EndElemSegment(wabt::Index index) override
    {
        m_inElemSegment = false;
        return wabt::Result::Ok;
    }

    // Referenced functions might be roots of the call graph. So record as roots conservatively.
    wabt::Result OnRefFuncExpr(wabt::Index func_index) override
    {
        m_rootFuncs.push_back(static_cast<uint32_t>(func_index));
        if (m_inElemSegment) {
            m_tableFuncs.push_back(static_cast<uint32_t>(func_index));
        }
        return wabt::Result::Ok;
    }

    wabt::Result BeginFunctionBody(wabt::Index index, wabt::Offset size) override
    {
        ensureFunc(index);
        m_curFunc = static_cast<int32_t>(index);
        m_blockStack.clear();
        m_loopFrames.clear();
        m_maxOwnLoopDepth = 0;
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
        if (m_curFunc >= 0) {
            LoopFrame frame;
            frame.siteStart = m_callSites.size();
            m_loopFrames.push_back(frame);
            m_blockStack.push_back(static_cast<int32_t>(m_loopFrames.size()));
            if (static_cast<int32_t>(m_loopFrames.size()) > m_maxOwnLoopDepth) {
                m_maxOwnLoopDepth = static_cast<int32_t>(m_loopFrames.size());
            }
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

    wabt::Result OnEndExpr() override
    {
        if (m_curFunc >= 0 && !m_blockStack.empty()) {
            int32_t entry = m_blockStack.back();
            m_blockStack.pop_back();
            if (entry > 0) {
                closeLoop();
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
        CallSite site;
        site.caller = static_cast<uint32_t>(m_curFunc);
        site.callee = static_cast<uint32_t>(func_index);
        site.loopDepth = static_cast<int32_t>(m_loopFrames.size());
        m_callSites.push_back(site);
        m_funcs[m_curFunc].is_leaf_function = 0;
        return wabt::Result::Ok;
    }

    wabt::Result OnCallIndirectExpr(wabt::Index sig_index, wabt::Index table_index) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].call_indirect_count++;
            m_funcs[m_curFunc].is_leaf_function = 0;
            IndirectSite site;
            site.sig = static_cast<uint32_t>(sig_index);
            site.loopDepth = static_cast<int32_t>(m_loopFrames.size());
            m_indirectSites.push_back(site);
        }
        return wabt::Result::Ok;
    }

    wabt::Result EndFunctionBody(wabt::Index index) override
    {
        if (m_curFunc >= 0) {
            while (!m_loopFrames.empty()) {
                closeLoop();
            }
            m_funcs[m_curFunc].max_own_loop_depth = m_maxOwnLoopDepth;
        }
        m_curFunc = -1;
        m_blockStack.clear();
        return wabt::Result::Ok;
    }

    // Everything that needs the whole call graph: call_site_count, caller_count,
    // caller_in_loop_count, max_caller_loop_depth, call_graph_depth, the two
    // exec_count_* estimates and the call_site_work / exec_work_* products.
    void recordRestFeatures(const FeatureNeeds& need)
    {
        const size_t N = m_funcs.size();
        for (size_t i = 0; i < N; ++i) {
            m_funcs[i].index = static_cast<int32_t>(i);
        }
        if (need.indirect) {
            recordIndirectCallers(N);
        }
        std::vector<std::unordered_set<uint32_t>> distinctCallers(need.callerCount ? N : 0);
        for (const auto& s : m_callSites) {
            if (s.callee < N) {
                m_funcs[s.callee].call_site_count++;
                if (need.callerCount) {
                    distinctCallers[s.callee].insert(s.caller);
                }
                if (s.loopDepth > 0) {
                    m_funcs[s.callee].caller_in_loop_count++;
                }
                if (s.loopDepth > m_funcs[s.callee].max_caller_loop_depth) {
                    m_funcs[s.callee].max_caller_loop_depth = s.loopDepth;
                }
            }
            m_adj[s.caller].push_back(s.callee);
        }
        if (need.callerCount) {
            for (size_t i = 0; i < N; ++i) {
                m_funcs[i].caller_count = static_cast<int32_t>(distinctCallers[i].size());
            }
        }

        setCallGraphDepths(N);
        if (need.sweep) {
            setExecCountSweep(N);
        }
        if (need.rounds) {
            setExecCountRounds(N);
        }
        std::unordered_map<uint32_t, std::vector<uint32_t>>().swap(m_adj);
        std::vector<CallSite>().swap(m_callSites);
        std::vector<IndirectSite>().swap(m_indirectSites);
        for (auto& f : m_funcs) {
            f.call_site_work = f.call_site_count * f.body_size;
        }
    }

    const std::vector<FuncFeature>& funcs() const { return m_funcs; }
    std::vector<FuncFeature>& mutableFuncs() { return m_funcs; }
    bool isImport(uint32_t idx) const
    {
        return idx < m_isImport.size() && m_isImport[idx];
    }

private:
    void setSig(wabt::Index idx, wabt::Index sig)
    {
        if (m_funcSig.size() <= idx) {
            m_funcSig.resize(idx + 1, UINT32_MAX);
        }
        m_funcSig[idx] = static_cast<uint32_t>(sig);
    }

    // static constexpr size_t kIndirectFanoutCap = 32;

    void recordIndirectCallers(size_t N)
    {
        if (m_indirectSites.empty() || m_tableFuncs.empty()) {
            return;
        }
        std::unordered_map<uint32_t, std::vector<uint32_t>> bySig;
        for (uint32_t f : m_tableFuncs) {
            if (f < m_funcSig.size() && m_funcSig[f] != UINT32_MAX) {
                bySig[m_funcSig[f]].push_back(f);
            }
        }
        for (auto& e : bySig) {
            std::sort(e.second.begin(), e.second.end());
            e.second.erase(std::unique(e.second.begin(), e.second.end()), e.second.end());
        }
        for (const IndirectSite& site : m_indirectSites) {
            auto it = bySig.find(site.sig);
            if (it == bySig.end()) {
                continue;
            }
            for (uint32_t callee : it->second) {
                if (callee >= N) {
                    continue;
                }
                m_funcs[callee].indirect_caller_count++;
                if (site.loopDepth > m_funcs[callee].max_indirect_caller_loop_depth) {
                    m_funcs[callee].max_indirect_caller_loop_depth = site.loopDepth;
                }
            }
        }
    }

    void ensureFunc(wabt::Index idx)
    {
        if (m_funcs.size() <= idx) {
            m_funcs.resize(idx + 1);
        }
    }

    void closeLoop()
    {
        const LoopFrame frame = m_loopFrames.back();
        m_loopFrames.pop_back();
        const int32_t depth = static_cast<int32_t>(m_loopFrames.size()) + 1;
        if (depth > maxLoopWeightDepth()) {
            return;
        }
        for (size_t i = frame.siteStart; i < m_callSites.size(); ++i) {
            m_callSites[i].loopWeight = satMul(m_callSites[i].loopWeight, defaultTrip());
        }
    }

    void setExecCountSweep(size_t N)
    {
        std::vector<int64_t> calls(N, 0);
        std::vector<uint32_t> order(N);
        for (size_t i = 0; i < N; ++i) {
            order[i] = static_cast<uint32_t>(i);
            if (m_funcs[i].call_graph_depth == 0) {
                calls[i] = 1;
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const int32_t da = m_funcs[a].call_graph_depth;
            const int32_t db = m_funcs[b].call_graph_depth;
            if (da != db) {
                return (da < 0 ? INT32_MAX : da) < (db < 0 ? INT32_MAX : db);
            }
            return a < b;
        });

        std::vector<std::vector<const CallSite*>> sitesByCaller(N);
        for (const auto& s : m_callSites) {
            if (s.caller < N) {
                sitesByCaller[s.caller].push_back(&s);
            }
        }
        for (uint32_t f : order) {
            const int64_t self = calls[f] > 0 ? calls[f] : 1;
            for (const CallSite* s : sitesByCaller[f]) {
                if (s->callee < N) {
                    calls[s->callee] = std::min(kMaxStaticIters,
                                                calls[s->callee] + satMul(self, s->loopWeight));
                }
            }
        }
        for (size_t i = 0; i < N; ++i) {
            m_funcs[i].exec_count_sweep_log2 = fixedLog2(static_cast<double>(calls[i]));
        }
    }

    // Bounded-round propagation from the entry points: after round k each
    // function holds the summed loop weight of every call path of length <= k
    // that reaches it. Crude next to an SCC condensation, but the compounding
    // it does around cycles turned out to be a useful "recursion is hot"
    // signal, and the clamp keeps it finite.
    void setExecCountRounds(size_t N)
    {
        std::vector<double> est(N, 0.0);
        std::vector<double> next(N, 0.0);
        auto seed = [&](std::vector<double>& v) {
            for (uint32_t r : m_rootFuncs) {
                if (r < N) {
                    v[r] = 1.0;
                }
            }
        };
        seed(est);
        const int rounds = static_cast<int>(std::min<size_t>(N, kEstRounds));
        for (int it = 0; it < rounds; ++it) {
            std::fill(next.begin(), next.end(), 0.0);
            seed(next);
            for (const auto& s : m_callSites) {
                if (s.caller < N && s.callee < N) {
                    next[s.callee] = std::min(kMaxEst,
                                              next[s.callee]
                                                  + est[s.caller] * static_cast<double>(s.loopWeight));
                }
            }
            est.swap(next);
        }
        for (size_t i = 0; i < N; ++i) {
            m_funcs[i].exec_count_rounds_log2 = fixedLog2(est[i]);
            m_funcs[i].exec_work_rounds_log2 =
                fixedLog2(est[i] * static_cast<double>(m_funcs[i].body_size));
        }
    }

    void setCallGraphDepths(size_t N)
    {
        std::queue<uint32_t> q;
        for (uint32_t e : m_rootFuncs) {
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

    struct CallSite {
        uint32_t caller = 0;
        uint32_t callee = 0;
        int32_t loopDepth = 0;
        int64_t loopWeight = 1;
    };
    struct LoopFrame {
        size_t siteStart = 0;  // first m_callSites index recorded inside
    };
    struct IndirectSite {
        uint32_t sig = 0;
        int32_t loopDepth = 0;
    };

    std::vector<FuncFeature> m_funcs;
    std::vector<bool> m_isImport;
    std::vector<uint32_t> m_rootFuncs;
    std::vector<CallSite> m_callSites;
    std::vector<IndirectSite> m_indirectSites;
    std::vector<uint32_t> m_funcSig;
    std::vector<uint32_t> m_tableFuncs;
    bool m_inElemSegment = false;
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_adj;
    int32_t m_curFunc = -1;
    // 0 = block/if, loopFrameIndex+1 = loop, so a relative branch depth
    // indexes straight into it. Its size is the current loop depth.
    std::vector<int32_t> m_blockStack;
    std::vector<LoopFrame> m_loopFrames;
    int32_t m_maxOwnLoopDepth = 0;
};

static int activeLevel()
{
    if (g_jitOptLevel == kJITOptLevelOff) {
        return -1;
    }
    if (g_jitOptLevel >= 0 && g_jitOptLevel < JITPredictorModel::kLevelCount) {
        return g_jitOptLevel;
    }
    return JITPredictorModel::kLevelCount / 2;
}

static int leafOfLevel(const FuncFeature& f, int level)
{
    using namespace JITPredictorModel;
    int n = kLevelRoot[level];
    for (int i = 0; i <= kNodeCount; ++i) {
        if (kNodeFeature[n] < 0) {
            return n;
        }
        int32_t v = f.feature(kNodeFeature[n]);
        n = (v <= kNodeThreshold[n]) ? kNodeLeft[n] : kNodeRight[n];
    }
    return -1;
}

int evaluateLeaf(const FuncFeature& f)
{
    const int level = activeLevel();
    return level < 0 ? -1 : leafOfLevel(f, level);
}

int evaluateTree(const FuncFeature& f)
{
    using namespace JITPredictorModel;
    const int level = activeLevel();
    if (level < 0) {
        return 0;
    }
    for (int i = 0; i <= level; ++i) {
        const int leaf = leafOfLevel(f, i);
        if (leaf >= 0 && kNodeLeft[leaf] == 1) {
            return 1;
        }
    }
    return 0;
}

bool writeFeatures(const std::vector<FuncFeature>& funcs, const char* path)
{
    FILE* out = fopen(path, "w");
    if (out == nullptr) {
        return false;
    }
    fprintf(out, "index");
    for (int i = 0; i < kFeatureNameCount; ++i) {
        fprintf(out, "\t%s", kFeatureNames[i]);
    }
    fprintf(out, "\n");
    for (const auto& f : funcs) {
        fprintf(out, "%d", f.index);
        for (int i = 0; i < kFeatureNameCount; ++i) {
            fprintf(out, "\t%d", f.feature(i));
        }
        fprintf(out, "\n");
    }
    fclose(out);
    return true;
}

bool collect(const uint8_t* wasm, size_t size, FeatureCollector& reader,
             const FeatureNeeds& need = FeatureNeeds::model())
{
    wabt::Features features;
    features.EnableAll();
    wabt::ReadBinaryOptions options(features, nullptr, false, false, false);

    wabt::ByteSpan data(wasm, size);
    if (wabt::Failed(wabt::ReadBinary(data, &reader, options))) {
        return false;
    }
    reader.recordRestFeatures(need);
    return true;
}

} // namespace

bool predictJITCandidates(const uint8_t* wasm, size_t size,
                          std::vector<uint32_t>& outIndices)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    outIndices.clear();
    FeatureCollector reader;
    // Repeating costs nothing in normal runs (g_jitPredictRepeats is 1) and lets
    // the measurement harness lift a millisecond-scale cost off the clock floor.
    for (int rep = 1; rep < g_jitPredictRepeats; ++rep) {
        FeatureCollector warm;
        if (!collect(wasm, size, warm)) {
            return false;
        }
        for (const auto& f : warm.funcs()) {
            if (!warm.isImport(static_cast<uint32_t>(f.index)) && f.body_size != 0) {
                evaluateTree(f);
            }
        }
    }
    if (!collect(wasm, size, reader)) {
        return false;
    }

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

    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_jitPredictTimeNs = static_cast<uint64_t>(t1.tv_sec - t0.tv_sec) * 1000000000ull
        + static_cast<uint64_t>(t1.tv_nsec) - static_cast<uint64_t>(t0.tv_nsec);
    return true;
}

const char* g_jitCompileListPath = nullptr;
int g_jitOptLevel = -1;

int jitOptLevelCount()
{
    return JITPredictorModel::kLevelCount;
}

const char* g_jitCodeDumpPath = nullptr;
uint64_t g_jitPredictTimeNs = 0;
int g_jitPredictRepeats = 1;

bool loadJITCompileList(const char* path, std::vector<uint32_t>& outIndices)
{
    outIndices.clear();
    FILE* f = fopen(path, "r");
    if (!f) {
        return false;
    }
    unsigned int idx;
    while (fscanf(f, "%u", &idx) == 1) {
        outIndices.push_back(idx);
    }
    fclose(f);
    return true;
}


bool dumpJITFeatures(const uint8_t* wasm, size_t size, const char* path)
{
    FeatureCollector reader;
    if (!collect(wasm, size, reader, FeatureNeeds::all())) {
        return false;
    }
    return writeFeatures(reader.funcs(), path);
}

} // namespace Walrus
