#include "JITPredictor.h"
#include "jit_model_data.h"

#include "wabt/binary-reader.h"
#include "wabt/binary-reader-nop.h"
#include "wabt/feature.h"
#include "wabt/opcode.h"
#include "wabt/result.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

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
    int32_t loop_count = 0;
    int32_t branch_count = 0;

    float feature(int idx) const
    {
        switch (idx) {
        case 0:
            return float(call_frequency);
        case 1:
            return float(call_freq_x_body);
        case 2:
            return float(local_count);
        case 3:
            return float(call_indirect_count);
        case 4:
            return float(call_graph_depth);
        case 5:
            return float(loop_count);
        case 6:
            return float(branch_count);
        default:
            return 0.0f;
        }
    }
};

class FeatureCollector : public wabt::BinaryReaderNop {
public:
    wabt::Result OnImportFunc(wabt::Index /*import_index*/,
                              nonstd::string_view /*module_name*/,
                              nonstd::string_view /*field_name*/,
                              wabt::Index func_index, wabt::Index /*sig_index*/) override
    {
        ensureFunc(func_index);
        if (m_isImport.size() <= func_index) {
            m_isImport.resize(func_index + 1, false);
        }
        m_isImport[func_index] = true;
        return wabt::Result::Ok;
    }

    wabt::Result OnFunction(wabt::Index index, wabt::Index /*sig_index*/) override
    {
        ensureFunc(index);
        return wabt::Result::Ok;
    }

    wabt::Result OnExport(wabt::Index /*index*/, wabt::ExternalKind kind,
                          wabt::Index item_index, nonstd::string_view /*name*/) override
    {
        if (kind == wabt::ExternalKind::Func) {
            m_exportedFuncs.push_back(static_cast<uint32_t>(item_index));
        }
        return wabt::Result::Ok;
    }

    wabt::Result BeginFunctionBody(wabt::Index index, wabt::Offset /*size*/) override
    {
        ensureFunc(index);
        m_curFunc = static_cast<int32_t>(index);
        return wabt::Result::Ok;
    }

    wabt::Result OnLocalDecl(wabt::Index /*decl_index*/, wabt::Index count,
                             wabt::Type /*type*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].local_count += static_cast<int32_t>(count);
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnOpcode(wabt::Opcode /*opcode*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].body_size++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnLoopExpr(wabt::Type /*sig_type*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].loop_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrExpr(wabt::Index /*depth*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].branch_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrIfExpr(wabt::Index /*depth*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].branch_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result OnBrTableExpr(wabt::Index /*num_targets*/,
                               wabt::Index* /*target_depths*/,
                               wabt::Index /*default_target_depth*/) override
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
        m_callEdges.emplace_back(static_cast<uint32_t>(m_curFunc),
                                 static_cast<uint32_t>(func_index));
        return wabt::Result::Ok;
    }

    wabt::Result OnCallIndirectExpr(wabt::Index /*sig_index*/, wabt::Index /*table_index*/) override
    {
        if (m_curFunc >= 0) {
            m_funcs[m_curFunc].call_indirect_count++;
        }
        return wabt::Result::Ok;
    }

    wabt::Result EndFunctionBody(wabt::Index /*index*/) override
    {
        m_curFunc = -1;
        return wabt::Result::Ok;
    }

    void finalize()
    {
        for (size_t i = 0; i < m_funcs.size(); ++i) {
            m_funcs[i].index = static_cast<int32_t>(i);
        }
        for (const auto& e : m_callEdges) {
            if (e.second < m_funcs.size()) {
                m_funcs[e.second].call_frequency++;
            }
            m_adj[e.first].push_back(e.second);
        }
        bfsDepths();
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

    void bfsDepths()
    {
        std::queue<uint32_t> q;
        for (uint32_t e : m_exportedFuncs) {
            if (e < m_funcs.size() && m_funcs[e].call_graph_depth == -1) {
                m_funcs[e].call_graph_depth = 0;
                q.push(e);
            }
        }
        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();
            auto it = m_adj.find(u);
            if (it == m_adj.end()) {
                continue;
            }
            for (uint32_t v : it->second) {
                if (v < m_funcs.size() && m_funcs[v].call_graph_depth == -1) {
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
};

int evaluateTree(const FuncFeature& f)
{
    using namespace JITPredictorModel;
    int n = 0;
    // The tree is a finite DAG; node count bounds the loop, but we cap iterations
    // defensively in case of malformed data.
    for (int i = 0; i <= kNodeCount; ++i) {
        if (kNodeFeature[n] < 0) {
            return kNodeLeft[n];
        }
        float v = f.feature(kNodeFeature[n]);
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
    wabt::ReadBinaryOptions options(features, /*log_stream=*/nullptr,
                                    /*read_debug_names=*/false,
                                    /*stop_on_first_error=*/true,
                                    /*fail_on_custom_section_error=*/false);
    if (wabt::Failed(wabt::ReadBinary(wasm, size, &reader, options))) {
        return false;
    }
    reader.finalize();

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
