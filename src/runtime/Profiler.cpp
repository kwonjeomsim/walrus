/*
 * Copyright (c) 2024-present Samsung Electronics Co., Ltd
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

#if defined(WALRUS_PROFILER)

#include "Walrus.h"
#include "runtime/Profiler.h"
#include "runtime/Module.h"

#include <csignal>
#include <cstdio>
#include <ctime>
#include <unistd.h>

namespace Walrus {

bool Profiler::s_enabled = false;
thread_local uint64_t* g_profileChildNs = nullptr;

Profiler& Profiler::instance()
{
    static Profiler instance;
    return instance;
}

uint64_t Profiler::nowNs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

static void profilerSignalHandler(int sig)
{
    Profiler::instance().dump();
    _exit(128 + sig);
}

void Profiler::enable(const std::string& outputPath)
{
    m_outputPath = outputPath;
    s_enabled = true;
    signal(SIGTERM, profilerSignalHandler);
    signal(SIGINT, profilerSignalHandler);
}

void Profiler::registerModule(Module* module, const std::string& source)
{
    m_modules.push_back({ module, source });
}

// Export name of function `idx`, or empty if the function is not exported.
static std::string functionExportName(Module* module, uint32_t idx)
{
    for (auto exp : module->exports()) {
        if (exp->exportType() == ExportType::Function && exp->itemIndex() == idx) {
            return exp->name();
        }
    }
    return std::string();
}

void Profiler::dump()
{
    if (m_dumped || !s_enabled) {
        return;
    }
    m_dumped = true;

    FILE* out = fopen(m_outputPath.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "warning: --profile-output: cannot open \"%s\" for writing\n", m_outputPath.c_str());
        return;
    }

    fprintf(out, "==== Walrus per-function profile (raw) ====\n");
    for (auto& entry : m_modules) {
        fprintf(out, "# source = %s\n", entry.source.c_str());
    }
    fprintf(out, "%-32s %6s %6s %12s %16s\n",
            "function", "module", "index", "bytecode_sz", "self_time_ns");

    for (size_t m = 0; m < m_modules.size(); ++m) {
        Module* module = m_modules[m].module;
        const size_t n = module->numberOfFunctions();
        for (size_t i = 0; i < n; ++i) {
            ModuleFunction* fn = module->function(static_cast<uint32_t>(i));

            if (fn->byteCodeSize() == 0) {
                continue;
            }
            const std::string exportName = functionExportName(module, static_cast<uint32_t>(i));
            const std::string name = exportName.empty() ? std::string("func") : ("$" + exportName);
            fprintf(out, "%-32s %6zu %6zu %12zu %16llu\n",
                    name.c_str(), m, i,
                    fn->byteCodeSize(),
                    static_cast<unsigned long long>(fn->profileTimeNs()));
        }
    }

    fclose(out);
}

} // namespace Walrus
#endif
