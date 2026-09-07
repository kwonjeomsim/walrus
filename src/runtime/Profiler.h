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

#ifndef __WalrusProfiler__
#define __WalrusProfiler__

#include <cstdint>
#include <string>
#include <vector>

namespace Walrus {

class Module;

extern thread_local uint64_t* g_profileChildNs;

class Profiler {
public:
    static Profiler& instance();

    static bool enabled() { return s_enabled; }
    void enable(const std::string& outputPath);
    void registerModule(Module* module, const std::string& source);
    static uint64_t nowNs();
    void dump();

private:
    Profiler() {};

    static bool s_enabled;

    struct Entry {
        Module* module;
        std::string source;
    };

    std::string m_outputPath;
    std::vector<Entry> m_modules;
    bool m_dumped = false;
};

} // namespace Walrus

#endif // __WalrusProfiler__
#endif
