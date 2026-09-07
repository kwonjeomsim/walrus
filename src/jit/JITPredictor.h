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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace Walrus {

// What Walrus works out while lowering a module and the wasm bytes do not
// carry: the operand-stack slots a function needs, and the size of the bytecode
// the JIT will actually consume. Indexed by function index; an empty vector
// means the caller had no module to read them from.
struct RuntimeFuncInfo {
    int32_t requiredStackSize = 0;
    int32_t byteCodeSize = 0;
};

bool predictJITCandidates(const uint8_t* wasm, size_t size, std::vector<uint32_t>& outIndices,
                          const std::vector<RuntimeFuncInfo>& runtime = {});

bool dumpJITFeatures(const uint8_t* wasm, size_t size, const char* path,
                     const std::vector<RuntimeFuncInfo>& runtime = {});

extern const char* g_jitCompileListPath;
extern int g_jitOptLevel;
constexpr int kJITOptLevelOff = -2;
int jitOptLevelCount();
// When set, every compiled function appends "<index> <bytes>" here. The cost
// side of the benefit-per-byte label comes from this, measured rather than
// estimated from bytecode size.
extern const char* g_jitCodeDumpPath;

bool loadJITCompileList(const char* path, std::vector<uint32_t>& outIndices);

} // namespace Walrus
