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

// Every feature is read out of the module bytes, so both entry points take the
// wasm binary and nothing else.
bool predictJITCandidates(const uint8_t* wasm, size_t size, std::vector<uint32_t>& outIndices);

bool dumpJITFeatures(const uint8_t* wasm, size_t size, const char* path);

extern const char* g_jitCompileListPath;
extern int g_jitOptLevel;
constexpr int kJITOptLevelOff = -2;
int jitOptLevelCount();
extern const char* g_jitCodeDumpPath;
// Nanoseconds spent in predictJITCandidates, so --jit-stats can report what the
// static predictor costs. Set once per module load; zero in every other mode.
extern uint64_t g_jitPredictTimeNs;
// How many times to repeat the extraction before reporting the time. One-shot
// runs sit near the clock's noise floor on small modules, so the harness can
// ask for many and divide.
extern int g_jitPredictRepeats;

bool loadJITCompileList(const char* path, std::vector<uint32_t>& outIndices);

} // namespace Walrus
