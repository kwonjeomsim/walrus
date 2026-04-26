#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Walrus {

// Run the embedded decision tree on a wasm module and emit indices of
// functions to JIT-compile. Returns true on success; on parse failure
// `outIndices` is left empty.
bool predictJITCandidates(const uint8_t* wasm, size_t size,
                          std::vector<uint32_t>& outIndices);

} // namespace Walrus
