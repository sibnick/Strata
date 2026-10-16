#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::program::gpu_ownership {

inline bool validate_full_profile(int64_t n_layers, int64_t n_experts,
                                  const std::vector<std::pair<int32_t, int32_t>>& profile,
                                  std::string& err) {
    err.clear();
    if (n_layers <= 0 || n_experts <= 0 ||
        (uint64_t) n_layers > std::numeric_limits<size_t>::max() / (uint64_t) n_experts) {
        err = "invalid expert geometry";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_experts;
    if (profile.size() != count) {
        err = "profile contains " + std::to_string(profile.size()) + " of " + std::to_string(count) +
              " layer/expert pairs";
        return false;
    }
    std::vector<uint8_t> seen(count, 0);
    for (const auto& [layer, expert] : profile) {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
            err = "profile contains an out-of-range layer/expert pair";
            return false;
        }
        const size_t index = (size_t) layer * (size_t) n_experts + (size_t) expert;
        if (seen[index] != 0) {
            err = "profile contains a duplicate layer/expert pair";
            return false;
        }
        seen[index] = 1;
    }
    return true;
}

inline bool required_cache_bytes(const std::vector<uint64_t>& layer_blob_bytes, int64_t n_experts,
                                 int64_t first_layer, int64_t past_last_layer, uint64_t& bytes,
                                 uint64_t& slots, std::string& err) {
    bytes = 0;
    slots = 0;
    err.clear();
    if (n_experts <= 0 || first_layer < 0 || past_last_layer <= first_layer ||
        (uint64_t) past_last_layer > layer_blob_bytes.size()) {
        err = "invalid layer range or expert count";
        return false;
    }
    const uint64_t layers = (uint64_t) (past_last_layer - first_layer);
    if (layers > std::numeric_limits<uint64_t>::max() / (uint64_t) n_experts) {
        err = "expert slot count overflows";
        return false;
    }
    slots = layers * (uint64_t) n_experts;
    for (int64_t layer = first_layer; layer < past_last_layer; ++layer) {
        const uint64_t blob = layer_blob_bytes[(size_t) layer];
        if (blob == 0 || blob > std::numeric_limits<uint64_t>::max() - 255) {
            err = "invalid or overflowing expert blob size";
            return false;
        }
        const uint64_t aligned = (blob + 255) & ~uint64_t{255};
        if (aligned > std::numeric_limits<uint64_t>::max() / (uint64_t) n_experts) {
            err = "expert cache size overflows";
            return false;
        }
        const uint64_t layer_bytes = aligned * (uint64_t) n_experts;
        if (layer_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
            err = "expert cache size overflows";
            return false;
        }
        bytes += layer_bytes;
    }
    return true;
}

template <typename SlotOf>
bool owns_layers(int64_t first_layer, int64_t past_last_layer, int64_t n_experts, SlotOf&& slot_of,
                 int32_t& missing_layer, int32_t& missing_expert) {
    for (int64_t layer = first_layer; layer < past_last_layer; ++layer) {
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            if (slot_of(layer, expert) < 0) {
                missing_layer = (int32_t) layer;
                missing_expert = (int32_t) expert;
                return false;
            }
        }
    }
    return true;
}

}  // namespace strata::program::gpu_ownership
