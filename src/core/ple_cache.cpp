// src/core/ple_cache.cpp - VRAM-resident n-gram embedding cache for PLE.
#include "strata/core/ple_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

PleVramCache::PleVramCache() = default;

PleVramCache::~PleVramCache() {
    close();
}

bool PleVramCache::open(double gib, std::string& err) {
    close();
    if (gib <= 0.0) {
        err = "PleVramCache::open: cache size must be positive";
        return false;
    }

    const uint64_t slot_bytes = (uint64_t) strata::kernels::NG_N_EMBD * sizeof(float);  // 10240 bytes
    const uint64_t target_bytes = (uint64_t) (gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t total_slots = target_bytes / slot_bytes;

    if (total_slots < WAYS) {
        err = "PleVramCache::open: cache size too small for set associativity";
        return false;
    }

    // Align to multiple of WAYS
    total_slots = (total_slots / WAYS) * WAYS;
    num_sets_ = total_slots / WAYS;
    capacity_slots_ = total_slots;
    gib_ = (double) (capacity_slots_ * slot_bytes) / (1024.0 * 1024.0 * 1024.0);

    const size_t alloc_bytes = (size_t) capacity_slots_ * (size_t) slot_bytes;
    const cudaError_t ce = cudaMalloc((void**) &d_data_, alloc_bytes);
    if (ce != cudaSuccess) {
        err = std::string("PleVramCache: cudaMalloc (") + std::to_string(alloc_bytes / (1024 * 1024)) +
              " MiB) failed: " + cudaGetErrorString(ce);
        d_data_ = nullptr;
        capacity_slots_ = 0;
        num_sets_ = 0;
        return false;
    }

    ways_.resize((size_t) (num_sets_ * WAYS));
    for (uint64_t s = 0; s < num_sets_; ++s) {
        for (uint32_t w = 0; w < WAYS; ++w) {
            const size_t idx = (size_t) s * WAYS + w;
            ways_[idx].key = NgramKey{};
            ways_[idx].slot_idx = (uint32_t) idx;
        }
    }

    next_way_.assign((size_t) num_sets_, 0);
    requests_ = 0;
    hits_ = 0;
    misses_ = 0;
    return true;
}

void PleVramCache::close() {
    if (d_data_ != nullptr) {
        (void) cudaFree(d_data_);
        d_data_ = nullptr;
    }
    capacity_slots_ = 0;
    num_sets_ = 0;
    gib_ = 0.0;
    ways_.clear();
    ways_.shrink_to_fit();
    next_way_.clear();
    next_way_.shrink_to_fit();
}

int32_t PleVramCache::lookup(const NgramKey& key) noexcept {
    if (num_sets_ == 0 || key.empty()) return -1;
    ++requests_;

    const uint64_t set = hash_ngram_key(key) % num_sets_;
    const size_t base = (size_t) set * WAYS;

    for (uint32_t w = 0; w < WAYS; ++w) {
        if (ways_[base + w].key == key) {
            ++hits_;
            return (int32_t) ways_[base + w].slot_idx;
        }
    }

    ++misses_;
    return -1;
}

uint32_t PleVramCache::insert(const NgramKey& key) noexcept {
    if (num_sets_ == 0 || key.empty()) return 0;

    const uint64_t set = hash_ngram_key(key) % num_sets_;
    const size_t base = (size_t) set * WAYS;

    // Check if key is already present or if an empty slot is available
    for (uint32_t w = 0; w < WAYS; ++w) {
        if (ways_[base + w].key == key) {
            return ways_[base + w].slot_idx;
        }
    }
    for (uint32_t w = 0; w < WAYS; ++w) {
        if (ways_[base + w].key.empty()) {
            ways_[base + w].key = key;
            return ways_[base + w].slot_idx;
        }
    }

    // Round-robin clock eviction in the set
    const uint32_t victim_w = next_way_[(size_t) set];
    next_way_[(size_t) set] = (uint8_t) ((victim_w + 1) % WAYS);

    ways_[base + victim_w].key = key;
    return ways_[base + victim_w].slot_idx;
}

std::string PleVramCache::report() const {
    if (!is_enabled()) return {};
    char buf[256];
    const double hit_pct = requests_ > 0 ? 100.0 * (double) hits_ / (double) requests_ : 0.0;
    const double pcie_saved_mb = (double) (hits_ * (uint64_t) strata::kernels::NG_N_EMBD * sizeof(float)) / 1e6;
    std::snprintf(buf, sizeof buf,
                  "ple vram cache: %llu requests, %llu hits (%.1f%%), %llu misses, %.1f MB PCIe traffic saved (%.2f GiB VRAM, %llu slots)",
                  (unsigned long long) requests_, (unsigned long long) hits_, hit_pct,
                  (unsigned long long) misses_, pcie_saved_mb, gib_, (unsigned long long) capacity_slots_);
    return buf;
}

}  // namespace strata::core
