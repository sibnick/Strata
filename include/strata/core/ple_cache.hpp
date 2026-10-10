#pragma once

#include "strata/kernels/ngram.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Trigram key identifying a unique n-gram embedding.
struct NgramKey {
    int32_t token = -1;
    int32_t prev0 = -1;
    int32_t prev1 = -1;

    constexpr bool operator==(const NgramKey& o) const noexcept {
        return token == o.token && prev0 == o.prev0 && prev1 == o.prev1;
    }
    constexpr bool empty() const noexcept {
        return token == -1 && prev0 == -1 && prev1 == -1;
    }
};

/// 64-bit splitmix-style hash for NgramKey.
inline uint64_t hash_ngram_key(const NgramKey& k) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    h = (h ^ (uint32_t) k.token) * 0x100000001b3ull;
    h = (h ^ (uint32_t) k.prev0) * 0x100000001b3ull;
    h = (h ^ (uint32_t) k.prev1) * 0x100000001b3ull;
    h ^= (h >> 32);
    return h * 0x9e3779b97f4a7c15ull;
}

/// VRAM-resident cache for 2560-float n-gram embeddings (Layer 1 PLE).
///
/// Keeps the most frequently and recently used n-gram embeddings in GPU memory,
/// eliminating SSD reads, CPU dequantization, and PCIe D2H transfer for cache hits.
class PleVramCache {
public:
    PleVramCache();
    ~PleVramCache();
    PleVramCache(const PleVramCache&) = delete;
    PleVramCache& operator=(const PleVramCache&) = delete;

    /// Allocate `gib` gigabytes of VRAM for the cache.
    /// Each slot is 2560 floats (10 KiB).
    bool open(double gib, std::string& err);
    void close();

    bool is_enabled() const noexcept { return d_data_ != nullptr && capacity_slots_ > 0; }
    uint64_t capacity_slots() const noexcept { return capacity_slots_; }
    double gib() const noexcept { return gib_; }

    /// Look up an n-gram in the cache. Returns slot index [0..capacity_slots) on hit, or -1 on miss.
    int32_t lookup(const NgramKey& key) noexcept;

    /// Select a slot to insert the n-gram (empty slot or round-robin victim) and register the key.
    /// Returns the assigned slot index.
    uint32_t insert(const NgramKey& key) noexcept;

    /// Device pointer to a slot's 2560-float buffer in VRAM.
    float* slot_dev_ptr(uint32_t slot) noexcept {
        return d_data_ + (size_t) slot * strata::kernels::NG_N_EMBD;
    }
    const float* slot_dev_ptr(uint32_t slot) const noexcept {
        return d_data_ + (size_t) slot * strata::kernels::NG_N_EMBD;
    }

    /// Temporarily pin a slot so it is never evicted while in flight in the current window.
    void pin(uint32_t slot) noexcept {
        if (slot < ways_.size()) ways_[slot].pinned = 1;
    }
    void unpin(uint32_t slot) noexcept {
        if (slot < ways_.size()) ways_[slot].pinned = 0;
    }

    uint64_t requests() const noexcept { return requests_; }
    uint64_t hits() const noexcept { return hits_; }
    uint64_t misses() const noexcept { return misses_; }

    /// One-line summary of cache hit rate and PCIe traffic eliminated.
    std::string report() const;

private:
    static constexpr uint32_t WAYS = 8;

    struct CacheWay {
        NgramKey key;
        uint32_t slot_idx = 0xFFFFFFFFu;
        uint8_t referenced = 0;
        uint8_t pinned = 0;
    };

    float* d_data_ = nullptr;
    uint64_t capacity_slots_ = 0;
    uint64_t num_sets_ = 0;
    double gib_ = 0.0;

    std::vector<CacheWay> ways_;      // num_sets_ * WAYS
    std::vector<uint8_t> next_way_;   // round-robin pointer per set

    uint64_t requests_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
};

}  // namespace strata::core
