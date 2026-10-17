#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "strata/program/gpu_ownership.hpp"

using namespace strata::program::gpu_ownership;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    const std::vector<std::pair<int32_t, int32_t>> full{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::string err;
    CHECK(validate_full_profile(2, 2, full, err));
    CHECK(err.empty());

    CHECK(!validate_full_profile(2, 2, {{0, 0}, {0, 1}, {1, 0}}, err));
    CHECK(err.find("contains 3 of 4") != std::string::npos);
    CHECK(!validate_full_profile(2, 2, {{0, 0}, {0, 1}, {1, 0}, {1, 0}}, err));
    CHECK(err.find("duplicate") != std::string::npos);
    CHECK(!validate_full_profile(2, 2, {{0, 0}, {0, 1}, {1, 0}, {2, 0}}, err));
    CHECK(err.find("out-of-range") != std::string::npos);

    uint64_t bytes = 0, slots = 0;
    const std::vector<uint64_t> blob_bytes{257, 512, 1024};
    CHECK(required_cache_bytes(blob_bytes, 3, 0, 2, bytes, slots, err));
    CHECK(bytes == 512 * 3 + 512 * 3);
    CHECK(slots == 6);
    CHECK(required_cache_bytes(blob_bytes, 3, 0, 2, bytes, slots, err, 64));
    CHECK(bytes == 512 * 3 + 768 * 3);
    CHECK(!required_cache_bytes(blob_bytes, 3, 1, 4, bytes, slots, err));
    CHECK(!required_cache_bytes({std::numeric_limits<uint64_t>::max()}, 1, 0, 1, bytes, slots, err));
    CHECK(!required_cache_bytes({std::numeric_limits<uint64_t>::max()}, 1, 0, 1, bytes, slots, err, 1));

    const std::vector<int32_t> owner{0, 1, 2, 3};
    auto slot_of = [&](int64_t layer, int64_t expert) { return owner[(size_t) (layer * 2 + expert)]; };
    int32_t missing_layer = -1, missing_expert = -1;
    CHECK(owns_layers(0, 2, 2, slot_of, missing_layer, missing_expert));
    auto partial_slot_of = [&](int64_t layer, int64_t expert) {
        return layer == 1 && expert == 1 ? -1 : slot_of(layer, expert);
    };
    CHECK(!owns_layers(0, 2, 2, partial_slot_of, missing_layer, missing_expert));
    CHECK(missing_layer == 1 && missing_expert == 1);

    if (failed == 0) std::printf("gpu_ownership_test: ok\n");
    return failed ? 1 : 0;
}
