// Byte parity of the existing grouped gather, including forced partial flushes and staging-slot reuse.
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Dev {
    uint8_t* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n)); }
    ~Dev() { cudaFree(p); }
};

void trial(int n, bool tail, bool unaligned, bool mixed) {
    const size_t half = unaligned ? 79 : 80, down = unaligned ? 111 : 112;
    const size_t up_off = 128, down_off = 256, tail_off = tail ? 384 : 0;
    const size_t blob_bytes = 416, zero = 512, gu_stride = half * 2;
    const size_t ngu = 16 * gu_stride + zero + 32, nd = 16 * down + zero + 32, nt = 16 * 16 + 32;
    Dev blobs(16 * blob_bytes + 16), stage(8 * blob_bytes);
    Dev refgu(ngu), refd(nd), reft(nt), gu(ngu), d(nd), t(nt);
    std::vector<uint8_t> src(16 * blob_bytes + 16);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) ((i * 37 + i / blob_bytes * 13) % 251);
    ck(cudaMemcpy(blobs.p, src.data(), src.size(), cudaMemcpyHostToDevice));
    for (auto* buffer : {&refgu, &refd, &reft, &gu, &d, &t})
        ck(cudaMemset(buffer->p, 0xa5, buffer == &refgu || buffer == &gu ? ngu :
                                      buffer == &refd || buffer == &d ? nd : nt));
    ck(cudaDeviceSynchronize());
    cudaStream_t compute, copy;
    ck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    ck(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    cudaEvent_t copied[8], used[8];
    for (int i = 0; i < 8; ++i) {
        ck(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
        ck(cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming));
    }
    for (int q = 0; q < n; ++q) {
        const uint8_t* b = blobs.p + q * blob_bytes + (unaligned ? 1 : 0);
        mmq::gather_native(b, b + up_off, half, b + down_off, down, refgu.p + q * gu_stride,
                           refd.p + q * down, compute, tail ? b + tail_off : nullptr,
                           tail ? reft.p + q * 16 : nullptr, q + 1 == n ? zero : 0);
    }
    mmq::GatherGroup group;
    int launches = 0, fallbacks = 0;
    auto flush = [&](size_t padding) {
        if (group.n <= group.first) return;
        if (!mmq::gather_native_group(group, up_off, half, down_off, down, gu.p, gu_stride,
                                      d.p, down, compute, tail_off, tail ? t.p : nullptr, padding)) {
            ++fallbacks;
            for (int q = group.first; q < group.n; ++q) {
                const auto* b = group.blob[q];
                mmq::gather_native(b, b + up_off, half, b + down_off, down, gu.p + q * gu_stride,
                                   d.p + q * down, compute, tail ? b + tail_off : nullptr,
                                   tail ? t.p + q * 16 : nullptr, q + 1 == group.n ? padding : 0);
                ++launches;
            }
        } else ++launches;
        group.first = group.n;
    };
    // More than eight staged experts exercises ring wrap; resident-only groups also force a nonzero first.
    int staged = 0;
    for (int q = 0; q < n; ++q) {
        const uint8_t* b = blobs.p + q * blob_bytes + (unaligned ? 1 : 0);
        const bool streamed = mixed && q % 5 != 0;
        const int slot = staged % 8;
        if (streamed) {
            if (staged >= 8) ck(cudaStreamWaitEvent(copy, used[slot], 0));
            ck(cudaMemcpyAsync(stage.p + slot * blob_bytes, b, blob_bytes, cudaMemcpyDeviceToDevice, copy));
            ck(cudaEventRecord(copied[slot], copy));
            ck(cudaStreamWaitEvent(compute, copied[slot], 0));
            b = stage.p + slot * blob_bytes;
        }
        group.blob[q] = b;
        group.n = q + 1;
        if (q + 1 == n) flush(zero);
        else if (streamed || (!mixed && q == 4)) flush(0);
        if (streamed) {
            ck(cudaEventRecord(used[slot], compute));
            ++staged;
        }
    }
    ck(cudaStreamSynchronize(compute));
    ck(cudaStreamSynchronize(copy));
    std::vector<uint8_t> cpu_gu(ngu, 0xa5), cpu_d(nd, 0xa5), cpu_t(nt, 0xa5);
    for (int q = 0; q < n; ++q) {
        const uint8_t* b = src.data() + q * blob_bytes + (unaligned ? 1 : 0);
        std::copy_n(b, half, cpu_gu.data() + q * gu_stride);
        std::copy_n(b + up_off, half, cpu_gu.data() + q * gu_stride + half);
        std::copy_n(b + down_off, down, cpu_d.data() + q * down);
        if (tail) std::copy_n(b + tail_off, 16, cpu_t.data() + q * 16);
    }
    std::fill_n(cpu_gu.data() + n * gu_stride, zero, 0);
    std::fill_n(cpu_d.data() + n * down, zero, 0);
    for (const auto& pair : {std::make_pair(&refgu, &gu), std::make_pair(&refd, &d), std::make_pair(&reft, &t)}) {
        size_t size = pair.first == &refgu ? ngu : pair.first == &refd ? nd : nt;
        std::vector<uint8_t> expected(size), actual(size);
        ck(cudaMemcpy(expected.data(), pair.first->p, size, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(actual.data(), pair.second->p, size, cudaMemcpyDeviceToHost));
        const auto& cpu = pair.first == &refgu ? cpu_gu : pair.first == &refd ? cpu_d : cpu_t;
        if (expected != actual || actual != cpu) throw std::runtime_error("gather byte/tail/sentinel parity failed");
    }
    if (unaligned != (fallbacks > 0)) throw std::runtime_error("alignment fallback not exercised");
    std::printf("PASS n=%d tail=%d unaligned=%d mixed=%d launches=%d fallback=%d\n",
                n, tail, unaligned, mixed, launches, fallbacks);
    for (int i = 0; i < 8; ++i) { ck(cudaEventDestroy(copied[i])); ck(cudaEventDestroy(used[i])); }
    ck(cudaStreamDestroy(compute));
    ck(cudaStreamDestroy(copy));
}

void invalid_groups() {
    Dev source(1024), gu(2048), d(2048);
    for (auto bounds : {std::make_pair(-1, 1), std::make_pair(0, 0), std::make_pair(2, 1),
                        std::make_pair(0, 17)}) {
        mmq::GatherGroup g;
        g.first = bounds.first; g.n = bounds.second;
        if (mmq::gather_native_group(g, 128, 80, 256, 112, gu.p, 160, d.p, 112, nullptr))
            throw std::runtime_error("invalid group bounds accepted");
    }
    mmq::GatherGroup g;
    g.n = 1; g.blob[0] = source.p;
    if (mmq::gather_native_group(g, 128, 80, 256, 112, gu.p + 1, 160, d.p, 112, nullptr) ||
        mmq::gather_native_group(g, 128, 80, 256, 112, gu.p, 161, d.p, 112, nullptr))
        throw std::runtime_error("unaligned destination/stride accepted");
    ck(cudaDeviceSynchronize());
}
} // namespace

int main() {
    try {
        int devices = 0;
        ck(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        for (int device = 0; device < devices; ++device) {
            ck(cudaSetDevice(device));
            invalid_groups();
            for (int n : {1, 5, 15, 16})
                for (bool tail : {false, true})
                    for (bool unaligned : {false, true})
                        for (bool mixed : {false, true}) trial(n, tail, unaligned, mixed);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
