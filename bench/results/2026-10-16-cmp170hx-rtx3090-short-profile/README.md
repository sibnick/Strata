# Short CUDA stage diagnosis and MMQ capacity blocker

2026-10-16 follow-up to the static K37 long-context evidence. **No input exceeded 2,048 tokens and no generation cap exceeded 32.** No split, context capacity, residency rule, reserve, runtime source, driver or system setting was changed.

## Finding and decision

Short phase profiles identify the CMP prefix's routed MoE dequantization and products as the dominant cost, rather than GDN recurrence, short-context attention, or a strictly serialized handoff. The existing opt-in K-quant MMQ path is a relevant candidate because it avoids the per-expert FP16 dequantization/products, but **it failed the unchanged memory guard** on this placement. It must not be recommended here.

MMQ retained all 24,576 owned experts and reached READY, then the first short warmup sampled **971 MiB free on CMP**, below the 1,024 MiB guard. The harness terminated only its own process (`-15`). It completed no MMQ answer, no MMQ throughput comparison and no MMQ profile. The subsequent profile arm was not launched. No reserve was lowered or run retried.

This is a measured workspace-feasibility blocker, not a claim that MMQ is faster or incorrect. Continuing with the same placement would require at least 53 MiB of additional CMP free memory just to reach the guard, **plus practical working margin**. The bounded lifetime audit below found no existing, straightforward memory-recovery switch that safely supplies that budget without changing this configuration's guarantees. No speculative runtime optimization was implemented.

## Build and exact settings

The old `/tmp/strata-gpu-owned-cuda/strata` and its cache were no longer present when this follow-up began. The recorded shared model/pack/MTP artifacts remained available. A fresh local CUDA build was made:

```sh
cmake -S . -B build/cuda-short-profile \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  '-DCMAKE_CUDA_ARCHITECTURES=80;86' \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF
cmake --build build/cuda-short-profile --target strata -j 4
```

Both commands passed. CUDA 13.4.92, Release, sm80+sm86; tested source parent `d6602822ecf698191cb3671cea732439d627eab1`. `git diff 0327ba27 HEAD -- src include CMakeLists.txt` was empty: runtime/build source is unchanged from the original candidate. This does not imply binary identity. The rebuilt executable SHA-256 is `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214`; K-quant library SHA-256 is `f0476d2db4fc631d77b2923f35dacdc7dcbb80462e583356c85d773768a12ef4`. `build-identity.json` records commands, compiler, source comparison and cache hash. HIP/SYCL were not built or tested; no backend source was edited.

`profile-short.py` copies the exact engine argv and environment policy from `../2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context/main-results.json`, replacing only the binary path. Thus CMP owns `[0,37)`, RTX `[37,48)`, `--gpu 1,0 --gpu-owned-experts --trim-stage-weights --max-context 262144 --kv int8 --prefill 256 --no-prefill-borrow --mmap-experts`, direct PLE, no prompt-cache reuse, MTP on. Full argv, artifact identities and environment overrides are in each arm's `results.json`. Large model files were not rehashed; original manifest identities were checked by size/mtime, with fresh hashes only for small artifacts.

Arms differ only by these explicit opt-ins:

| Arm | Additional environment | Result |
|---|---|---|
| `control` | none | warmup + 9 requests passed |
| `profile` | `STRATA_PREFILL_TIMING=1` | warmup + 9 requests passed |
| `mmq-control` | `STRATA_MMQ_KQ=1` | stopped by VRAM guard during warmup |

The MMQ library's successful load is recorded in the raw log; this was not a silent fallback to the FP16 path. Native expert format pairs are `(12,7)`, `(12,8)`, `(13,8)` (Q4_K/Q5_1, Q4_K/Q8_0, Q5_K/Q8_0). The loader and MMQ format checks are in `src/prefill/mmq_kq_loader.cpp:58-114` and `src/prefill/moe_mmq.cu:192-254`.

## Protocol, warmup and useful work

The actual pack tokenizer generated varied field-note prompts with three independently expected codes (`ORIGIN: 714`, `CENTER: 928`, `FINAL: 365`) at beginning/middle/end. `prompts/` records text, input IDs, hashes, offsets and expected answers. Each arm reused one resident process for one 1,977-token warmup, followed by three ordered repetitions each of the 502-, 994- and 1,977-token prompts. All successful answers completed within the 32-token cap; `DONE` confirms zero reused input tokens and full fresh input counts.

The terminal input token is handled by the engine's token path: the batched prompt stage processed 501 / 993 / 1,976 tokens, respectively. Stage attribution uses exactly one multi-chunk prefix timing record (`n > 256`) and suffix records of at most 256 tokens; `summarize.py` asserts matching prefix/suffix token totals.

Startup was **371.399 s control**, **244.481 s profile**, **233.895 s MMQ**. These are separate serial starts, not controlled storage/cache comparisons. Each successful process amortized startup across ten short requests. The baseline source-read counters include cold/warm storage effects; no global cache drops or model modifications were performed.

## Timing-disabled baseline and profiler overhead

Warmup is excluded. Values are engine prefill times, not startup-inclusive wall times:

| Input tokens | Control repetitions (ms) | Control median (ms) | Profile median (ms) | Profiler increase |
|---:|---|---:|---:|---:|
| 502 | 1599.8, 1545.8, 1530.6 | 1545.8 | 2007.0 | 29.84% |
| 994 | 2969.7, 2876.0, 2881.9 | 2881.9 | 3803.3 | 31.97% |
| 1977 | 5483.5, 5499.9, 5551.5 | 5499.9 | 7314.5 | 32.99% |

The existing CUDA-event profiler is materially intrusive, particularly with thousands of per-expert marks. Use it for attribution, **not** as baseline throughput or as a denominator for a claimed speedup. There is no completed timing-disabled MMQ request to compare against control. These arms are not randomized/interleaved; warmup/repetitions do not eliminate all clock, thermal and cache effects.

Median request wall time from submitting GEN to receiving DONE, including generation, was **1.759 / 3.118 / 5.741 seconds** for control, versus **2.214 / 4.032 / 7.546 seconds** with profiling (502 / 994 / 1,977 tokens). The profiler increases even this end-to-end measurement. Exact wall times, decode times and MTP accepted/offered pairs are in `results.json` and `summary.json`; control's three large requests accepted/offered 9/13, 9/13 and 9/12 drafts, matching profile. First-token arrival time was not separately timestamped in this short harness; no independent TTFT is claimed.

For the 1,977-token profiled requests, median per-stage summaries were:

| Timeline component | CMP prefix ms | RTX suffix ms, summed over 8 chunks |
|---|---:|---:|
| Recorded GPU timeline | 7105 | 1556 |
| Host wall inside stage calls | 7114 | 1625 |
| Expert dequant | 3388 | 586 |
| Expert gate/up + down | 3134 | 771 |
| GDN recurrence | 57 | 8 |
| GDN projection/conv/output, excluding recurrence | 113 | 48 |
| QSA attention | 28 | 10 |
| QSA projection/indexer/select | 38 | 23 |
| Host grouping label | 95 | 58 |
| Embed/steps label, including inter-chunk stream gaps | 95 | rounded 0 |

The CMP dequant label is about 47.7% and the two expert product labels about 44.1% of its recorded timeline. These are stream-interval attributions: CUDA event gaps can include waits/launch starvation; they do not establish achieved FLOPs or pure kernel execution times. Phase labels are rounded to integer ms. Component medians need not sum to the median total. The source's `host staging` counter is cumulative and is deliberately not summed.

The CMP owns 28 GDN + 9 QSA layers; the RTX owns 8 GDN + 3 QSA (`include/strata/core/geometry.hpp:75-77`). The recorded stage GPU timeline ratio is about 4.57, versus layer-count ratio 3.36. That is evidence of unequal measured stage service under this profiler, not a throughput prediction from layer counts.

Stages overlap: **do not add their wall times** to get request duration. The prefix hands off chunk residuals through host memory, starts the suffix asynchronously and proceeds (`src/prefill/prefill.cpp:4158-4178`); suffix upload is at `:2188-2194`. Pipeline drain occurs at `Prefill::run` completion (`:4317-4359`). This permits RTX bursts followed by waiting for CMP-produced chunks. The profile's embed/steps intervals and wall/timeline gaps do not indicate a handoff-sized dominant cost at these lengths, but no independent copy/wait timer was enabled, so a pure handoff duration is not claimed.

The source-backed cost path is `src/prefill/prefill.cpp:3868-3950`: without MMQ, each selected expert is dequantized into FP16 buffers and executed through separate gate/up and down products. MMQ gathers quantized expert groups and executes grouped products instead (`:3898-3929`). GDN recurrence (`:2698-2707`) and QSA attention are small in these short profiles. **This does not establish their cost near 262K context.**

## Memory guard and partial MMQ evidence

The same guards were active in all arms: start with at least 6 GiB host available, stop below 5 GiB (4 GiB floor plus 1 GiB reaction margin), stop below 1,024 MiB free on either GPU, stop if another compute process appears. Polls are about once per second and cannot prevent every transient threshold crossing or guarantee against OOM.

| Arm | Minimum host available GiB | Minimum RTX free MiB | Minimum CMP free MiB |
|---|---:|---:|---:|
| Control | 17.30 | 1633 | 1043 |
| Profile | 17.21 | 1604 | 1043 |
| MMQ warmup, incomplete | 16.79 | 1573 | **971** |

All arms reported the same owned expert counts/bytes and one 3,993,600-byte reusable host load-staging buffer. Successful later requests reported **0.0 MB expert reads per request**; the first post-DONE counter includes 77,024.6 MB of startup expert reads and is not a fresh request delta. Other OS reads are not identified as expert loads. `summary.json` preserves those raw counter lines. MMQ was interrupted before DONE/counter publication, so its completed-request read counter and answer verification are unavailable; residency was established at startup, not inferred from partial progress.

Actual sample intervals were 1.000-1.101 seconds across all arms. Maximum sampled RSS was 5,025,083,392 / 5,030,289,408 / 5,059,313,664 bytes (control / profile / MMQ); sampled process swap and system swap use were zero. These samples cover startup and requests, not just prefill.

MMQ's last complete checkpoint was `PP 512 1977 927 552.4`; it is incomplete and its early rate is **not** a candidate speed result. At UTC `2026-10-16T19:51:08.891+00:00`, the guard sampled CMP 971 MiB and stopped the process. Model compute processes were absent afterwards; allocations returned to roughly 23.7 GiB RTX / 64.9 GiB CMP free.

## Bounded scratch-lifetime audit

References below apply to unchanged tested source. This is a feasibility audit, **not an implemented or validated memory optimization**.

- MMQ exact owned-prefill pricing rises **346 to 414 MiB**. CMP free after cache allocation changes 1714 to 1706 MiB (library/context difference), after head/prompt initialization 1196 to 1124 MiB, and after verifier/drafter binding 1110 to 1038 MiB. The first-use sampled plateau is another approximately 67 MiB lower in both paths (1043 and 971). The source/logs do not isolate that first-use allocation, so it is not assigned specifically to cuBLAS or one library.
- `Prefill::carve` allocates the same two FP16 dequant slots even when every expert layer uses MMQ (`prefill.cpp:1306`). Their payload is `2 * (1280*2560 + 2560*640) * 2 = 19,660,800 bytes`, **18.75 MiB raw**. Under this code's owned-page accounting (`Alloc::take`, `:387-419`, 256-byte padding and 2 MiB pages), the four allocations price at **24 MiB**, not 20 MiB. A conditionally all-MMQ layout could omit these slots, but fallback formats, MMQ-disabled execution, `relayout` and `bytes_needed_impl` must remain coherent. Even a full 24 MiB recovery leaves **29 MiB below the guard**, before any practical margin. It is not a standalone remedy.
- MMQ group weights are separately allocated at `:1312-1313`; the largest GU group uses Q5_K (176 bytes per 256 values) and the largest down group Q8_0 (34 bytes per 32 values), verified against `third_party/ggml/ggml-common.h:251-256,338-356`. At 16 experts, including 4096-byte tails, payloads are **36,048,896 bytes GU** and **27,856,896 bytes down**, pricing at **36 + 28 = 64 MiB** owned. The two additional small identity/bounds allocations each price at 2 MiB. These 68 MiB of MMQ allocations explain the measured owned-prefill price increase. Both group matrices are gathered before their products and remain live within a group's computation (`:3886-3929`); aliasing GU and down with each other would overwrite needed weights.
- GDN, QSA and MoE activation scratch **already share one maximum-sized region** (`:974-997`, `:1262-1305`). Counting the attention and MoE regions as separately reclaimable memory would double-count an existing saving. MMQ products still need their activation/intermediate/output arrays.
- The dense GEMM scratch (64 MiB payload) and 32 MiB cuBLAS workspace are separate allocations (`:971-972`, `:1216-1219`). Dense/native projections use them in attention and the shared expert (`gemm.cu:901-952`; `prefill.cpp:3005-3017`). MMQ group work is queued after shared-expert projection work on the same stage stream, even when the shared work is deferred (`:3245-3248`, `:3452-3479`). This suggests a narrowly gated **future lifetime-aliasing design** rather than extra buffers being mathematically all live together. It is not an existing switch or a proven generally safe alias: stage-helper cross-device users (`:1638-1655`), fallback products, buffer relayout, alignment/tails, and each backend must be covered. The cuBLAS workspace is not interchangeable with dequant scratch.
- Moving layer 36 to RTX is not a supported memory remedy: that layer's 512 expert blobs alone need **1,500 MiB**, before dense weights/state. Baseline sampled RTX surplus above the guard is only 609 MiB in this new run (483 MiB in the original long run). No split experiment was attempted.

Thus no straightforward existing supported option was found to recover the 53 MiB deficit **with working margin** while retaining full-context capacity, fixed chunk 256, full residency and both reserves. A later scratch-lifetime change would be separate focused optimization work requiring explicit layout safety tests, backend builds and controlled validation; this result is not sufficient reason to implement or run it now.

## Reproduction and limits

Successful commands, in order:

```sh
python3 -B profile-short.py --mode control
python3 -B profile-short.py --mode profile
```

Then `python3 -B profile-short.py --mode mmq-control` failed as expected on the VRAM guard; `mmq-profile` was not started. All run directories refuse overwrite. The harness loads the prior runner's resource/identity helpers and the prior deterministic prompt generator, but never invokes its long-run schedule. Each process is bounded to a 1,200-second startup and 120-second short request; no disk-progress stall timeout is used.

CPU checks (from this directory):

```sh
python3 -B -m unittest -v test_profile_short
python3 -B summarize.py > summary.json
sha256sum -c SHA256SUMS
```

Tests cover the hard prompt cap, complete-answer validation, no reuse, stage attribution/token matching and phase parsing. The original long-context evidence checksums remain unchanged. New raw logs, per-second monitor records, exact argv/environment/artifact identities and outcome JSON are preserved here; no new PCIe trace was collected, so the short profiles cannot supply copy-engine transfer attribution.

This is a single-host CUDA diagnosis and a failed opt-in feasibility test. It is not broad quality validation, near-max attention profiling, successful MMQ output parity, a faster-placement result, or completion of the broader Fossil comparison ticket. The ticket stays open. No more GPU runs are scheduled.
