# K37 short MMQ comparison with a user-authorized 640 MiB monitor

2026-10-16, CMP 170HX 64 GiB + RTX 3090 24 GiB, CUDA only. This is a new evidence bundle; the prior long-context and 1,024 MiB-guard MMQ-failure evidence are unchanged.

## Result

**MMQ completed correctly and reduced measured short-request wall time by 1.91-2.16x against a timing-disabled control.** Both arms completed one warmup and nine measured requests; all 20 answers passed exact three-code recall, produced identical token IDs across corresponding arms, and read their full prompts with zero reuse. All 24,576 experts remained owned/resident at startup. Every measured request reported zero expert-read deltas.

The user explicitly changed the GPU-free **monitor** threshold from 1,024 to **640 binary MiB**. Configured engine allocation reserves remained 1,024 MiB on both GPUs, and the host stop stayed at 5 GiB available (4 GiB floor plus 1 GiB reaction headroom). MMQ's sampled CMP minimum was **938.3125 MiB**, 298.3125 MiB above the new monitor guard. No guard fired, no OOM was reported, and both processes exited zero and released their allocations.

This supports preserving `STRATA_MMQ_KQ=1` as an **opt-in short-input recipe under the user's changed guard**, not recommending it under the old 1,024 MiB guard. No default, reserve, runtime kernel, scratch layout or split was changed. This is not a near-max MMQ memory/quality validation.

## Timing-disabled comparison

Each cell is the median of three measured requests, excluding the warmup:

| Actual input tokens | Control prefill ms | MMQ prefill ms | Control wall to DONE s | MMQ wall to DONE s | Wall speedup |
|---:|---:|---:|---:|---:|---:|
| 502 | 1517.8 | 674.1 | 1.7249 | 0.9054 | 1.905x |
| 994 | 2848.9 | 1256.4 | 3.1010 | 1.4978 | 2.070x |
| 1977 | 5451.5 | 2401.5 | 5.6875 | 2.6348 | 2.159x |

Prefill speedup was 2.252 / 2.268 / 2.270x. At 1,977 tokens, wall times were control **5.6875, 5.6814, 5.7365 s**, MMQ **2.6348, 2.6406, 2.6304 s**. The raw arrays and all other sizes are in `summary.json`.

Startup was **192.304 s control** and **196.915 s MMQ**, separate from request timings. It was amortized across ten requests per process. Control ran first, then MMQ, serially; these were not randomized or interleaved trials. Warmup does not eliminate storage/cache-order, thermal, clock or host-load differences. Both arms use identical resource sampling, but observed wall/first-token arrival can include monitor scheduling and stdout queue latency.

`STRATA_PREFILL_TIMING` was absent in **both** arms. The prior profiler's approximately 30-33% overhead is not present in this denominator. No new phase profile or PCIe trace was collected; prior phase attribution motivated MMQ but is not used to claim a speedup here.

All answers completed in 22 output tokens, including EOS, below the 32-token cap. MTP remained enabled; accepted/offered drafts are retained per request. For the three large requests, control was **9/13, 9/13, 9/12**, MMQ **9/13, 9/12, 9/12**. Identical final output does not imply identical speculative execution. Large-request decode times were control **233.1, 229.2, 227.6 ms**, MMQ **230.2, 232.0, 227.3 ms**. Observed first-token times were control **5.4678, 5.4663, 5.5224 s**, MMQ **2.4185, 2.4226, 2.4171 s**; these are host-observed TTFTs, not independent GPU event intervals.

## Identity, protocol and opt-in recipe

`control/results.json` and `mmq/results.json` retain the complete executable argv, explicit environment overrides, actual guard, prompt hashes, model/pack/MTP identities, source revision, binary/cache/library hashes, answers, generated IDs, counters and timestamps. `analyze.py` asserts that both arms have identical argv, artifacts, executable/cache/library hashes and guards; their only environment difference is **`STRATA_MMQ_KQ=1`**.

The executable is the already-built unchanged-runtime CUDA Release sm80+sm86 binary at `build/cuda-short-profile/strata`, SHA-256 `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214`; `libstrata_kq.so` is `f0476d2db4fc631d77b2923f35dacdc7dcbb80462e583356c85d773768a12ef4`. Build commands/compiler/cache identity remain in `../2026-10-16-cmp170hx-rtx3090-short-profile/build-identity.json`. No rebuild or runtime edit was made here. Harness source parent was `a238151e1332abd0f5ae11b8a9843421877cd5f0`; runtime/build sources still match `0327ba27`.

Hardware/driver UUIDs are in `cleanup.json`: NVIDIA driver 610.43.03, physical index 0 RTX 3090, index 1 CMP 170HX. Engine `--gpu 1,0` makes CMP logical CUDA0. The same K37 placement, chunk 256, INT8 KV capacity 262,144, MTP/spec2, direct PLE, mmap experts, no borrowing, trim stage weights, all-owned experts and one 3,993,600-byte host staging buffer were retained. Large model weights were **not freshly rehashed**: original hashes and size/mtime checks were reused; small metadata artifacts were freshly checked.

The actual pack tokenizer generated the exact same prompt text, IDs and metadata as the preserved short-profile bundle. `prompts/` contains deterministic varied natural-text notes, chat formatting, seeds 83/97/109, independently expected `ORIGIN: 714`, `CENTER: 928`, `FINAL: 365`, and needle offsets near beginning/middle/end. Exact input lengths are 502, 994 and 1,977; capacity 262,144 is **allocated capacity, not processed tokens**. The batched prompt stage handles input minus the terminal seed token. Prefix/cache reuse is disabled and `DONE` verifies zero reuse.

For each arm the order was warmup at 1,977, then three repetitions of 502, 994, 1,977. The guarded commands from the repository root were:

```sh
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-mmq-640/run-short.py --mode control
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-mmq-640/run-short.py --mode mmq
```

Both passed. The harness refuses to overwrite existing result directories; preserve this bundle and use a new evidence location for any explicitly authorized rerun. Do not run an unguarded bare engine command as a substitute: the 640 MiB monitor belongs to this harness, not an engine allocation-reserve option. Inherited `STRATA_*` and `CUDA_VISIBLE_DEVICES` are cleared, then exact recorded overrides are applied. MMQ startup positively confirms loading the K-quant library.

## Memory, residency and reads

NVML reports actual physical memory counters in bytes, independently of the rounded `nvidia-smi` display:

| Arm | Minimum RTX free bytes (MiB) | Minimum CMP free bytes (MiB) | Minimum host available GiB |
|---|---:|---:|---:|
| Control | 2,117,795,840 (2019.6875) | 1,092,943,872 (1042.3125) | 13.766 |
| MMQ | 2,027,618,304 (1933.6875) | 983,891,968 (938.3125) | 12.245 |

Maximum sampled RSS was 4.676 / 4.743 GiB (control/MMQ); process swap was zero. Maximum system swap use was **1,310,720 bytes**, not process paging. Regular monitor intervals are about one second; forced post-request samples shorten some gaps to 0.062 / 0.252 s. Maximum observed gap was 1.100 / 1.101 s. Sampling is not a hard OOM guarantee and does not bound unsampled transients.

The MMQ startup ownership diagnostics report **18,944 experts / 59,192,115,200 bytes on CMP**, **5,632 / 17,825,792,000 bytes on RTX**, all 24,576 resident, with CPU complement and cache loans disabled. Owned-prefill pricing remains 346 MiB control, 414 MiB MMQ. Unlike the earlier interrupted warmup at 971 MiB, this run proceeds through full warmup and repeated complete requests; its lower 938.3125 MiB minimum must not be hidden by quoting the earlier partial-run minimum. Exact attribution of the additional first-use/steady allocations is not established.

The first post-warmup file-tier diagnostic includes **77,024.6 MB startup-cumulative expert reads**; it is not a request delta. The harness waits for the diagnostic after each DONE and requires each subsequent measured request to report **0.0 MB expert reads**. Other OS reads remain nonzero (e.g. PLE/file-backed work); zero expert reads is not a claim of zero storage traffic. Per-request counter lines, process I/O counters and raw logs are preserved.

## Why the 3090's free 1.5 GiB does not establish K36 fit

The user's free-memory observation is reasonable: the earlier long run showed **1,507 MiB total free**, while this run has more desktop/device headroom. But that is **total free**, not bytes available after preserving a 640 MiB guard. At 1,507 MiB, only 867 MiB is available above the new guard; layer 36's experts alone need 1,500 MiB.

The bounded offline recalculation uses the preserved `../2026-10-16-cmp170hx-rtx3090-k37/ledger-chunk256.csv`, its hash, and unchanged source. It does not start a ledger executable or CUDA/model process. Exact component arithmetic for moving layer 36 from CMP to RTX (K37 to K36):

| Component | RTX added bytes | CMP removed bytes |
|---|---:|---:|
| Layer 36 expert payload | 1,572,864,000 | 1,572,864,000 |
| Canonical arena payload | 29,742,976 | 27,121,536 |
| Native projection allocation estimate | 69,206,016 | 69,206,016 |
| Session arena payload | 3,268,608 | 3,268,608 |
| Per-stage prefill delta | 0 | 0 |
| Output head / MTP / fixed allowance delta | 0 | 0 |
| Total with ledger arena/session granule estimates | **1,675,624,448 (1598 MiB)** | **1,671,430,144 (1594 MiB)** |

Layer 36 uses **Q4_K gate/up + Q5_1 down**, 3,072,000 bytes per expert times 512; it is not priced as an all-Q4 average. The native artifact also contains Q4_K/Q8_0 and Q5_K/Q8_0 layers, whose costs are retained in the existing per-layer ledger. Layer 36 is GDN: moving it leaves QSA counts at 9/3, so it adds no 262K QSA KV layer. Its incremental recurrent/session payload is 3,268,608 bytes. Chunk/context, global MMQ plan and stage count are unchanged, so prefill scratch does not grow by another whole layer's scratch. Its existing MMQ increment is already present in the measured K37 MMQ free bytes, not added twice.

Source: `src/program/split_memory_ledger.cpp:188-237` sums mixed-format expert blobs, stage-trimmed canonical/native weights, session and owned prefill; `src/program/generate.cpp:3120-3146` retains all routers on CUDA0 but not the suffix, explaining the asymmetric canonical deltas. `src/core/native_dense.cpp:186-227` sums per-matrix allocation estimates; `src/core/session.cpp:56-75` prices range-local GDN/QSA state; `src/prefill/prefill.cpp:1874-1939` prices fixed per-stage prefill from chunk/context and the MMQ plan.

Against **new actual resident free bytes**, not the whole 24 GiB card or unloaded free memory:

| Arm | K37 sampled RTX free MiB | After estimated 1598 MiB K36 increment | Deficit below 640 MiB |
|---|---:|---:|---:|
| Control | 2019.6875 | 421.6875 | 218.3125 |
| MMQ | 1933.6875 | 335.6875 | 304.3125 |

The target requires **at least 2238 MiB K37 free** before extra unmodeled margin. The MMQ arithmetic deficit is exactly **319,094,784 bytes** under these inputs. These are exact arithmetic operations on byte counters and ledger estimates, **not exact physical allocation predictions**: per-matrix granule rules and fixed allowances are estimates, and a different split could alter graph/temporary costs. Even experts alone exceed the available surplus above 640 MiB in both arms. No safe K36 capacity was established, so **K36 was not run or recommended**. This is not a best-split claim.

## Verification and remaining limits

From this directory:

```sh
python3 -B -m unittest -v test_short
python3 -B analyze.py > summary.json
sha256sum -c SHA256SUMS
```

Six CPU tests passed, including the exact 640 MiB boundary, one-byte-below rejection, unchanged host reaction headroom, other-process rejection, prompt identity and K36 arithmetic. Offline result checks verify all 20 completed requests, no reuse, identical output IDs, matching invocation/build/artifact/guard identities, residency and zero measured expert rereads. Prior evidence checksums also pass. `cleanup.json` records no compute processes after both zero exits.

CUDA only; HIP/SYCL untested. No long run, scratch change, new algorithm, split change, push or merge. Three-code recall and token identity on these prompts do not establish general numerical/quality parity; MMQ may change rounding. No near-max MMQ attention, peak memory or output test, MTP-off comparison, alternate placement or expert-service comparison was performed. The broader Fossil ticket `5f8e4f2509` remains open.
