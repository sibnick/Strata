# K37 MMQ near-limit context: one successful request

2026-10-16, CMP 170HX64 GiB + RTX3090 24 GiB, CUDA. The user explicitly requested long-context MMQ on K37, superseding the previous short-only restriction for this one test. **The original261,682-token prompt passed exact retrieval with MMQ, full fresh prefill, and37 completed output tokens.**

Prefill was **341,240.9 ms /766.854 tokens/s**, compared with historical timing-disabled non-MMQ **821,160 ms /318.674 tokens/s** on the same prompt: **2.406x**. Observed TTFT was341.288s versus historical821.207s. This is a historical-run comparison, not randomized/interleaved same-day paired evidence: clocks, thermal/cache/host/desktop state and the rebuilt executable differ. No broad numerical-quality or best-placement claim is made.

## Configuration and identity

K37 restored the original stage placement: CMP logicalCUDA0 layers[0,37), RTX logicalCUDA1[37,48), physical order1,0, no P2P. Both engine allocation reserves were **1,024 MiB** and both external monitor thresholds **640 binary MiB**. K36's RTX256 exceptions did not carry over. Host start6 GiB, stop5 GiB (4 GiB floor plus1 GiB reaction) stayed unchanged.

The same validated rebuilt executable and artifact identities as the successful short-MMQ arm were used. Exact argv/env, SHA256s, source parent `ed6dd202`, metadata and original large-file hashes/size/mtime checks are in `mmq/results.json`. Large weights were **not freshly rehashed**; small metadata freshly checked. Executable `build/cuda-short-profile/strata` SHA256 `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214`, library `f0476d2db4fc631d77b2923f35dacdc7dcbb80462e583356c85d773768a12ef4`; CUDA13.4.92 Release sm80+86. Build details remain in `../2026-10-16-cmp170hx-rtx3090-short-profile/build-identity.json`. Runtime/build source matches historical0327ba27; the original executable is no longer present, so binary identity with the historical control is not claimed.

Only the existing opt-in `STRATA_MMQ_KQ=1` is added to the original configuration. `STRATA_PREFILL_TIMING` is absent; profiler overhead is not included. INT8 KV capacity262,144, fixed chunk256, trim stage weights, full-owned experts, mmap/direct expert reads, direct PLE, no prefill borrowing, prompt cache0 and MTP/spec2 are unchanged. Engine startup positively reports the K-quant library loaded.

Ownership startup diagnostics:

| GPU | Layers | Experts | Resident expert bytes |
|---|---|---:|---:|
| CMP | [0,37) | 18,944 | 59,192,115,200 |
| RTX | [37,48) | 5,632 | 17,825,792,000 |

All24,576 experts are reported resident, CPU complement/cache loans disabled, one3,993,600-byte reusable host staging buffer. No runtime, layout, driver/system or model-artifact modification was made.

## Prompt, request sequence and correctness

The process ran one1,977-token short warmup, the **original32,788-token** sanity fixture, then **one original261,682-token near-limit fixture**. No redundant8K/128K ladder, repeated near-max request or MTP-off run was performed.

The original long fixtures were reused read-only, not regenerated; `prompt-references.json` records their exact committed paths, file hashes, metadata and historical results hash. The original actual pack tokenizer produced natural-text field notes in the same chat wrapper, with deterministic beginning/middle/end needles and independently specified answers. Token counts/ID hashes are verified before submission. Short warmup text/IDs are retained in `prompts/`.

Near-limit original ID hash is `67a876781b4fde7323b1b340a5e55700acb3f6741dc19d953d32229c42cf55ed`; source prompt text hash `10c7eb3f6a8a385458ecbaee8d71e00f2bae6995b6d1664ba57f3d0d5ea6ecbc`. Needle token offsets59 /130846 /261559. Exact completed answer:

```text
ORIGIN: IRIS-4Q7-LANTERN
CENTER: CEDAR-8M2-QUARRY
FINAL: ORBIT-6V9-KETTLE
```

The32K and near-limit answers generated37 tokens includingEOS, cap192, identical token IDs to their historical non-MMQ answers. `DONE` verifies **zero reused,261,682 freshly read** for near-limit. The batched prefill checkpoint ends at261,681 because the last input seed token takes the token path; total input is261,682, not261,681. Input plus output cap reserves **261,874 of262,144** cells; actual input+output261,719. This is near-limit context, not262,144 input tokens. Allocated capacity is distinguished from actual processing.

## Timing and progress

| Request | Input | Output/cap | Prefill ms | Prefill tokens/s | Observed TTFT s | Wall to DONE s | MTP accepted/offered |
|---|---:|---:|---:|---:|---:|---:|---|
| Warmup | 1977 | 22/32 | 2640.5 | 748.722 | 2.666 | 2.893 | 9/12 |
| 32K | 32788 | 37/192 | 40146.6 | 816.707 | 40.169 | 40.556 | 18/23 |
| Near-limit | 261682 | 37/192 | 341240.9 | 766.854 | 341.288 | 341.763 | 18/24 |

Startup **172.172s** is separate and amortized across the three requests. One resident process was reused, but every prompt had zero prefix reuse. Timeout is `max(1800s,4*input_tokens/preceding_measured_prefill_rate+300s)`; the near-limit bound was1800s after the32K measured816.707tokens/s. No disk-progress watchdog was used. Request writing is threaded so monitoring continues during pipe submission.

Raw engine PP checkpoints are durable; `progress.txt` timestamps approximately30-second checkpoints. Near-limit PP rates went from827.9tokens/s at25,344 tokens to767.8 at255,744; final766.854. The slowdown at increasing context is measured but this test has no phase breakdown to attribute it. Progress/ETA was reported after the32K sanity stage.

Historical non-MMQ32K prefill94,747.1ms /346.058tokens/s is2.360x the new prefill time. Historical near-limit TTFT821.207s and prefill821.160s are about2.406x the new result. This does not reuse intrusive phase-timed measurements as the denominator.

**Decode did not improve:** near-limit421.7ms (37/0.4217=87.740tokens/s by engine convention) versus historical389.3ms /95.042tokens/s. MTP accepted18/24 (75%) versus historical20/22 (90.9%); one short decode is not a representative MTP/quality comparison. Host-observed TTFT and request wall times include pipe/queue/monitor scheduling, not independent kernel events.

## Memory, swap, storage and PCIe

Overall sampled minimum free memory: **CMP967,114,752 bytes /922.3125MiB**, **RTX2,006,646,784 bytes /1913.6875MiB**. Both minima also occur during near-limit processing. CMP remained282.3125MiB above640; the old1024 monitor would not have permitted this run.

Near-limit/overall host available minimum **13,033,177,088 bytes /12.138GiB**; max sampled RSS **5,107,630,080 bytes /4.757GiB**; process swap0, max system swap1,310,720 bytes. Per-stage minima are in `summary.json`. Sampling about1s plus forced post-request observations, max observed gap1.100s. These are **sampled extrema, not true guaranteed peaks**; polling cannot prevent every transient pressure event/OOM. No guard or OOM occurred, engine exit0.

The first warmup file-tier line includes startup-cumulative **77,024.6MB expert reads**. Both subsequent32K and near-limit intervals report **0.0MB expert reads**, not an inference from cumulative counters. Near-limit OS read10,869.3MB (one major fault); other file-backed/PLE traffic remains, so no zero-storage claim. Request-specific counters and process cumulative I/O samples remain raw.

`mmq/pcie.log` captures timestamped whole-device `nvidia-smi dmon` RX/TX, not model-only transfers or PCIe saturation. Near-limit314samples/device: RTX meanRX59.37/TX10.72, max646/84; CMP meanRX86.20/TX43.73, max238/557, in the tool's printedMB/s units. It is not a phase-isolated or copy-engine attribution. No new profiler was enabled. Telemetry helper and engine were stopped; `cleanup.json` records no compute processes and released allocations.

## Reproduction and scope

From repository root:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-16-cmp170hx-rtx3090-k37-mmq-long-context -p test_long.py -v
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k37-mmq-long-context/run-long.py --mode mmq
```

Both passed. The runner refuses overwrite and imports immutable original helper/prompt modules. From this directory `python3 -B analyze.py > summary.json` deterministically regenerates the offline comparison; `sha256sum -c SHA256SUMS` checks the bundle. Four CPU tests cover original near-limit ID count/hash/headroom, output completion/no-reuse validation, both640 guards and measured-rate timeout. Independent checks verify exact binary/model/argv/env, all three completed requests, safety samples, no profiling and original evidence checksums.

This is **one successful K37 MMQ near-limit recall test**, not a general quality evaluation, repeated near-max acceptance, controlled placement comparison or expert-service comparison. No K36 long test, MTP-off or repeat was run. CUDA host only; HIP/SYCL untested. No pushes/merges/new agents/optimizations. Broader Fossil ticket5f8e4f2509 remains open.
