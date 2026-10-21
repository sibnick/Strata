# K36 MMQ with user-authorized RTX allocation reserve 256 MiB

2026-10-16, CUDA CMP 170HX + RTX 3090. **K36 passed startup and all ten short requests** after the user authorized retry following discussion of reducing the separate RTX allocation reserve. Prior failed K36 admission and K37 evidence are preserved unchanged.

Only `--layer-split 37` to `36` and `--vram-reserve-later-mib 1024` to `256` differ in engine argv from the successful K37 MMQ run. RTX physical index 0 external monitor is 256 binary MiB; CMP physical index 1 monitor remains 640 MiB and its allocation reserve remains 1,024 MiB. Host start6 GiB / stop5 GiB (4 GiB floor plus 1 GiB reaction) is unchanged. MMQ is still opt-in (`STRATA_MMQ_KQ=1`); timing is disabled. No runtime/scratch/driver/system modification was made.

## Correctness, residency and safety

One warmup at 1,977 tokens followed by three repetitions each of 502, 994 and 1,977 tokens completed, output cap32. All ten answers generated the exact expected ORIGIN714/CENTER928/FINAL365 three-code answer in22 tokens including EOS, with zero prompt reuse and full fresh input. Corresponding measured output IDs match the earlier K37 MMQ run. Every post-warmup request reports0.0 MB expert-read delta; the first77,024.6 MB diagnostic includes startup-cumulative reads, not a warmup delta. Other OS reads remain nonzero.

Startup confirmed all24,576 experts resident:

| Device | Owned layers | Experts | Expert bytes |
|---|---|---:|---:|
| CMP | [0,36) | 18,432 | 57,619,251,200 |
| RTX | [36,48) | 6,144 | 19,398,656,000 |

Minimum sampled free memory was **360,382,464 bytes /343.6875 MiB RTX**, **2,670,002,176 bytes /2546.3125 MiB CMP**. RTX therefore had only **87.6875 MiB above its256 MiB monitor threshold**. Host available minimum was13,487,013,888 bytes /12.561 GiB; max sampled RSS5,081,366,528 bytes /4.732 GiB. Process swap0; max system swap1,310,720 bytes. No monitor trigger or OOM; exit0. Maximum sampling gap1.100s, with extra post-request samples. Polling does not bound unsampled transients or guarantee against OOM.

This demonstrates short-input feasibility at the **changed reserve**, not comfortable VRAM margin or near-max memory safety. Full INT8 KV context262,144 remains allocated, but no input above1,977 tokens was processed. Longer attention/workspace behavior and other prompts remain untested. No long run is implied or scheduled.

## Measured timings and earlier K37 comparison

Warmup excluded; medians of three requests. Startup **179.600s** is separate and amortized across ten requests.

| Input tokens | K36 prefill ms | K36 wall to DONE s | Earlier K37 MMQ wall s | Earlier K37/K36 wall ratio |
|---:|---:|---:|---:|---:|
| 502 | 670.6 | 0.8776 | 0.9054 | 1.032 |
| 994 | 1238.6 | 1.4481 | 1.4978 | 1.034 |
| 1977 | 2362.6 | 2.5728 | 2.6348 | 1.024 |

K36 large-request wall times2.5586/2.5728/2.5772s and prefill2349.9/2362.6/2366.3ms are retained in raw results. Large-request decode205.7/207.6/207.9ms, observed TTFT2.3669/2.3786/2.3829s. All measured K36 requests accepted10 of11 MTP drafts; earlier K37 large requests accepted9/13,9/12,9/12. Identical final IDs therefore do not establish identical speculative execution or a pure stage-placement speed effect.

The small2-3% wall difference is **descriptive only**: K37 was earlier, not randomized/interleaved or repeated under matched allocation reserves/monitor settings. Clocks, thermal/cache/host state, speculative execution and measurement scheduling may differ. No faster/best-split conclusion is established. No profiler overhead is included in either denominator.

## Reproduction, identity and evidence

From repository root:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-16-cmp170hx-rtx3090-k36-reserve256 -p test_short.py -v
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k36-reserve256/run-short.py --mode mmq
```

Both passed. Runner refuses overwrite. Exact argv/env, guards, source parent5f54e1ea, executable/cache/library hashes, model/pack/MTP manifest identities, startup/request times, counter lines, outputs and token IDs are in `mmq/results.json`; raw log/monitor and progress are separate. Executable SHA256 remains `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214`, unchanged CUDA13.4.92 Release sm80+86 build. Build details remain in the short-profile bundle. Original large model hashes were reused with size/mtime checks, not freshly rehashed; small metadata checked freshly.

Prompt text, actual tokenizer IDs and hashes match the preserved natural-text/chat-formatted fixtures, deterministic seeds83/97/109, independent expected needle codes near beginning/middle/end. K37 comparison path is `../2026-10-16-cmp170hx-rtx3090-mmq-640/mmq/results.json`. `analyze.py` checks exact identity and permitted argv differences offline, zero reuse/output counts/recall and residency. `cleanup.json` verifies no compute process and returned allocations.

From this directory, `python3 -B analyze.py > summary.json` regenerates the deterministic analysis; `sha256sum -c SHA256SUMS` verifies evidence. Six CPU tests passed for asymmetric boundaries, unchanged host protection and explicit exited/live process error handling; existing bundles' checksums remain valid.

No pushes/merges, backend/runtime changes, long inputs or further split experiments. CUDA host only; HIP/SYCL untested. Recall on these fixtures is not generalized quality parity. Near-max K36/MMQ, repeated near-max success, MTP-off and expert-service comparisons remain untested; broader Fossil ticket5f8e4f2509 stays open.
