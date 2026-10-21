# Offline utilization analysis, 2026-10-16

The saved telemetry supports the user's observation: the CMP 170HX was nearly continuously active during prefill, while the RTX 3090 showed a low median with intermittent high activity. This is consistent with unequal stage service times in the fixed 37/11 layer partition. It does **not** establish a specific kernel, transfer, or synchronization bottleneck, or show that changing the split would be safe or faster.

No model startup or GPU experiment was performed for this analysis. Original evidence and its `SHA256SUMS` are unchanged. The user's new direction replaces the repeated-long-run plan; remaining ticket comparisons are not authorization to repeat near-262K prompts.

## Measurements from the existing run

`analyze.py` uses the interior of each successful request's prefill: request elapsed time from 2 seconds through TTFT minus 2 seconds. This excludes phase boundaries conservatively, but it is approximate rather than a CUDA-event phase trace. Both GPUs are identified by physical `nvidia-smi` index: **0 = RTX 3090, 1 = CMP 170HX**, opposite to the command-order CUDA device indices under `--gpu 1,0`.

| Prefill input | Interior samples per GPU | RTX mean / median utilization | CMP mean / median utilization |
|---|---:|---:|---:|
| 8,202 tokens | 18 | 33.39% / 16.5% | 97.72% / 98.5% |
| 32,788 tokens | 84 | 40.15% / 16% | 97.37% / 98% |
| 131,094 tokens | 370 | 35.66% / 16% | 97.85% / 99% |
| 261,682 tokens | 752 | 34.64% / 16% | 97.75% / 99% |

Near-max details:

- The RTX was at or below 20% in 468/752 samples (62.2%), at or above 90% in 119/752 (15.8%), and ranged from 0% to 100%. The CMP was at or above 90% in all 752 samples (range 91-99%). These are **sample fractions**, not measured idle-time or occupancy fractions.
- The pattern persists through all four elapsed-time quarters: RTX means 36.59%, 31.92%, 31.86%, 38.25%; CMP means 97.76%, 97.91%, 97.76%, 97.58%. It is not confined to startup or pipeline fill/drain.
- Whole-device power averaged 172.42 W on the RTX and 172.28 W on the CMP. "RTX unused" is too strong: low sampled kernel activity is not zero power, zero memory work, or a missing suffix stage.
- Matching the same approximate prefill interior to PCIe log timestamps gives RTX mean RX/TX 86.98/1,757.31 MB/s, and CMP 120.56/38.38 MB/s (751 samples each). These counters include all device traffic and do not label activation handoffs, expert transfers, display traffic, or other copies. They cannot prove link saturation, nor be converted into model transfer bytes without attribution and integration.

Startup is separate: 275 samples average 28.22% RTX and 27.18% CMP utilization. Startup took 298.212 seconds. Decode lasted only 0.389-0.601 seconds per completed answer, shorter than the monitor cadence, so the saved samples cannot reliably characterize decode utilization. The interrupted repeat is excluded from the successful-prefill table.

The 1,550 monitor samples have actual consecutive intervals of 1.001-1.793 seconds (median 1.079). GPU utilization reports whole-device activity over the driver's sampling window, not achieved compute throughput or SM occupancy. Devices and PCIe rows are queried at different instants. Sampling/aliasing and graphics activity prevent exact correlation of individual spikes to chunks. The raw engine log has no per-line wall timestamps or per-stage kernel timings; `PP` checkpoints establish aggregate progress, not the service time of each GPU.

## What the tested source establishes

The following paths are byte-unchanged between candidate build commit `0327ba279ba50a34894f6aac16e5fe3c21106611` and this analysis's parent commit `aea11eaf3a0ede54a41374d5e1e53c4628b65711`. Line references apply to that tested source:

- `src/program/generate.cpp:7900-7930` initializes a separate prompt path for each layer stage and connects the CMP prefix to the RTX suffix. The raw startup log confirms `[0,37)` on CMP and `[37,48)` on RTX, with all 24,576 owned experts resident.
- `src/prefill/prefill.cpp:4158-4178` copies a chunk's residuals device-to-host and synchronizes the producing stream, waits for the successor's previous chunk if necessary, and launches the successor asynchronously. It then advances to the next chunk. `:2188-2194` uploads those residuals on the successor. This is a **host-staged chunk pipeline**, not strictly serialized whole-prompt execution and not guaranteed simultaneous full use of both GPUs.
- `src/prefill/prefill.cpp:4317-4359` drains the stage pipeline at the end of `Prefill::run`. `src/program/generate.cpp:11885-11898` and `include/strata/program/batch_read.hpp:22-27` show that this non-`--batch` command can process the prompt in one run rather than forcing a drain after every 256-token chunk.
- `src/prefill/prefill.cpp:2122-2126` gates the idle-successor expert-helper path on a **single-chunk** prompt run. That helper is not evidence that the RTX shares the CMP's 37 prefix layers during these multi-chunk long prompts.
- `src/prefill/prefill.cpp:2000-2042` provides opt-in `STRATA_PREFILL_TIMING` phase events. It was not enabled in the recorded environment and there are no such timing records in the saved log. Existing coarse utilization and PCIe counters cannot retrospectively replace them.

Consequently, the source permits the CMP to work on a later chunk while the RTX processes the preceding one. A prefix taking longer per chunk would leave a shorter suffix waiting for new input and explains this pattern qualitatively. However, 37 versus 11 layers alone does not measure stage time: device/kernel differences, layer types, context-dependent attention, host work, stream waits and transfers can contribute. High CMP activity does not prove compute saturation. No percentage of time is assigned to those causes here.

The residency log and later zero request-delta expert reads argue against repeated expert file loading as the explanation for this run. Nonzero OS reads include other process I/O; neither they nor PCIe totals identify the bottleneck. The CMP's previous 1,043 MiB free minimum (only 19 MiB above the guard) also means extra work buffers or a different placement cannot be assumed to fit.

## Decision and reproducibility

The useful conclusion is an observed prefill-stage activity imbalance, with a source-backed pipeline explanation and unmeasured causes. No split, defaults, engine code, or claims of a faster placement were changed. The existing near-max run already cost 821.16 seconds of prefill; repeating it is not justified to re-establish the same utilization pattern. Even a short diagnostic requires the measured roughly five-minute startup. There is no essential new experiment for this finding.

If a distinct causal question later warrants hardware work, collect stage/phase timing on at most 2K input tokens and at most 32 output tokens under the existing guards, explicitly accounting for pipeline fill/drain and profiler overhead. Such a short run cannot establish near-max attention behavior or a faster full-context placement. No run is started or scheduled here. The broader Fossil ticket stays open.

From this `offline-analysis/` directory:

```sh
python3 -B -m unittest -v test_analyze
python3 -B analyze.py > utilization.json
sha256sum -c SHA256SUMS
```

`utilization.json` records input hashes, per-stage utilization/power, matched PCIe summaries, actual sample intervals, and near-max elapsed-time quarters. Its generation uses only committed logs; it does not load model weights or call CUDA. CPU tests cover phase selection, physical-device indexing, and summary statistics.
