# GPU-owned CUDA long-context run

## Outcome

The strict GPU-owned K37 baseline completed tokenizer-verified retrieval runs at 8K, 32K, 128K, and near the 262,144-token context limit. All four answers completed with no prompt-prefix reuse and returned the three exact codes planted at the beginning, middle, and end of each prompt.

The user then requested an immediate stop while a second near-max request was still prefilling. Its last complete engine checkpoint was 14,848 / 261,687 input tokens (5.67%); it produced no answer and is not a passing repeat. The process was stopped, GPU allocations were released, and no further model runs are authorized by this result.

This is evidence for one static K37 CUDA placement only. It does not satisfy the broader Fossil ticket acceptance: no completed repeat near-max run, MTP-off comparison, alternate-placement comparison, or expert-service comparison was run. The ticket remains open.

## Run identity and invocation

- Session branch: `sibnick-long-context-validation`; tested source commit `e5a06131a4bb3455de69786b6ce6dcf50c1d3aeb`, descended from `e5a06131` and `bb41737f`.
- Candidate binary: `/tmp/strata-gpu-owned-cuda/strata`, SHA-256 `06a89cd0a9e8725ead0da3b55cc85c4cc11e8d33e4864d2e97b3aeec354ad1eb`; build source commit `0327ba279ba50a34894f6aac16e5fe3c21106611`.
- Build: CUDA 13.4.92, Release, `sm_80;sm_86`, driver 610.43.03; CMake cache SHA-256 `fee6be1a4eefc5da96c3b06f9adceb2f88e1f4b21485ec213c47aac95bce86c7`.
- Model revision: `38bb39ee97821de2c9009abb7e93950eec396e66`; model shards at `/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/`. MTP artifacts: `/home/nikolay/projects/Strata-data/mtp/rt`.
- Original K37 artifact manifest: `../2026-10-16-cmp170hx-rtx3090-k37/artifacts.json`, SHA-256 `de3b5214d60af2709d19baf8deed4f1c89e35006e26023882794437011d2ddd6`. Large artifacts over 16 MiB were not freshly rehashed; their sizes and modification times were checked against that manifest. Small artifacts were freshly hashed. The earlier comparison evidence was not overwritten.
- Harness invocation: `python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context/run-long-context.py --mode main`.
- The exact engine argv is in `main-results.json` and the `COMMAND` line in `main-progress.log`. It reproduces the prior comparison configuration: `--gpu 1,0 --layer-split 37 --trim-stage-weights --max-context 262144 --kv int8 --mtp /home/nikolay/projects/Strata-data/mtp/rt --spec 2 --prefill 256 --no-prefill-borrow --mmap-experts --resident-budget-gib 0.001 --expert-profile /home/nikolay/projects/copilot-worktrees/170hx-3090/sibnick-upgraded-succotash/data/expert-profile.bin --ple-io direct --ple-row-cache 1024 --vram-reserve-mib 1024 --vram-reserve-later-mib 1024 --pool-workers 1 --serve --prompt-cache 0 --conversation-cache-min-free-mib 4096 --gpu-owned-experts`, with the recorded `--pack` and `--native` paths in the manifest.
- `STRATA_*` values were cleared and set to the values in `main-results.json`; `CUDA_VISIBLE_DEVICES` was unset. Other inherited environment variables were retained and were not serialized.
- `prepare-long-context.py` deterministically regenerates the five prompt files from the local pack tokenizer. Its output is written to this directory's `prompts/`; do not run it if you need to preserve a local edit there.

## Placement and residency

The command uses `--gpu 1,0`: CUDA device 0 is the CMP 170HX (physical `nvidia-smi` index 1), owning layers `[0,37)`; CUDA device 1 is the RTX 3090 (physical index 0), owning layers `[37,48)`. P2P was not used. All 24,576 experts were reported resident in their owning GPU caches: 18,944 on the CMP (59,192,115,200 bytes) and 5,632 on the RTX 3090 (17,825,792,000 bytes). The CPU complement and prompt-cache loans were disabled. The engine reported one reusable 3,993,600-byte host staging buffer and no expected file fallback after startup.

The engine reported 77,024.6 MB of startup file-tier reads for expert residency. Its first `file tier I/O this request` line reports process-cumulative totals from the engine's startup baseline (153,462.9 MB OS reads and 77,024.6 MB of expert reads), not an isolated 8K-request delta. The later per-request engine lines report zero expert reads; their OS-read totals are 1,728.0 MB at 32K, 5,967.3 MB at 128K, and 10,877.0 MB near-max. The corresponding post-ready `/proc` `io_read_bytes` deltas, including all process file activity rather than expert-only reads, are in `main-results.json`. These counters should not be interpreted as a pure SSD-expert-read measurement.

## Completed requests

Prompts use varied natural-language field notes in the same chat wrapper as the prior 256-token comparison. The actual pack tokenizer produced the counts below. `TTFT` includes prefill and first-token latency. Decode throughput and MTP acceptance are per request.

| Stage | Input tokens | Input + 192-token output cap | Prefill (ms) | Prefill tok/s | TTFT (s) | Decode tok/s | Output tokens | MTP accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,202 | 8,394 | 23,583.6 | 347.78 | 23.621 | 61.58 | 37 | 20/26 |
| 32K | 32,788 | 32,980 | 94,747.1 | 346.06 | 94.773 | 93.51 | 37 | 20/28 |
| 128K | 131,094 | 131,286 | 399,446.5 | 328.19 | 399.479 | 94.73 | 37 | 20/28 |
| Near-max | 261,682 | 261,874 | 821,160.0 | 318.67 | 821.207 | 95.04 | 37 | 20/22 |

Each completed `DONE` record confirms all input tokens were read, `reused=0`, and generation completed. Each generated answer exactly matched:

```text
ORIGIN: IRIS-4Q7-LANTERN
CENTER: CEDAR-8M2-QUARRY
FINAL: ORBIT-6V9-KETTLE
```

For the successful near-max request, input plus its configured generation capacity used 261,874 of 262,144 context cells, leaving 270 cells of capacity. It generated 37 tokens, so actual input plus output was 261,719 tokens. This is a near-full-context input, not a 262,144-token input; the generation headroom is included in the context accounting. Needles were at token offsets 59, 130,846, and 261,559 in the near-max prompt; all expected values were independently checked by the harness.

The second near-max prompt had 261,687 input tokens and a 192-token generation cap. The user stop time was not captured. `main-results.json` records its last complete `PP` checkpoint (14,848 / 261,687 at 361.9 tok/s after 41,030 ms), the latest monitor sample, and its interrupted status. A following trace line had begun the 15,104-token chunk, but there was no subsequent complete `PP` checkpoint, generated output, or retrieval check.

## Resource and transfer observations

- The runner sampled host and GPU state about once per second (1,550 samples); these polls are not a hard OOM guarantee. It stopped on the configured 1,024 MiB GPU-free threshold or below 5 GiB host `MemAvailable`; startup required 6 GiB. Neither stop guard was crossed.
- Minimum host `MemAvailable` was 17,801,887,744 bytes (about 16.58 GiB). Maximum sampled engine RSS was 5,045,956,608 bytes (about 4.70 GiB); engine swap was zero. Maximum system swap used was 1,002,700,800 bytes (about 0.93 GiB).
- Minimum free VRAM was 1,507 MiB on the RTX 3090 and 1,043 MiB on the CMP 170HX. The CMP minimum was only 19 MiB above the configured stop threshold; it is not a comfortable margin.
- `nvidia-smi dmon` PCIe data is whole-device throughput and may include other graphics activity. During the completed near-max request, physical GPU 0 (RTX 3090) averaged 86.9 MB/s RX and 1,760.6 MB/s TX (maxima 2,729 and 3,133 MB/s); physical GPU 1 (CMP 170HX) averaged 120.4 MB/s RX and 38.4 MB/s TX (maxima 244 and 580 MB/s). Per-request samples are in `main-results.json`; raw samples are in `main-pcie.log`.
- No OOM was observed. These measurements cover this single CUDA host and configuration only.

## Limits and remaining acceptance

MTP was enabled for all completed requests; the recorded acceptance counts do not compare MTP on versus off. Only one near-max request completed, and the repeat was stopped by user request. There was no alternate layer split, expert-service prototype, or broader output-quality test. Retrieval checks establish only recovery of these three planted codes; they do not establish general long-context quality. HIP and SYCL were not tested. Keep the Fossil ticket open and treat these results as partial static-layer-placement evidence.

## Evidence files

- `prepare-long-context.py`, `run-long-context.py`: deterministic prompt creation and guarded sequential runner.
- `prompts/`: tokenizer input IDs, text, per-prompt metadata, and tokenizer manifest, including the incomplete repeat prompt.
- `generated/`: completed answer text and token IDs for the four passing requests.
- `main-results.json`: exact argv/environment overrides, build/model identity, prompt hashes, per-request metrics, resource minima, output verification, and interrupted-repeat status.
- `main-engine.log`, `main-monitor.jsonl`, `main-pcie.log`, `main-progress.log`: raw engine, sampled resource, PCIe, and runner progress evidence.
- `SHA256SUMS`: checksums for this evidence bundle, excluding the checksum file itself.
