# K36 stage-prefill 2048/512

This follow-up evaluates `--prefill-stage-chunks 2048,512` on K36:
stage 0 is the CMP 170HX (physical GPU 1), and stage 1 is the RTX 3090
(physical GPU 0). The session uses 262,144 cells and INT8 KV. K36 and K37
source-based sizing was completed before starting inference; both boundaries
passed the mandatory prompt-init admission with this stage-1 chunk.

| Split / GPU stage | Chunk | Owned prefill scratch | Required free with mandatory 512 MiB headroom | Idle free at sizing | Remaining margin |
|---|---:|---:|---:|---:|---:|
| K36 CMP 170HX (GPU 1) | 2,048 | 1,430,257,664 B | 66,316,140,544 B | 68,049,436,672 B | 1,733,296,128 B |
| K36 RTX 3090 (GPU 0) | 512 | 507,510,784 B | 25,044,614,380 B | 25,265,438,720 B | 220,824,340 B |
| K37 CMP 170HX (GPU 1) | 2,048 | 1,430,257,664 B | 67,987,570,688 B | 68,049,436,672 B | 61,865,984 B |
| K37 RTX 3090 (GPU 0) | 512 | 507,510,784 B | 23,368,989,932 B | 25,265,438,720 B | 1,896,448,788 B |

The guarded K36 sequence passed: one warmup, three 1,977-token prompts with
generation cap 32, then the 32,788-token historical fixture. Every prompt
matched its full historical input and output-ID/recall checks; each input was
fresh, with no prompt reuse. The 1,976-row stage-0 handoff was consumed in
four 512-token stage-1 sub-chunks. The three measured short prefill times were
1,410.7, 1,396.8 and 1,409.0 ms (median 1,409.0 ms); the 32,788-token prefill
took 15,529.8 ms.

Each of the four post-warmup requests reported 0.0 MB of expert source reads.
The warmup reported 77,024.6 MB of expert reads while priming the run; that
counter is preserved separately and is not hidden by the post-warmup result.
Minimum sampled free VRAM was 256,835,584 B (245 MiB) on RTX 3090 and
1,675,952,128 B (1,598.5 MiB) on CMP 170HX. Minimum host `MemAvailable` was
16,599,871,488 B (15.46 GiB). The engine and PCIe monitor exited with status
0, and no compute process remained.

The user-authorized policy was used: both discretionary engine reserves were
0 MiB and no external GPU-free abort was applied. The 6 GiB host startup
threshold, 5 GiB host stop threshold, 512 MiB prompt-init headroom and
mandatory CUDA/admission checks remained enabled. The lowest observed RTX
free memory was only 245 MiB, so this successful bounded run does not remove
the low-memory risk or guarantee safety under different conditions. No
near-limit, 128K or 262K prompt was run.

The historical uniform-1024 K37 run had a 1,368.6 ms short-prefill median
and a 20,237.2 ms 32,788-token prefill. This is descriptive context, not a
contemporaneous control; the layer split also differs. K37/2048/512 passed
offline admission but was not run.

## Evidence and reproduction

- `offline.py`, `ledger.csv`, `ledger.log`, and `offline.json`: source ledger
  command, exact stage scratch, physical GPU snapshot and admission margins.
- `mixed2048-512/engine.log`, `monitor.jsonl`, and `pcie.log`: raw startup,
  prompt, memory, process and PCIe telemetry.
- `mixed2048-512/*-counters.json` and `mixed2048-512/*.json`: per-request
  source-I/O counters, historical fixture metadata and raw output lines.
- `summary.json`, `checks.json`, `cleanup.json`, and `provenance.json`:
  analyzed outcomes, cleanup status and build/source identities.

From the repository root, reproduce the offline sizing, CPU contract tests,
and result analysis with:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/offline.py
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/test_stage_chunks.py
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/analyze.py
```

The guarded inference command was:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/run.py
```

CUDA targets `strata`, `strata_kq`, and `strata-split-memory` built
successfully; the split-memory self-test passed. HIP and SYCL were not
tested. `SHA256SUMS` records the evidence file hashes.

## Local API service

The `api/` folder contains the pinned server config and a host-memory-guarded
launcher for a local OpenAI/Anthropic-compatible API. It reuses the exact
tested engine, model pack, expert profile, prefill chunks and environment.
Start with `bash api/start-api.sh`, stop with `bash api/stop-api.sh`.
The default URL is `http://127.0.0.1:8080`; the model ID is
`qwen3.8-flash-next-unsloth-ud-q4_k_xl`. No API key is configured because the
listener is bound to loopback only. Start requires at least 6 GiB
`MemAvailable`; the supervisor signals the API server to stop below 5 GiB.
The engine still performs its own mandatory CUDA allocation/admission checks.
The service log is `api/server.log`, the engine log is `api/engine.log`, and
the PID file is `api/server.pid` (runtime files are git-ignored).

The launch has not been tested with near-limit prompts. It sets the engine's
262,144 context and may allocate the full KV cache during model startup; the
prior bounded inference checks covered short requests and one 32,788-token
request only. The lowest measured free RTX memory in that run was 245 MiB.
Use short API requests initially and do not interpret this launch as
full-context safety validation.
