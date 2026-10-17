# K36 per-stage prompt-chunk sizing

This records the opt-in `--prefill-stage-chunks 2048,1024` implementation and
its offline admission check for K36. **No GPU inference workload was started.**
The source-based totals fit the sampled idle memory before one unchanged
mandatory check is applied; the K36 RTX 3090 is short of that prompt-init
headroom, so the requested warmup, three short requests and 32K request were
not run.

The option is limited to an explicit, fully owned, multi-GPU `--serve` layer
split. It takes one chunk per stage in GPU order; stage 0 must equal
`--prefill`. Each stage receives scratch and pinned handoff buffers sized for
its own chunk, and admission/ledger calculations price the stages separately.
A downstream stage iterates a larger handoff in its own chunk size. With the
option omitted, the existing uniform `--prefill` behavior is retained. The
last-stage MTP prompt-KV callback cadence can change with the consumer chunk;
the 262,144-cell INT8 session, model ownership and decode state are unchanged.
Because no inference ran, the handoff, MTP behavior and exact output parity
remain unverified on this hardware.

## Sizing and stop decision

The source-based ledger used context 262,144, INT8 KV, `STRATA_MMQ_KQ=1`,
batched gather, and `--additional-reserve-mib 0`. The additional CUDA reserve
was zero and the external GPU-free abort threshold was disabled, as
authorized. These settings do **not** remove the mandatory 512 MiB
prompt-initialization headroom or CUDA allocation checks.

| Split | Stage / physical GPU | Chunk | Owned prefill estimate | Ledger requirement before model load | Idle free at offline sizing | Margin after mandatory 512 MiB headroom |
|---|---|---:|---:|---:|---:|---:|
| K36 | CMP 170HX (GPU 1) | 2,048 | 1,430,257,664 B | 65,779,269,632 B | 68,049,436,672 B | 1,733,296,128 B |
| K36 | RTX 3090 (GPU 0) | 1,024 | 1,025,507,328 B | 25,025,740,012 B | 25,265,438,720 B | **-297,172,204 B** |
| K37 | CMP 170HX (GPU 1) | 2,048 | 1,430,257,664 B | 67,450,699,776 B | 68,049,436,672 B | 61,865,984 B |
| K37 | RTX 3090 (GPU 0) | 1,024 | 1,025,507,328 B | 23,350,115,564 B | 25,265,438,720 B | 1,378,452,244 B |

K36's raw ledger totals are below the idle free-memory snapshot, but the
RTX's raw margin is only 239,698,708 B. The retained check requiring
`Prefill::bytes_needed_owned` plus 512 MiB makes its required free memory
25,562,610,924 B, 297,172,204 B more than the measured preflight free memory.
The guarded runner recorded `offline-infeasible` and exited before starting
the engine or PCIe monitor. K37 passes this source estimate but was not run.

The actual preflight also passed the host start threshold (at least 6 GiB
MemAvailable) and found no unrelated compute process. The authorized policy
for any future GPU run remains: both engine reserves at 0 MiB, no external
free-VRAM abort, host start at 6 GiB and stop below 5 GiB, plus all mandatory
CUDA/allocation admission checks. No under-load minimum VRAM or host memory
was measured here; the table contains idle snapshots only.

## Descriptive comparison

The previous uniform-1024 K37 arm reported short prefill times of 1,368.6,
1,360.8 and 1,369.0 ms (median 1,368.6 ms), and 20,237.2 ms for its single
32,788-token request. Those numbers are historical context only, not a
contemporaneous control or an estimate of K36 mixed-chunk performance. The
prior K37/1024 run sampled minimum free memory of 382.3125 MiB on the CMP and
1,361.6875 MiB on the RTX under the same relaxed discretionary GPU-reserve
policy. No mixed-K36 performance, quality or memory-safety conclusion follows.

## Build, tests and retained evidence

The touched CUDA targets built successfully:

```text
cmake --build build/cuda-per-stage --parallel 8 --target strata strata_kq strata-split-memory
```

The CPU contract tests cover per-stage sizing, the preserved host/compute
guards, fixture IDs/freshness contract, consumer sub-chunk calculations and
startup configuration checks. The split-memory self-test passed without
model or CUDA allocations. HIP and SYCL were not built or tested.

Reproduce the offline checks from the repository root:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill/offline.py
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill/test_stage_chunks.py
build/cuda-per-stage/strata-split-memory --self-test
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill/run.py
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill/analyze.py
(cd bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill && sha256sum --check --quiet SHA256SUMS)
```

`run.py` is deliberately a gated runner: it refuses to start inference when
any stage lacks the mandatory prompt-init headroom. Its retained
`mixed2048-1024/results.json` and `run-preflight.log` show that stop. No engine
log, request output, performance result or under-load monitor exists because
the engine was never started. `offline.json`, CSV/log, build/test/preflight
logs, checks, cleanup state, provenance and `SHA256SUMS` retain the completed
evidence. The runner will not overwrite `mixed2048-1024`; archive or remove
that specific output directory before a fresh preflight attempt.
