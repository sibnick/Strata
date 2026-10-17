# Upstream merge impact: K36 2048/512

## Scope and result

This report covers merging `origin/main` at `477ac13cbe8e582c20484871ce459d20d82df09d` into
`sibnick-per-stage-prefill-chunk`. The pre-merge branch is preserved at
`backup/sibnick-per-stage-prefill-chunk-before-origin-main-1319480d` (commit `1319480d`); the previously tested
K36 implementation baseline is `ef991b2f`.

The merge keeps the explicit K36 split (`--gpu 1,0 --layer-split 36`), local per-stage chunks (`2048,512`),
GPU-owned expert accounting, batched gather, PLE timing, and the existing default-off behavior of the local
features. It does not change or restart the running API. The failed Nsight capture remains accurately recorded;
there is no usable kernel timeline and no optimization claim.

## Conflict decisions

- `src/kernels/ngram.cpp`: retained local opt-in PLE timing and combined it with upstream PLE table locking,
  reordered/duplicate-prefetch matching, and draining outstanding reads on error.
- `src/prefill/prefill.cpp`: retained per-stage handoff sizing and timing while integrating upstream pipeline
  chunk selection, handoff events/workers, and grouped resident-expert gathering with the local batch-gather
  condition. Kept one shared gather plan for the main walk. The merged 32-slot short-chunk staging ring is included
  in the scratch estimate.
- `src/program/generate.cpp`: retained per-stage chunk parsing and GPU-owned allocation/admission; integrated
  upstream `--prefill-pipe` and primary-device split checks. The owned-expert ledger now includes the same
  per-slot guard bytes as the direct-MMQ slot allocator. Its default-zero behavior is unchanged; a CPU test covers
  guarded rounding and overflow.

## Ranked effects for this machine

| Rank | Change | Expected K36 effect | Evidence and limit |
|---|---|---|---|
| 1 | Short-chunk resident grouped gather now reserves `max(ring_slots, 32)` expert slots. The K36 RTX stage uses 512 tokens, where the old ring had 8 slots. | The staged walk can keep more grouped experts ahead and may reduce gather/wait bubbles, at the cost of 94,371,840 B (90 MiB) more scratch on the RTX 3090. The old offline estimate of 507,510,784 B becomes 601,882,624 B. | `native_experts.txt` gives a 3,993,600 B maximum blob; the old and new single-allocation sizes round to 2 MiB pages. Applying this delta to the earlier idle snapshot leaves 126,452,500 B (about 120.6 MiB) after the mandatory 512 MiB prompt-init headroom. This is a source-based estimate, not a merged CUDA startup or live admission test. The CMP 2048-token ring remains 96 slots and its scratch is unchanged. |
| 2 | K-quant prefill now uses `quantize_scatter` by default when `STRATA_QUANT_GATHER` is unset. | The K36 run uses `STRATA_MMQ_KQ=1`, so this changes its prompt quantization schedule. The new path is intended to quantize each token once instead of once per routed expert; `STRATA_QUANT_GATHER=1` selects the former gather schedule. | The source describes the output rows as the same bytes. No model-level parity or performance run was allowed after the merge, so neither quality parity nor speed on the 170HX/3090 is established here. This is separate from pointer-table MMQ. |
| 3 | Serving now applies model-card sampling when callers omit sampling, and caps unbudgeted thinking early for sufficiently large `max_tokens` to reserve answer room. | API responses and reasoning/output token allocation can differ after deploying the merged server, even without changing the model or engine flags. | The checked-in API config has no explicit sampling block. The already-running server remains the old process and is unchanged. CPU mock tests cover the new defaults; no real API request was sent. |
| 4 | Wide dequantization is selected by default on compute capability 8.6 for supported quant types and aligned buffers. | This can change an RTX 3090 operation if it reaches `dequant_bf16`; the CMP 170HX (sm80) does not take this new default. | Upstream reports bitwise output and short-prompt gains on RTX 3060/A4000, not this 3090. Applicability to the K36 hot path and local performance are unmeasured. |
| 5 | PLE table operations are synchronized; prefetch matching handles reordered and duplicate keys. | Improves concurrent table safety and match reuse; the mutex can serialize concurrent PLE table operations. | The local timing instrumentation remains opt-in and PLE settings are unchanged. No GPU or concurrent-request comparison was run. |

## Compatibility and gated features

- `STRATA_MMQ_TAB` was explicitly configured `OFF`; the pointer-table kernels and their stated ~37 MiB per-GPU
  load are not built. `STRATA_MMQ_DIRECT` remains unset, so the runtime is intended to keep gathered MMQ. The
  pinned llama.cpp commit remains `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The previously reported roughly
  50% pointer-table-related regression with its flag off was not reproduced or disproved on this build; default-off
  source gating and a successful compile are not performance evidence.
- The chosen layer boundary is explicit, so upstream automatic split placement does not select a different K36.
  `--prefill-pipe`, `--pipeline-windows`, and `STRATA_SPLIT_OVERLAP` are not enabled by the API config; no
  overlap behavior is assumed. The config also has no `parallel` setting or `--batch`, so request batching stays
  off. The hardware has no P2P route, so the new peer-copy capability probe should not enter its copy-test path
  for this pair; merged startup logging was not collected.
- The existing full INT8 KV and MTP/spec settings were not changed. Upstream's extra MTP prompt reservation and
  QSA split-attention paths require opt-in flags that are absent from the config. The merged source does not
  establish runtime admission, output parity, MTP acceptance, or quality on this model.

## Validation and limits

- CPU ownership test: `c++ -std=c++20 -Wall -Wextra -Werror -Iinclude tests/program/gpu_ownership_test.cpp -o /tmp/strata-gpu-ownership-test && /tmp/strata-gpu-ownership-test` — passed.
- Merged-build self-tests: `build/merge-origin-main-cuda/gpu_ownership_test` and
  `build/merge-origin-main-cuda/strata-split-memory --self-test` — both passed; the sizing self-test allocates no CUDA memory.
- Per-stage chunk contract tests: `python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/test_stage_chunks.py` — 5 passed.
- Serving tests: `python3 -B -m unittest serve.test_model_card_sampling serve.test_admit_order serve.test_effort_paths serve.test_parallel` — 52 passed. The run emitted `ResourceWarning`s from `telemetry.py` for unclosed `/proc/stat` and `/proc/cpuinfo` reads.
- CUDA build: `cmake --build build/merge-origin-main-cuda --parallel 2` — passed all configured CUDA and test targets. The isolated Release configuration used CUDA 13.4.92, `CMAKE_CUDA_ARCHITECTURES=80;86`, `STRATA_MMQ_TAB=OFF`, `STRATA_MMQ_KQUANTS=ON`, and the pinned llama.cpp source. NVCC reported unused local variables/functions in `src/qwen35/q36_engine.cu`; the C++ compiler reported an unused `w` in `src/program/generate.cpp`. These were warnings; the build completed.
- No merged-source inference, prompt parity, full CUDA startup admission, or performance test was run. The offline admission margin above uses the pre-merge idle-GPU snapshot and exact scratch delta; it is narrow enough that a merged startup-ledger check should precede any inference. The full ledger sizing path initializes CUDA, so it was not run against the loaded API; only its `--self-test` mode ran, which allocates neither a model nor CUDA memory.
- HIP and SYCL were not built or tested. The running API at `127.0.0.1:8080` remains loaded and healthy on its original binary; no request was sent during this work.
