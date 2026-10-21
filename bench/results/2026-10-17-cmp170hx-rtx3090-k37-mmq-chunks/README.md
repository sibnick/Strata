# K37 MMQ: bounded chunk-size experiment

Evidence for Fossil `5f8e4f2509`, kept OPEN. This experiment changes only the
runtime-supported `--prefill` argument: 256, 384, 512. No source, default,
tile, kernel, pointer-table, model, pack, MTP or routing-profile changes.

## Measured outcome

All three chunks passed the ledger, runtime admission, requested/actual
geometry, full-residency, exact historical output-ID and recall gates.
Every request had zero prefix reuse; every measured request had zero
expert source reads. No variant was clamped or rejected. There were three
32K requests total, not an expanded long-context matrix.

| Chunk | Three short prefill times (ms) | Short median (ms) | Reduction vs first 256 | Single 32K prefill (ms / tok/s) | 32K time reduction |
|---|---|---:|---:|---:|---:|
| 256 | 2131.3, 2105.1, 2107.4 | 2107.4 | reference | 34828.3 / 941.418 | reference |
| 384 | 1830.3, 1816.7, 1804.3 | 1816.7 | 13.79% | 29861.8 / 1097.991 | 14.26% |
| 512 | 1676.3, 1684.5, 1697.9 | 1684.5 | 20.07% | 27020.4 / 1213.454 | 22.42% |
| 256 bracket | 2112.7, 2108.9, 2109.9 | 2109.9 | 0.12% slower | not run | not run |

The baseline medians differed by just 2.5 ms (0.12%) despite the observed
RTX boost-clock variation. Against the final bracket, short reductions are
13.90% / 20.16% for 384 / 512. This supports the short result without
pretending the sequential 32K checks are randomized repeat proof.
Startup was about 131..133 seconds each.

Short median decode times were 229.3 / 229.5 / 234.5 ms for 256 / 384 /
512, and short median wall-to-DONE was 2.358 / 2.059 / 1.937 seconds.
The bracket was 231.7 ms decode and 2.345 seconds wall. All arms'
measured short MTP accepted/offered pairs were 9/12, 9/13, 9/13.
For 32K, decode was 371.3 / 366.2 / 377.6 ms, wall was 35.207 / 30.269 /
27.405 seconds, and MTP accepted/offered was 18/22, 18/21, 18/22.
All short outputs had 22 IDs, all 32K outputs 37, exactly matching the
original fixtures; changing chunk size did not establish general quality
equivalence beyond these fixtures.

| Arm | Min CMP free (MiB) | Min RTX free (MiB) | Min host available (GiB) | Max engine RSS (GiB) | Max sampling gap (s) |
|---|---:|---:|---:|---:|---:|
| 256 | 922.3125 | 1917.6875 | 9.758 | 4.757 | 1.10001 |
| 384 | 920.3125 | 1883.6875 | 10.197 | 4.768 | 1.09865 |
| 512 | 862.3125 | 1849.6875 | 9.973 | 4.790 | 1.10036 |
| 256 bracket | 938.3125 | 1933.6875 | 10.420 | 4.744 | 1.10012 |

These are sampled extrema including startup, not allocation guarantees or
unobserved instantaneous peaks. Engine swap stayed zero. System swap was
already present, at most 159.25 MiB in the main arms and 152.75 MiB in the
bracket. No unrelated compute was observed. Reserve-inclusive startup
ledger margins on CMP were 496.125 / 470.125 / 426.125 MiB;
RTX margins were 1367.345 / 1341.345 / 1297.345 MiB.
Actual free-memory differences need not equal conservative scratch-price
differences because transient MMQ pools and allocator state also vary.

CMP stayed at 1470 MHz through the sampled 32K work, with phase maxima
220.04 / 219.26 / 222.83 W and 50 / 50 / 49 C for 256 / 384 / 512.
RTX 32K clock ranges were 1785..1935 / 1815..1935 / 1785..1950 MHz,
with phase maxima 193.55 / 197.64 / 200.88 W and 43 / 44 / 44 C.
Power limits stayed CMP 250 W / RTX 300 W. RTX clock-event masks included
0x4 and 0; CMP 32K samples showed 0. Raw per-phase telemetry is retained
in `summary.json`; these observations alone do not establish a bottleneck.

**Recommendation:** 512 is the best measured opt-in for these short/32K
fixtures, at +70 MiB priced scratch per stage and a sampled CMP margin of
only 222.3125 MiB over the 640 MiB stop. 384 offers a smaller +26 MiB
scratch increase and about 14% lower prefill time. Keep 256 as the
previously validated near-limit configuration; do not change the default
or extrapolate these gains, output parity or safety to a near-limit prompt.
Full 262,144-cell allocation was retained throughout, but no long-context
behavior beyond 32K was exercised in this experiment.

All four engines exited 0. All four PCIe helpers were terminated by the
harness after their arm, their PIDs are absent, and the final compute list
is empty. The runtime source diff from `97b83332` is empty. No push, merge
or PR is part of this experiment.

## Fixed conditions and provenance

The starting commit is `2fcdbd59`, descending from batching commit `97b83332`.
`offline.py` and `run.py` require no runtime diff from `97b83332`, so no
pointer-table work is inherited. The validated executable remains at the
original `sibnick-animated-robot/build/cuda-batched/strata` path, not a new
build. SHA-256:

- Engine: `aba34837fe52effa15450a85a82d6cde9af6a55c7866361ecd1f49099a5f6af9`.
- Loaded K-quant DSO: `3ba4cc2c791be319e5c010965b95c75443fb52194b52ea28c43c1fd8397335a0`.

Each arm records argv, environment overrides, build-cache hash, actual loaded
DSO path, original artifact-manifest hashes and current artifact sizes/mtimes.
Small artifact hashes are freshly checked; large model/pack files retain
their historical hashes with size/mtime provenance, not a fresh full-file
rehash. Model files are read-only. The original short and 32K ID fixtures
are hash-checked, not regenerated.

`--gpu 1,0 --layer-split 37` means physical CMP 170HX 1 is logical CUDA0,
layers [0,37), 18,944 experts / 59,192,115,200 payload bytes. Physical RTX
3090 0 is logical CUDA1, layers [37,48), 5,632 experts / 17,825,792,000
payload bytes. All 24,576 experts must pass owned-cache admission.
INT8 KV has all 262,144 cells allocated. `--spec 2`, original MTP artifacts,
direct PLE, `--prompt-cache 0`, `--no-prefill-borrow`,
`STRATA_MMQ_KQ=1`, and `STRATA_PREFILL_BATCH_GATHER=1` stay fixed.
No timing profiler or gather diagnostic counters are enabled.

## Offline allocation gate

`offline.py` compiles only the existing `src/program/split_memory_ledger.cpp`
and links it against the validated static libraries; it does not rebuild
the engine or DSO. The helper's sibling DSO is a symlink to that same
library. An initial sizing invocation without that symlink revealed a
logged FP16 fallback; those preliminary numbers were rejected, not used.
The retained ledgers require a confirmed K-quant DSO load.

The actual parser is `src/program/generate.cpp`'s `--prefill`, not a guessed
environment setting. Explicit 384 is accepted. Below 1024,
`Prefill::stream_all_min_tokens()` keeps the same routed-expert strategy
and 8-slot staging ring. No boundary override or 1024 arm is used.

| Configured chunk | Owned-prefill price, each stage | Increase vs 256 | Session/cache increase |
|---|---:|---:|---:|
| 256 | 434,110,464 B (414 MiB) | 0 | 0 |
| 384 | 461,373,440 B (440 MiB) | 27,262,976 B (26 MiB) | 0 |
| 512 | 507,510,784 B (484 MiB) | 73,400,320 B (70 MiB) | 0 |

The ledger calls the source's `Prefill::bytes_needed_owned`, not a linear
bytes/token approximation. `Alloc::take` adds/aligns its 256-byte padding
and rounds each owned take to 2 MiB pages; GEMM scratch/workspace, the shared
maximum of attention/MoE scratch, fixed dequant/group buffers, the single
ring allocation and PLE buffers are included. `bytes_needed_impl` retains
the conservative embedding-alias price and 8 MiB slack. These are allocation
estimates, not individually measured cudaMalloc sizes.

Session arenas remain 2,964,729,344 B / 1,035,491,840 B for stages 0 / 1.
Expert payload and cache indexing are unchanged. The source's primary
exact-prefill reserve includes another 64 MiB; the later-stage correction
is `(ceil(owned_bytes/MiB) + 64 - (160 + floor(chunk*680/1024))) * MiB`
when positive. Startup verifies this byte correction as well as the primary
configured chunk and buffer price. The later stage also keeps its original
293,531,596-byte MTP-bind reserve and 96 MiB verifier allowance.

Chunk-scaled host allocations in `Prefill::init` include stage 0's two
handoffs, `2*T*10240*4` B (20 / 30 / 40 MiB), and each stage's two PLE
buffers, `2*T*2560*4` B (5 / 7.5 / 10 MiB). Thus these large host buffers
increase by 15 / 30 MiB total for 384 / 512 versus 256. Each stage's
mapped routing table is `4*(3*T*10 + 2*(512+512/16+2))` B, increasing
15 / 30 KiB. Other small host token/step/routing vectors and upload
staging also scale with T; host extrema are measured rather than treating
these listed buffers as total RSS. The extra device token-ID buffer is
`4*T` B per stage (outside the conservative ledger count). No KV streaming
pool is introduced.

The reserve-inclusive K37 ledger total is 67,528,294,400 /
23,832,460,524 B at 256; 67,555,557,376 / 23,859,723,500 B at 384;
67,601,694,720 / 23,905,860,844 B at 512 (stages 0 / 1).
`run.py` compares each against freshly sampled free memory before spawning
an engine. A positive ledger margin allows startup admission testing; it
is not a fit or peak-memory guarantee. The engine's own full-residency
admission remains authoritative; no reserve is reduced to make a run fit.

## Bounded protocol and safety

Serial order is 256, 384, 512, with one warmup and at most three measured
1,977-token short requests per arm, cap 32. The baseline also runs one
original 32,788-token request, cap 192. Larger variants can run one 32K
request only after short correctness/memory gates pass and the median is
less than 5% slower than baseline. A measured short more than 10% slower
stops that arm early. Any changed output IDs stop the arm immediately,
even if the printed retrieval answer might still match. Rejected raw
outputs are retained separately from passed requests.

The permitted fourth, short-only 256 bracket follows 512 because the RTX's
sampled short-phase clocks changed from 1785..1815 MHz at the first baseline
to 1935 MHz at 384. It is exactly one warmup plus three measured short
requests, not another 32K request. `run-initial.py` preserves the exact
three-arm harness SHA-256
`05269d95ca21de106f491765f9780d09d20c7bb6503fe0a3dcff0866608e9fac`.
The final `run.py` adds only the bracket selector, output folder and
short-only schedule; its own hash is recorded in that bracket's manifest.

The per-request PP checkpoints must prove the requested chunk geometry,
not just its CLI value. The last prompt token is processed outside batched
prefill: short shapes should be 7*256+184, 5*384+56, or 3*512+440; their
batched sum is 1,976, while DONE freshness must be all 1,977 tokens.
The 32K batched sum is 32,787: 128*256+19, 85*384+147, or 64*512+19.
No prefix reuse is allowed; non-warmup source expert reads must be zero.

Both engine reserves remain 1,024 MiB. External stops are 640 MiB free
on either GPU and 5 GiB MemAvailable (4 GiB floor plus 1 GiB reaction
headroom), with at least 6 GiB before startup. Approximately one-second
samples record GPU free, power, clocks, temperature and clock-event masks,
host availability, engine RSS/swap, system swap, other compute and actual
sample gaps. The PCIe helper must stay responsive and is stopped per arm.
Power limits are not changed: CMP 250 W, RTX 300 W. Polling is not an OOM
guarantee. No resets, cache dropping, clock changes, unrelated process
termination or concurrent inference are permitted.

No near-limit, 128K or 262K prompt is run here; full-context allocation is
preserved, not fully exercised. A single sequential 32K request per arm
is descriptive controlled-condition evidence, not randomized repeat proof.

## Offline tile audit

`tile_audit.py`, `tile-audit.json` and `kq-resources.txt` inspect the current
unchanged DSO and pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (clean dependency diff).
The driver attributes are queried without creating a CUDA context; run
with `env -u CUDA_VISIBLE_DEVICES` on this host.
The resource report's trailing whitespace was normalized; engine logs
and monitor records are unedited.

The actual path is `Prefill`'s routed groups of up to 16 experts,
`mmq::Context::run`, `strata_kq_run`, then `mul_mat_q_case` and
`mul_mat_q_switch_J`. Gate/up is [1280,2560], down [2560,640].
Both output row counts are divisible by 128, so `fallback=false`.
`ncols_opt=ncols_max=max_rows`, the maximum routed token count of an
expert in that group, NOT the prompt chunk or its T*10 routing rows.
Batching the gather does not change this selection.

On sm80 and sm86, Q4_K/Q5_K/Q6_K use Ampere configs: I=128, 256
threads, K=256, stream-K enabled, target occupancy 1. Available J values
are 8,16,24,32,40,48,64,80,96,112,128. The selector scans ascending J,
rejects shared-memory-infeasible configs, minimizes ceil(max_rows/J),
keeps the first tie, and stops at one tile. For max_rows <=128 it picks
the smallest available J covering max_rows. Above 128 the first-tie rule
matters; the exact conditional map through 512 is retained in JSON.
There is no observed max_rows histogram from these counters-OFF runs,
so this is not an observed launch-frequency distribution.

The source's dynamic shared bytes are
`128*76*4 + 4*J + round_up(144*J,1024)`: 40,992 B at J8,
44,160 B at J32 and 57,856 B at J128. All fit both GPUs' per-block
opt-in limits (CMP 166,912 B, RTX 101,376 B). Queried CMP has 74 SMs,
167,936 shared B/SM and 65,536 registers/SM; RTX has 82 SMs,
102,400 shared B/SM and the same registers/SM.

Compiled sm80 Q4_K uses 102 / 128 / 169 / 205 registers/thread at
J8 / 16 / 24 / 32; Q5_K uses 107 / 148 / 173 / 204 and Q6_K
118 / 155 / 192 / 229. These provide at most two register-limited
256-thread CTAs at Q4 J8/16, Q5 J8 and Q6 J8, and at most one
at the larger listed J values. Actual occupancy may be lower; registers
allocate in hardware granules. Larger variants reach 255 registers and
some have nonzero stack sizes. Stack alone is not proof of spill traffic.
The raw reports retain sm86 as well.

Stream-K additionally chooses either the tile count when wave efficiency
is at least 90%, or the SM count, with a fixup kernel when needed. Its
temporary pool can grow with selected J and is not a permanent
chunk-proportional cache allocation. Runtime memory sampling covers this
path; peak FLOPS do not establish its bottleneck.

A separate, narrowly grounded tile candidate would compare I64/J32
against existing I128/J32 for Q4_K groups with max_rows 25..32: the current
205 registers/thread constrain the upper bound to one CTA/SM, and halving
I reduces accumulator extent but doubles output-row tiles. This is a
resource-motivated hypothesis, not a speed claim or a selected engine
change; first establish whether those groups materially occur. No tile
implementation, rebuild sweep or kernel tuning is authorized by this run.

## Reproduction and evidence files

- `offline.py`, `ledger-*.csv`, `ledger-*.log`, `offline.json`,
  `sizing-build.log`: source sizing, identity and compile/link commands.
- `run.py`, `chunk*/results.json`, `engine.log`, `monitor.jsonl`, `pcie.log`
  and per-request JSON: immutable run manifests and raw evidence.
- `test_chunks.py`, `cpu-tests.log`: bounded CPU fixture, output, shape,
  startup, identity, pricing and guard contract checks.
- `analyze.py`, `summary.json`, `checks.json`, `cleanup.json`: offline
  comparisons, measured safety extrema, and final process cleanup.
- `SHA256SUMS`: every retained evidence file except itself and Python cache.

The experiment runner refuses existing arm directories to avoid overwriting
measurements. Run it only with explicit hardware authorization and after
reviewing the reserve-inclusive ledgers. The default engine behavior is
byte-identical to `97b83332`: only evidence files change in Git.

Completed checks (paths below are relative to the repository root):

```text
python bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks/offline.py
  passed: all three source-based native-MMQ ledgers
env -u CUDA_VISIBLE_DEVICES python bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks/tile_audit.py
  passed: read-only source/resource/driver-attribute audit
python bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks/test_chunks.py
  passed: 8 bounded CPU contract checks
python bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks/analyze.py
  passed: bounded schedules, fixed flags/artifacts, harness hashes, guards and cleanup
```

No CUDA/HIP/SYCL engine rebuild was needed or performed: no runtime file
changed. The temporary sizing executable, object and DSO symlink were
removed after collecting their provenance.
