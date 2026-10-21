# K37 MMQ: chunks 768 and 1024 with authorized GPU reserve removal

Both requested chunks passed unchanged runtime admission, full residency,
actual chunk geometry and the bounded short/32K requests. Neither was
clamped. Chunk 1024 was fastest here, but its sampled CMP free memory fell
to **382.3125 MiB**, below the earlier experiment's 640 MiB stop.
This is not evidence that 1024 is safe for near-limit inputs.

Evidence for Fossil `5f8e4f2509`, kept OPEN. Only evidence files change;
no runtime, default, kernel, tile, pointer-table or artifact changes.

## Authorization and retained checks

The request was "please relax all memory limits and check 768 and 1024".
The explicit clarification was "Yes--remove discretionary GPU limits;
retain host 5 GiB and mandatory allocation checks" (the original wording
is retained in `authorization.json`).

For this experiment only, both `--vram-reserve-mib` and
`--vram-reserve-later-mib` are **0**, instead of 1024 MiB. The external
free-VRAM abort threshold on both GPUs is disabled, instead of 640 MiB.
Free VRAM is still sampled. The user accepted the increased GPU OOM risk.
These settings are recorded in each arm's argv/guards and are not launch
defaults or a recommendation to remove protections generally.

Host startup still requires MemAvailable >=6 GiB; the monitor stops below
5 GiB (4 GiB floor plus 1 GiB reaction headroom). Resident headroom and
pinning protections stay at 4 GiB. Mandatory CUDA allocation, full-cache
admission, exact-prefill pricing, MTP/verifier reservations, allocator
margins and serve's 512 MiB `own_fits` headroom remain unchanged.
The parser accepts reserve 0 directly; nothing was patched out.

No reset, cache dropping, clock/power change, unrelated process termination,
offload, allocation workaround or retry was used. Runs were serial; no
unrelated compute was observed. Polling cannot guarantee avoidance of OOM.

## Measured outcome

One original 1,977-token cap-32 warmup and three measured repeats per arm,
then one original 32,788-token cap-192 request per arm:

| Chunk | Three short prefill times (ms) | Short median (ms / tok/s) | Single 32K prefill (ms / tok/s) |
|---|---|---:|---:|
| 768 | 1511.2, 1500.4, 1499.8 | 1500.4 / 1317.649 | 22533.4 / 1455.084 |
| 1024 | 1368.6, 1360.8, 1369.0 | 1368.6 / 1444.542 | 20237.2 / 1620.185 |
| Historical 512, different GPU safety | 1676.3, 1684.5, 1697.9 | 1684.5 / 1173.642 | 27020.4 / 1213.454 |

Against historical 512, short prefill time fell 10.93% / 18.75%, and
single-32K prefill time fell 16.61% / 25.10%, for 768 / 1024.
Within this experiment 1024 took 8.78% less short prefill time and 10.19%
less 32K prefill time than 768. These are time reductions, not percentages
of throughput increase.

The comparison to 512 is historical and under different discretionary GPU
reserves and external safety. No 512/256 rerun or extra baseline was made.
Arm order was 768 then 1024, not randomized. Each 32K result is a single
descriptive observation, not repeated statistical proof. RTX boost clocks
also varied. Moreover, 1024 crosses an execution-strategy boundary: the
gain cannot be attributed solely to batch size.

Short warmup prefill was 1702.2 / 1566.7 ms and startup 132.409 / 133.049 s.
Measured short median decode was 228.5 / 208.4 ms and median wall-to-DONE
1.7407 / 1.5804 s. Short MTP accepted/offered was 9/12, 9/13, 9/13 at
768, versus 10/11 on all three 1024 repeats.
For 32K, decode was 371.9 / 388.2 ms, wall 22.9754 / 20.6323 s, and
MTP accepted/offered 18/22 / 18/23. Thus faster prefill did not imply
faster 32K decode; MTP trajectories changed despite identical output IDs.

Every warmup/measured short output matched the original 22 token IDs and
retrieved all three numeric needles. Both 32K outputs matched the original
37 IDs and all three text needles. DONE reported all input tokens fresh,
zero reused, for every request. This establishes fixture parity, not general
quality equivalence across changed routing/chunk arithmetic.

All eight measured requests reported `0.0 MB of expert reads`. Warmup
counters include the startup cache fill (77,024.6 MB); they cannot isolate
warmup-only expert reads. Measured counters reset after that first request.
Nonzero OS storage reads remain, including direct PLE reads; zero expert
source reads does not mean zero file I/O.

## Actual geometry and the 1024 boundary

Startup confirms the requested primary scratch price and later-stage exact
scratch correction, logical-device layer mapping, all 24,576 resident
experts, loaded K-quant DSO, and `READY 262144 stop`. No fallback chunk
message occurred. Actual PP checkpoints, not just argv, prove:

| Chunk | Short batched shape / count | 32K batched shape / count | Source-selected strategy |
|---|---|---|---|
| 768 | 768, 768, 440 / 3 | 42*768 + 531 / 43 | Routed MMQ with batched gather |
| 1024 | 1024, 952 / 2 | 32*1024 + 19 / 33 | Full 1024 chunks: resident-filtered `stream_all`; tails: routed MMQ |

Batched sums are 1,976 / 32,787; the final input token is processed outside
batched prefill. DONE freshness is the full 1,977 / 32,788.

`src/prefill/prefill.cpp` defaults `stream_all_min()` to 1024. With this
layer split, source pointer and ring override, effective rings are 8 slots
below that boundary and 96 at/above it. Serve's startup "streamed ring
96 slots" configuration line also appears at 768; it does not override
the small-chunk 8-slot rule. No `STRATA_PREFILL_STREAM_MIN`,
`STRATA_PREFILL_RING` or `STRATA_PF_FUSED` override is set.

At 1024, `stream_all` requires ring >8, actual T>=1024 and a source pointer.
Its source plan skips experts already resident in the owning stage cache
before transient/blob source lookup. With complete residency there are no
expert-stream staging jobs; the compute walk gets resident cache pointers.
Measured zero expert reads support preservation of this intended behavior.

Grouping uses `(stream_all || batch_gather)`; at 1024 full chunks therefore
use the preexisting stream-all grouping path. Shared-expert scheduling also
changes: `defer_shared = !stream_all && stream_ahead_enabled()` is false on
those full chunks, so shared work runs earlier. Native fused IQ experts
remain off in this CUDA build with the original unset environment.
Native MMQ products and tiles are unchanged.

The per-chunk strategy labels are source-derived from these predicates,
confirmed configuration/residency and actual PP shapes, not a newly enabled
profiler or branch trace. Timing profiler and gather counters stayed OFF.
Source excerpts/hashes are in `provenance.json`; no tile experiment or new
resource/bottleneck claim is made here. The earlier offline tile audit in
`../2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks/` remains separate.

## Source-exact offline sizing and admission

`offline.py` compiles only the existing split-memory-ledger program and
links validated static libraries. It does not rebuild the runtime or DSO.
Its sibling K-quant DSO symlink points to the same unchanged library and
both retained sizing logs confirm that load, not an FP16 fallback.

| Chunk | Owned-prefill price, each stage | Increase vs 512 | Session/cache increase |
|---|---:|---:|---:|
| 512 reference | 507,510,784 B (484 MiB) | 0 | 0 |
| 768 | 572,522,496 B (546 MiB) | 65,011,712 B (62 MiB) | 0 |
| 1024 | 1,025,507,328 B (978 MiB) | 517,996,544 B (494 MiB) | 0 |

Compared with 256, the increases are 132 / 564 MiB per stage. The 1024 jump
includes the 96-slot ring rather than 8 slots; the maximum expert blob is
3,993,600 B. Pricing calls `Prefill::bytes_needed_owned`, retaining padded
allocations rounded to 2 MiB, GEMM workspace, max-attention/MoE region,
fixed dequant/group buffers, PLE, conservative embedding alias and 8 MiB
slack. These are source allocation estimates, not per-cudaMalloc traces.

Session arenas stay 2,964,729,344 / 1,035,491,840 B for stages 0 / 1;
expert payload/cache indexing and full-context INT8 state are unchanged.
The extra device token-ID buffer outside the ledger is 3,072 / 4,096 B
per stage at 768 / 1024.

Large chunk-scaled host buffers are stage 0's two handoffs
`2*T*10240*4` B (60 / 80 MiB) and each stage's two PLE buffers
`2*T*2560*4` B (15 / 20 MiB). Combined these are 90 / 120 MiB,
30 / 60 MiB more than 512. Each stage's mapped routing table is
`4*(3*T*10 + 2*(512+512/16+2))` B: 96,528 / 127,248 B, a
30 / 60 KiB increase versus 512. Smaller vectors/upload staging also
scale; these listed buffers are not total RSS.

The primary exact-prefill guard retains its extra 64 MiB. For stage 1,
the conservative rule is `160 + floor(T*680/1024)` MiB plus a rounded
366 MiB ring reserve when T>=1024. Correction
`max(0, owned_bytes + 64MiB - ruleMiB*MiB)` is zero for both tested chunks.
Both startup logs confirm zero correction and the unchanged 293,531,596 B
MTP-bind reserve and 96 MiB verifier allowance.

Raw ledgers retain their fixed 1 GiB `additional_cuda_and_safety_reserve`.
For the authorized zero-discretionary-reserve preflight only, the harness
subtracts that fixed allowance and nothing else; it does not modify the
ledger or any runtime admission rule. Raw and adjusted totals are retained:

| Chunk | Raw total, stages 0 / 1 (B) | Adjusted total, stages 0 / 1 (B) | Pre-start margin CMP / RTX (MiB) |
|---|---|---|---|
| 768 | 67,666,706,432 / 23,970,872,556 | 66,592,964,608 / 22,897,130,732 | 1388.125 / 2259.345 |
| 1024 | 68,119,691,264 / 24,423,857,388 | 67,045,949,440 / 23,350,115,564 | 956.125 / 1827.345 |

Positive preflight margins allowed startup testing; unchanged runtime checks
remained authoritative. Both arms admitted all experts without offload.
No allocation/admission rejection, clamp, OOM or CUDA error occurred.
**1024 did emit the retained low-VRAM warning at 480 MiB before its first
request**, below the runtime's 512 MiB warning level. This warning was not
hidden or treated as an allocation failure.

## Sampled safety and cleanup

| Chunk | Min CMP free (MiB) | Min RTX free (MiB) | Min host available (GiB) | Max engine RSS (GiB) | Max sample gap (s) |
|---|---:|---:|---:|---:|---:|
| 768 | 804.3125 | 1791.6875 | 11.7915 | 4.8257 | 1.09942 |
| 1024 | 382.3125 | 1361.6875 | 11.9565 | 4.8479 | 1.09980 |

Host availability in the first monitored startup sample was 16.5305 /
16.6764 GiB. Engine swap stayed zero; preexisting system swap peaked at
131.75 MiB. Raw samples include free GPU memory despite no GPU abort
threshold, host availability/RSS/swap, temperatures, power, clocks,
clock-event masks, unrelated compute and actual gaps. PCIe logging was
responsive. These are sampled extrema, not unobserved allocation peaks.

During 32K, CMP clocks were 1470 MHz, phase maxima 219.35 / 224.28 W and
47 C, clock-event mask 0. RTX clock ranges were 1770..1950 / 1800..1950 MHz,
power maxima 190.23 / 191.35 W, temperatures 42 / 43 C; masks included
0 and 0x4. Short measured CMP clocks ranged 1485 / 1470..1485 MHz and
RTX 1830..1935 / 1770..1950 MHz. Power limits stayed CMP 250 W / RTX 300 W.
Phase telemetry in `summary.json` preserves ranges without inferring a
GPU bottleneck from these observations.

Both engines and both PCIe helpers exited 0. Engine PIDs 855669 / 867257
and helper PIDs 855667 / 867255 are absent; final compute is empty.
Idle free memory returned to CMP 64897 MiB / RTX 24096 MiB without reset.
The sizing helper/object/DSO symlink were removed after use.

**Recommendation:** 1024 is the fastest measured option for these fixtures
under the explicitly relaxed GPU protections, not a safe default:
only 382 MiB CMP free was sampled. 768 retains the below-boundary strategy
and about 804 MiB sampled CMP free while improving on historical 512.
Keep the earlier protected/near-limit conclusions separate; do not
extrapolate output parity, gains or memory safety to 128K/near-limit inputs.
Full 262,144-cell allocation was preserved, not exercised by a 262K prompt.
Fossil stays OPEN for these untested limits; this bounded experiment stops here.

## Identity, commands and retained evidence

This continues evidence commit `fe6a6ea6a2079ca40b7325d2f2efe590fdf1d16b`,
descending from clean batching `97b83332` and long validation `2fcdbd59`.
Runtime diff from `97b83332` across `src`, `include`, `CMakeLists.txt` is
empty. No pointer-table work is inherited.

The binary remains `sibnick-animated-robot/build/cuda-batched/strata`.
SHA-256 engine:
`aba34837fe52effa15450a85a82d6cde9af6a55c7866361ecd1f49099a5f6af9`;
actual loaded sibling `libstrata_kq.so`:
`3ba4cc2c791be319e5c010965b95c75443fb52194b52ea28c43c1fd8397335a0`.
Each results manifest records loaded DSO path, build-cache hash, argv,
environment, fixture hashes and historical artifact hashes with fresh
size/mtime checks. Small files are rehashed; large shards/pack files are
not fully rehashed again. Artifacts were not modified or downloaded.

`--gpu 1,0 --layer-split 37`: physical CMP 1/logical CUDA0 owns [0,37),
18,944 experts / 59,192,115,200 B; physical RTX 0/logical CUDA1 owns [37,48),
5,632 experts / 17,825,792,000 B. Original model/pack/MTP/profile, `--spec 2`,
direct PLE, no prefill borrowing, prompt cache 0, `STRATA_MMQ_KQ=1`,
`STRATA_PREFILL_BATCH_GATHER=1` and all other runtime settings are unchanged.
Inherited `STRATA_*`, `CUDA_VISIBLE_DEVICES`, `LD_PRELOAD` are removed before
applying the recorded original environment overrides.

Commands from the repository root (each result below passed):

```sh
D=bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks768-1024
python "$D/offline.py"
python "$D/test_variants.py"
python -u "$D/run.py" --chunk 768 > "$D/progress768.log" 2>&1
python -u "$D/run.py" --chunk 1024 > "$D/progress1024.log" 2>&1
python -m py_compile "$D"/*.py
python "$D/analyze.py"
(cd "$D" && sha256sum --check --quiet SHA256SUMS)
```

Sizing passed both chunks with the native K-quant library. Five CPU tests
passed: unchanged identities/source, GPU-only limit removal/host guard,
immutable fixtures/geometry/IDs/freshness, exact sizing, and reserve-zero
startup/clamp rejection. The offline analyzer rechecks raw output records,
counters, authorization/argv scope, actual shapes and historical IDs, loaded
DSO/artifact identities, sampling guards and cleanup. Syntax and checksum
checks passed. No runtime/backend rebuild was needed; no extra GPU work
was launched for validation.

`run.py` is preserved byte-identical to both recorded harness hashes.
Its inherited regression-stop messages sometimes use "baseline" or
"contemporaneous"; the actual code/reference is historical 512, not a
contemporaneous control. Neither stop fired.
Raw logs, monitors, outputs, counters and manifests remain in each arm
directory. `summary.json`, `checks.json`, `cleanup.json`, `provenance.json`
and `SHA256SUMS` are offline products. Evidence is committed separately;
no push, merge or pull request is part of this task.
