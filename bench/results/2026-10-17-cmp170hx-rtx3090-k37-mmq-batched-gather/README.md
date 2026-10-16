# K37 MMQ: batching resident expert gathers

Local date 2026-10-17, Linux, CUDA 13.4.92, CMP 170HX 64 GiB + RTX 3090 24 GiB.
This is a bounded short-prompt experiment, not a long-context or placement test.

**Default-off batching reduced median prefill time by11.9-13.3% on the three
original fixtures, with identical historical output IDs and no guard violation.**
All30 requests across seven arms passed. Source, focused tests and raw evidence
are retained here; defaults remain unchanged.

## Measured result

Same candidate executable/DSO, no trace, timing profiler or gather counters.
Warmup is excluded; each cell is the median of three fresh-input requests.

| Input tokens | OFF prefill ms | ON prefill ms | Time reduction | Prefill speedup | OFF / ON wall to DONE s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 502 | 681.9 | 600.8 | 11.89% | 1.135x | 0.9091 / 0.8261 |
| 994 | 1242.3 | 1082.1 | 12.90% | 1.148x | 1.4808 / 1.3101 |
| 1977 | 2426.2 | 2103.8 | 13.29% | 1.153x | 2.6728 / 2.3394 |

Execution order: preserved1977, candidate OFF1977, ON1977; after that gate passed,
ON502/994 followed by OFF502/994 (reversed order); then separate counters OFF/ON1977.
Every arm has its own warmup. This is sequential bounded A/B with reversed smaller
arms, **not randomized/interleaved repeated independent process pairs**.
Startup137.7-141.7s per performance arm is separate from request timing.
Raw repetitions, decode, MTP and wall times are in `summary.json` and each arm.
The small number of short requests does not establish general significance or
predict other hardware, models, workloads, splits or long-context behavior.

The preserved unpatched executable (`cc8d0ca2...`) measured2389.5ms median1977-token
prefill (2386.9 /2390.8 /2389.5). Candidate OFF measured2426.2ms
(2437.6 /2410.1 /2426.2), **1.54% slower**, not the separate pointer-table build's
approximately50% regression. Candidate ON was2103.8ms (2109.9 /2103.8 /2083.8).
The same-binary OFF/ON comparison is the optimization denominator; preserved vs
OFF retains build-layout/time-of-run uncertainty. A redundant third clean-executable
performance arm was omitted: fresh OFF directly measures the old schedule against
the preserved binary, and clean/candidate/preserved K-quant SASS is identical.

Median decode OFF /ON:502 tokens224.0 /223.2ms,994 tokens231.4 /226.8ms,
1977 tokens234.4 /233.9ms. **No decode speedup claim** is supported by these
short windows. MTP accepted/offered per1977 repetition is9/12,9/13,9/14 in both
arms; the994 ON middle request is9/12 versus OFF9/13. All output IDs still match.

Separate counter-enabled1977-token requests provide mechanism evidence, not speed
evidence (the1976-token batched portion excludes the final input seed token):

| Owning stage | OFF gathers | ON gathers | Experts copied, both arms | Destination bytes, both arms | Alignment fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMP, logical0 | 83589 | 5361 | 83589 | 261412134912 | 0 |
| RTX, logical1 | 24489 | 1573 | 24489 | 77328470016 | 0 |
| Total | 108078 | 6934 | 108078 | 338740604928 | 0 |

**93.58% fewer actual gather launches**, copying exactly the same experts and
315.477GiB of repeated destination bytes, including padding. These are device-local
copies of resident weights, not315GiB of host/PCIe traffic, not unique model bytes,
and not removed numerical dequantization. Launch count falls far more than total
prefill time; MMQ products, weight-copy bytes and the rest of the engine remain.
The old profiled "dequant" label included gather/copy plus host-launch gaps and
must not be read as all removable arithmetic. No intrusive profile supplied the
performance numbers above. `checks.json` verifies experts/bytes equal on each GPU
and the expected per-expert OFF count; all30 exact outputs, residency, guard
samples, binary identity and clean exits are independently rechecked.

Across all arms, sampled minima: **CMP936.3125MiB**, **RTX1933.6875MiB** free;
host available**11.299GiB**. Max engine RSS**4.748GiB**, engine swap0; system swap
was132.25MiB (already used at the start). Maximum sample gap1.1007s.
No unrelated compute, guard trip, OOM or model fallback was observed.
Power limits stayedCMP250W /RTX300W. Performance-request samples show CMP
SM1470-1485MHz,35-41C,73.27-218.59W across arms;
RTX1785-1950MHz,35-41C,111.90-190.37W. These include
request-boundary/decode observations, not isolated kernel power measurements.
Raw event-reason masks and per-arm bounds are retained rather than assuming
there was no clock limiting. `cleanup.json` records final no-compute proof.

## Source and build provenance

Parent `3480a3be887494ad79c7400bcd747b44dc97f3ee` is the clean K35 evidence branch,
before the separate in-place MMQ implementation. Its runtime/build source diff
against historical `0327ba27` is empty. No code, patched header or CMake header-copy
logic from `sibnick-in-place-k37-mmq` was imported.

The pristine llama.cpp dependency is commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. A read-only copy of the preserved
unpatched build's dependency was copied into this worktree's ignored
`build/llama-pristine`; its tracked diff is empty. `mmq.cuh` SHA256 is
`50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b`.
No dependency or model download was needed.

A clean CUDA Release SM80+86 executable was built **before** runtime edits and
saved as `build/clean-unpatched/strata`. The candidate changes only the host gather
schedule in `prefill.cpp`; `moe_mmq.cu`, its header, K-quant ABI/loader/library source,
all MMQ math and llama.cpp headers remain unchanged. `build-identity.json` records
the commands, hashes, toolchain and corrected build/test attempts.

The whole DSO hashes differ between build locations and after CMake reconfiguration
(ggml's version-stamp object was rebuilt). This is not evidence of different GPU
instructions: `verify-build.py` compares **1,534,144 nonempty SASS lines**, retaining
every function name, instruction address, predicate, operand and encoding. Only
cuobjdump container metadata is omitted. Preserved, freshly built clean and candidate
K-quant DSOs all hash to
`74fd8cec289c5932582625ee8b13d7bbe3cdedcb4463b7af158d9ebb38592f56`.
The actual loaded DSO path is independently checked in `/proc/PID/maps` in each arm.
We do not claim identical whole executable bytes or zero host-code layout effects.

## Implementation and prior work

Open issues/PRs were searched before editing. Existing
[#1107](https://github.com/Niko1221/Strata/pull/1107) covers grouping on short prompts
by enlarging staging from eight to 32 slots; it documents an earlier undercounted
allocation bug. This experiment does **not** import that allocation change.
[#1822](https://github.com/Niko1221/Strata/pull/1822) is the separate patched-header
in-place approach, not this baseline or candidate. No GitHub thread was modified.

`STRATA_PREFILL_BATCH_GATHER=1` enables the existing `GatherGroup` helper for the
routed native-MMQ walk as well as the already-grouped streamed walk. Sixteen resident
experts share one gather into the same contiguous MMQ buffers. Each staged expert
immediately flushes the open group, waits for its copy and records its release event
**before the next iteration can refill its source slot**. Thus mixed residency is
safe without increasing the eight-slot ring or holding staged slots until expert16.
Resident runs interspersed with staged experts can still share launches.

A forced flush advances `gg.first` without resetting the MMQ slot index. Only the
final flush zeros the original 4096-byte padding after both group buffers; a final
partial group uses its actual last slot. Unaligned group inputs use the old
per-expert helper. Existing copy-stream ordering and streamed-walk `used_of` /
`give_back` logic are retained. `STRATA_PREFILL_GROUP_GATHER=0` disables grouping.
Unset/zero batching uses the previous path. No cache, duplicate weights, extra
allocation, route ownership, chunk threshold, tile tuning or model change.

Fused, non-native, FP16, CPU and peer/helper gather paths are not extended. Local
native-MMQ work still groups independently on each owning GPU. HIP explicitly
declines the new opt-in; SYCL has a separate path. HIP/SYCL compilers are not
installed here, so neither backend was built or tested.

`STRATA_PREFILL_GATHER_COUNTS=1` is a separate diagnostic: per invocation/device,
actual routed native gather kernels issued, experts copied, destination bytes
(weights + optional scales + padding), and failed group-alignment attempts.
It does not count MMQ products, peer/helper kernels, memsets as gather kernels,
or unique source bytes. It adds no GPU event, synchronization or atomic; counters
are neither updated nor printed when disabled. Counter-enabled arms are excluded
from performance comparisons.

## Configuration, correctness and safety

Exact commands/environments are copied from the parser-verified original
`../2026-10-16-cmp170hx-rtx3090-mmq-640/mmq/results.json`, not reconstructed from
prose. Only executable, batching toggle and separately requested counters differ.
`STRATA_TRACE` is removed from **all** arms; no phase-timing profiler is enabled.
The same candidate binary is used OFF and ON. MTP/spec2 remains enabled.

Physical GPU1 CMP is logical CUDA0, layers [0,37); physical GPU0 RTX is logical
CUDA1, layers [37,48). Startup must report 18,944 / 5,632 owned experts and exactly
59,192,115,200 / 17,825,792,000 resident bytes, all24,576 resident, fixed chunk256,
INT8 context capacity262,144 and no prompt borrowing. Both allocation reserves
remain1024 MiB. UUIDs, startup ownership/admission reports and loaded DSO are checked
before prompts. No model, pack, MTP or shard is written. Large artifact identities
reuse the original hashes with current size/mtime; small metadata is freshly hashed.

Original committed502 /994 /1977-token natural-text exact-retrieval fixtures and
ID hashes are read directly, not regenerated. Cap32. Every request must report
zero prefix reuse, the full input freshly read, completed stop below cap32, and
the **exact historical22 output IDs**, including EOS, for:

```text
ORIGIN: 714
CENTER: 928
FINAL: 365
```

Warmup includes startup-cumulative expert I/O; each measured request must report
0.0 MB expert reads. Exact IDs are stronger than merely checking decoded recall,
but are not a broad numerical quality evaluation or a full session-state hash.
The existing state-hash switch needs prompt caching enabled, which would change
the required configuration, so it was not used.

Only one workload is active. Before each engine startup, no compute process may
be present and host MemAvailable must be at least6 GiB. During startup and inference,
NVML guards both GPUs at640 binary MiB; host reaction threshold5 GiB retains the
4 GiB floor +1 GiB reaction margin. Approximately1-second samples retain free
VRAM, host RSS/swap, system available/swap, GPU power/limits/clocks/temperature/
throttle bits, phase and sample gap. Other compute appearing triggers stop of
**our own process only**. Each arm sends EOF and verifies exit0/no compute.
No resets, clock/power changes, cache drops or changes to other processes.
Polling extrema are sampled, not guaranteed peaks or an OOM guarantee.

## Reproduction

From repository root, build with the exact configuration in `build-identity.json`
using a pristine local llama.cpp source, then:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather -p test_batch.py -v
cmake --build build/cuda-batched --target strata prefill_gather_test prefill_mmq_kquant_test -j 3
env -u CUDA_VISIBLE_DEVICES /usr/bin/ctest --test-dir build/cuda-batched -R '^(prefill_gather_test|prefill_mmq_kquant_test)$' --output-on-failure
```

CPU tests cover immutable fixtures/exact output IDs, admission/mapping and both
hard guards, plus both lookahead modes and mixed-residency staging-ring schedules.
The CUDA test checks byte-for-byte CPU-reference and per-expert parity on both GPUs,
full and partial groups, nonzero `first`, forced flushes, NVFP4 tails, zero padding
and untouched sentinels, unaligned fallback, invalid bounds/strides, two-stream
ordering and eight-slot wrap/release. The existing K-quant product test covers
unchanged MMQ output against its numerical reference. These are synthetic mixed-ring
tests, not full-engine mixed-residency benchmarking.

The runner refuses to overwrite any arm. Use a new `--arm` name to reproduce:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather/run.py --arm new-off --engine build/cuda-batched/strata
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather/run.py --arm new-on --engine build/cuda-batched/strata --on
```

Default sizes are1977 tokens, warmup +three reps. `--sizes small medium` selects
the original502 /994 fixtures; `--counters --reps 1` selects separate mechanism
checks, **not speed evidence**. `analyze.py` summarizes persisted results offline.
`check-results.py` independently validates the seven recorded arms;
`verify-build.py` reproduces compiled K-quant instruction identity. All passed.
The four CPU tests and two final CUDA CTests above passed. The inherited empty
`CUDA_VISIBLE_DEVICES` initially hid both GPUs; the corrected explicit environment
command is shown above. Earlier failed setup/test attempts remain in the logs;
they were not inference failures. `sha256sum -c SHA256SUMS` checks this bundle
from its directory.
No32K/near-max request, another split, further optimization, push, merge or PR.
Fossil ticket5f8e4f2509 remains open.
