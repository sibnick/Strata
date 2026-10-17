# K37 MMQ batched gather: 32K pair and one near-limit validation

Local date2026-10-17, Linux, CMP170HX64GiB + RTX3090 24GiB.
This follows the user's approval to validate the successful opt-in at long context.
It is not another optimization, a placement search or an acceptance claim for every
model/workload. Runtime source, math, defaults and binaries are unchanged.

**Both32K sanity arms and the single batchingON261,682-token near-limit request
passed exact historical output IDs, complete fresh input and the unchanged safety
gates.** The controlled32K pair took40,792.7ms OFF versus34,812.6ms ON
(14.66% less prefill time,1.172x). Near-limitON observed295,825.7ms prefill,
884.582tokens/s and295.880s TTFT. The latter is not a controlled batching speedup.

## Measured results and limits

| Arm / request | Input tokens | Output / cap | Prefill ms | Prefill tokens/s | Observed TTFT s | Wall to DONE s | Decode ms | MTP accepted / offered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF warmup | 1977 | 22 /32 | 2655.2 | 744.577 | 2.680 | 2.912 | 251.7 | 9 /12 |
| OFF32K | 32788 | 37 /192 | 40792.7 | 803.771 | 40.815 | 41.197 | 397.3 | 18 /23 |
| ON warmup | 1977 | 22 /32 | 2320.8 | 851.861 | 2.346 | 2.570 | 245.5 | 9 /12 |
| ON32K | 32788 | 37 /192 | 34812.6 | 941.843 | 34.835 | 35.215 | 393.7 | 18 /23 |
| ON near-limit | 261682 | 37 /192 | 295825.7 | 884.582 | 295.880 | 296.289 | 425.2 | 18 /24 |

The32K pair uses identical current binary/configuration and a fresh process per
arm, each after its own1977-token warmup. OFF then ON, **one32K observation each**:
not randomized, not a repeated median or a general significance result. Startup
135.198s OFF /135.410s ON is separate. The measured32K difference is5980.1ms
prefill (14.6597% time reduction); wall-to-DONE falls41.197s to35.215s.

Near-limit is **one successfulON validation**, not a freshOFF/ON comparison.
Historical K37MMQ took341,240.9ms /766.854tokens/s, TTFT341.288s.
CurrentON took295,825.7ms /884.582tokens/s, TTFT295.880s: a descriptive
45,415.2ms lower time (13.31%; historical/current time ratio1.154).
The historical executable/DSO hashes differ; runtime trace was enabled then
and absent now; process order, cache, host/desktop state, thermals and measured
power/clocks differ. **Do not attribute that entire historical difference to
batching or call it a controlled near-limit speedup.** No near-limitOFF was run,
and the short13% result was not used to predict or authorize a timing claim.

Near-limit decode425.2ms /87.018tokens/s versus historical421.7ms /
87.740tokens/s is not an improvement; both accept18/24 MTP drafts. The32K
decode397.3ms OFF /393.7ms ON windows are too short for a decode speedup claim.
All three long answers have the same37 historical token IDs; actual near-limit
input+output totals261,719 cells. Reused0, fresh261,682. All measured32K and
near-limit intervals report0.0MB expert reads; the first warmup in each process
includes startup-cumulative77,024.6MB expert reads. This does not imply zero
storage traffic: directPLE and other file I/O remain.

Near-limit progress advanced from28,928 tokens at952.5tokens/s to241,408 at
888.4tokens/s, ending884.582tokens/s. The raw PP/progress/PCIe logs remain durable.
This records context-dependent slowing without a profiler or causal attribution.
Only the requested five GPU requests ran; no retries or extra control requests.

### Memory and telemetry

| Phase | CMP minimum free MiB | RTX minimum free MiB | Host minimum available GiB | Max engine RSS GiB | Max sample gap s |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF32K | 922.3125 | 1917.6875 | 10.662 | 4.764 | 1.0992 |
| ON32K | 922.3125 | 1917.6875 | 10.475 | 4.760 | 1.0998 |
| ON near-limit | 922.3125 | 1913.6875 | 10.236 | 4.770 | 1.1006 |

Those near-limit values are also the overall minima/maxima across both arms:
CMP967,114,752bytes, RTX2,006,646,784bytes, host10,990,931,968bytes;
maxRSS5,121,802,240bytes. Engine swap0, existing system swap132.25MiB
throughout. Both640MiB guards and host5GiB reaction threshold remain satisfied.
No guard/OOM, unexpected compute, failed request or configuration fallback.
Polling extrema are not guaranteed instantaneous peaks or an OOM guarantee.

Actual power limits stayedCMP250W /RTX300W. Near-limit samples: CMP
81.36-243.21W, SM1455-1470MHz,50-65C; RTX158.04-196.90W,
SM1755-1950MHz,38-49C. These include request-boundary/decode samples, not
isolated kernel-power measurements. CMP active clock-event reason masks are0;
RTX masks include0 and0x4, preserved raw rather than claiming no clock limiting.
No clock/power control was changed.

Near-limitPCIe272samples/device, tool-printedMB/s: RTXmeanRX43.35/TX13.75,
max612/80; CMPmeanRX54.96/TX46.46,max204/572. Whole-device averages include
the request window, not phase-isolated traffic attribution or a saturation test.
Per-arm and per-phase extrema and full request PCIe summaries are in `summary.json`.

Both engines and both PCIe helpers exited0. `cleanup.json` verifies all four owned
PIDs absent, no GPU compute remaining and memory released. FiveCPU harness tests
passed before and after execution. The independent five-request checker, deterministic
offline analysis, unchanged runtime/build identities and preserved previous bundle
checksums passed. No new kernel/backend build was necessary or performed.

## Source, artifacts and exact configuration

Engine source commit `97b83332` is the default-off gather-batching change. Executable
SHA256 `aba34837fe52effa15450a85a82d6cde9af6a55c7866361ecd1f49099a5f6af9`,
K-quant DSO `3ba4cc2c791be319e5c010965b95c75443fb52194b52ea28c43c1fd8397335a0`.
The **same unchanged executable and DSO** run both arms. No rebuild, patched
pointer-table/header code, kernel/tile/quantization change or new default.
`build-identity.json` refers to the preserved CUDA13.4.92 Release SM80+86 build
and pristine compiled-kernel identity from the preceding short experiment.
The runner checks the binary, DSO, CMake cache, CMakeLists and prefill source hashes
against that evidence and requires an empty runtime diff since `97b83332`.
The loaded DSO path is independently checked in `/proc/PID/maps`.

Exact argv and environments are in each arm's `results.json`. They reuse the
parser-verified successful K37 short protocol, not CLI spellings inferred from
prose. `STRATA_MMQ_KQ=1`, `STRATA_PREFILL_BATCH_GATHER=0` vs `1` is the only
OFF/ON difference. Trace, phase timing and diagnostic gather counters are absent
in both arms. The ON warmup must positively log the enabled gather path before
long requests can proceed.

PhysicalGPU1 CMP maps to logicalCUDA0, layers[0,37); physicalGPU0 RTX maps to
logicalCUDA1, layers[37,48). All24,576 experts remain resident in their owning
GPU caches (18,944 /5,632 experts,59,192,115,200 /17,825,792,000 resident bytes).
Startup checks UUIDs, ownership, full residency, context262,144 INT8, chunk256,
owned prompt buffers/no borrowing, MTP/spec2 and exact configuration reports.
Both allocation reserves remain1024MiB, both NVML monitor stops640binaryMiB.
The original directPLE, mmap/native experts, profile, MTP and model artifacts are
unchanged. No downloads or model/pack/MTP/shard modifications.
Small artifact metadata is freshly hashed; large-file hashes are retained from
the original manifest with current size/mtime checks, not freshly rehashed.

## Authorized schedule and correctness gates

Exactly five requests, no extra GPU prompts:

1. OFF process: original1977-token warmup, cap32; one original32,788-token sanity,
   cap192; EOF/exit0 and no compute remaining.
2. ON process, only after OFF passes all gates: same1977-token warmup, cap32;
   same32,788-token sanity, cap192.
3. Only if both sanity requests pass exact historical output IDs, zero reuse,
   complete fresh input, expert-read and safety checks: **one** original261,682-token
   near-limit fixture, batchingON, cap192. No near-limitOFF control or repeat.

The original fixtures are read from the committed earlier evidence directories
without regeneration or copying over them. `prompt-references.json` records exact
text/ID/metadata hashes, sizes, historical results and helper references.
Near-limit input ID SHA256 is
`67a876781b4fde7323b1b340a5e55700acb3f6741dc19d953d32229c42cf55ed`.
The raw text file hash differs from metadata's body-without-newline hash; both
original identities are preserved, not silently normalized.

Input261,682 plus output cap192 reserves261,874 of262,144 cells. This is
near-full context, **not262,144 input tokens**. Needle positions59 /130,846 /261,559
check beginning/middle/end. Required long answer:

```text
ORIGIN: IRIS-4Q7-LANTERN
CENTER: CEDAR-8M2-QUARRY
FINAL: ORBIT-6V9-KETTLE
```

Each long answer must finish below cap192, match all37 historical output IDs
includingEOS and independently decode to those exact expected lines. Warmups
must match the historical22 output IDs and714 /928 /365 recall. DONE must report
zero reused and the full input freshly read. The batched prompt checkpoint has
one fewer token because the final input seed takes the token path. Measured
post-warmup intervals must report0.0MB expert file reads; warmup's counter includes
startup loading. Exact-retrieval validation is not a broad numerical quality screen
or full internal-state hash.

## Safety, monitoring and cleanup

Before each engine startup, no compute process may be present, hostMemAvailable
must be at least6GiB, both GPU UUIDs must match and all binary/fixture checks pass.
During startup and every request, approximately1-second NVML samples enforce
both640MiB GPU stops and host5GiB reaction threshold (4GiB floor +1GiB margin).
Raw records include GPU free bytes, power/limits, SM/graphics clocks, temperature,
clock-event reason masks, hostMemAvailable/RSS/swap/systemswap, phase and sample gap.
Any unrelated compute triggers stop of our engine only. Polling does not guarantee
that every transient pressure/OOM is preventable.

PCIe telemetry uses an owned `nvidia-smi dmon` helper, timestamped in `pcie.log`;
the helper must be alive and produce parseable samples before prompts. These are
whole-device RX/TX rates in the tool's printedMB/s, not isolated model transfers,
copy-engine activity or saturation. No profiler is enabled.

Request submission is threaded so the monitor continues while a large token-ID
line crosses the pipe. Progress checkpoints are reported about30s apart without
a progress/disk watchdog. Timeout is the original measured-rate rule,
`max(1800s,4*input_tokens/preceding_tokens_per_second+300s)`, with the prior
near-limit766.854tokens/s as initial reference. The generous1800s minimum applies
to near-limit, rather than treating active prefill as a stall.

EOF shuts down successful arms; failure stops only the owned engine. The PCIe
helper is stopped and its thread joined after every arm. Exit codes, remaining
compute users and released memory are recorded. No resets/reboots, cache drops,
clock/power changes, killing unrelated users, relaxed reserves or guard bypasses.
Failures, if any, remain durable; no automatic retries.

## Reproduction and evidence

From repository root, with the unchanged validated binary and original artifacts:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long -p test_long.py -v
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/run.py --arm off
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/run.py --arm on
```

The runner refuses to overwrite an existing arm. The ON command requires the
passed OFF evidence, so executing it alone cannot bypass sanity admission.
The five CPU harness tests cover immutable original counts/hashes/headroom,
exact historical IDs/decoded recall/freshness, unchanged candidate identity,
both640 guards/host reaction/timeout and failed OFF admission.

`analyze.py` regenerates `summary.json` offline. `check-results.py` independently
checks the exact five-request schedule, outputs, safety, identical runtime
identities/configuration, disabled instrumentation, actual ON diagnostic and
cleanup. `SHA256SUMS` covers scripts, references, raw engine/PCIe/monitor/progress,
results, build identity, checks and README; run `sha256sum -c SHA256SUMS` here.
Preserved earlier evidence is not modified.

Only the one32K OFF/ON pair can serve as a controlled same-binary batching
comparison, and it is sequential with one sample per arm, not randomized repeated
evidence. The near-limit result may be compared **descriptively only** to historical
K37MMQ341,240.9ms prefill /341.288s observedTTFT; build, order, host/cache/thermal
and power state differ. There is no freshOFF near-limit denominator and no
extrapolation from the earlier short-prompt13% gain. HIP/SYCL, MTP-off,
other placements/models and repeated near-limit acceptance remain untested.
No further optimization, push, merge or PR; Fossil5f8e4f2509 staysOPEN.
