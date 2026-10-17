# K36 GPU utilization investigation

## Result

Two bounded K36 captures show that PLE cost depends strongly on the input and
cache state: the synthetic request exposed a 2,737 ms PLE wait, while the
historical fixture exposed 146 ms of wait despite 346 ms total direct-reader
collect time. Neither established a kernel or PCIe bandwidth limit. “100% on
both GPUs” is not a reliable target for one dependent request: the RTX suffix
can only process activations after the CMP prefix produces them, and sampled
utilization is not SM occupancy or useful throughput. The objective is lower
end-to-end latency or higher sustained throughput with both stages
productively occupied.

## Evidence

Before the later authorized diagnostic pause, a read-only snapshot at
`2026-10-17T19:24:41Z` showed `/health` as `status=ok`, `loaded=true`, and
`/v1/status` reported 12 requests and `in_flight=0`. No request was sent and
no prompt or answer content was read during that snapshot. The RTX 3090 had
249 MiB free and the CMP 170HX had 1,611 MiB free; host `MemAvailable` was
15.82 GB (14.74 GiB).

A read-only 20-second `nvidia-smi dmon -i 0,1 -d 1 -c 20 -s put` sample while
idle reported 0% SM and memory utilization on both GPUs in every sample.
RTX power was 23–29 W and CMP power 37–38 W. PCIe counters were near zero
(maximum observed 1 MB/s RX and 4 MB/s TX). These are idle observations, not
measurements of a request. GPU 0 is the RTX 3090 and GPU 1 the CMP 170HX.

The prior K37 long-context analysis at
`../2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context/offline-analysis/README.md`
contains the only useful saved busy-period utilization evidence: during the
261,682-token prefill, the CMP averaged 97.75% (median 99%) across 752
approximately 1.08-second samples; the RTX averaged 34.64% (median 16%) and
was at or below 20% in 62.2% of samples. The measurement window was an
approximate prefill interior, not a CUDA-event trace. Those whole-device
samples establish an observed K37 activity imbalance, but do not measure
occupancy, per-stage service time, or causality. The associated PCIe counters
are whole-device counters and cannot identify model handoff traffic.

The K36 2048/512 bounded-run record in `README.md` reports 15,529.8 ms for the
32,788-token prefill and 1,409.0 ms median for the three short prefills. Its
once-per-second monitor is not aligned to those phases, so it cannot establish
K36 busy-period utilization or distinguish GPU work from host launch, gather,
transfer, or stage-wait time. The source-backed pipeline stages activations
through pinned host buffers; stage dependency and unequal stage service time
are plausible explanations, not yet isolated causes.

### Bounded K36 timing capture

After explicit approval, the API was stopped cleanly, a single isolated
diagnostic was run, and the ordinary API launcher was started again with the
same config. The diagnostic used the exact K36 2048/512 config, 0 MiB engine
reserves, no external GPU-free abort, a fresh admission calculation including
the mandatory 512 MiB prompt-init headroom, host start/stop guards of 6/5 GiB,
and the engine's normal CUDA admission. It sent one synthetic 1,977-token
sequence of ordinary tokenizer IDs, capped output at 32, and did not store the
sequence or response text. The request stopped after two generated tokens.
No long, near-limit, 128K, or 262K request was run.

The request window was 19:39:25.520–19:39:29.744 UTC. The CMP primary stage
(CUDA device 0, physical GPU 1) reported 3,676 ms on its event timeline and
3,771 ms wall time; the RTX peer stage (CUDA device 1, physical GPU 0)
reported 362 ms and 377 ms, respectively. The primary stage's separate host
timer reported 2,737 ms in PLE. This interval includes waiting on the async
PLE future before layer 1; source shows that future builds n-gram row keys and
calls `PleTable::gather_batch`. In direct mode, that batches a reader issue /
collect and row decode. The CUDA event timer assigns inter-mark gaps to the
preceding phase, so its 2,756 ms “combine” interval is **not** evidence that
the combine kernel itself ran for that long: it nearly matches the 2,737 ms
host PLE wait. The reader wait, CPU key generation, and device-upload enqueue
were not timed separately.

The peer host timers reported 13 ms waiting for a chunk and 68 ms after each
chunk for draft-layer/progress work. Primary host chunk setup, per-chunk
waiting, and PLE-kernel time were reported as 0, 0, and 3 ms; its host PLE
wait was the outlier. The peer's 362 ms event timeline and the primary's roughly 939 ms residual
after subtracting the measured PLE wait suggest about a 2.6:1 prefix/suffix
service imbalance. That residual is not a pure SM-busy measure: CUDA event
intervals can still include other stream gaps. This one synthetic cold
sequence cannot establish steady-state service rates or motivate a layer-split
change.

The one-second monitor samples did not give a consistent utilization picture
within the roughly four-second request: `dmon` and the separate per-second
query disagreed on which GPU was at its utilization peak. PCIe was low in the sampled request window: at most 1 MB/s RX and 24 MB/s TX
on the RTX, and 34/9 MB/s on the CMP. This is not evidence of link saturation.
The engine process had 91–92 threads and 0–3 runnable at sampled instants;
from request+1.0 to +4.15 seconds its CPU tick count rose by 136 ticks. This
does not show whole-host CPU saturation. Decode was only two tokens, too short
for a decode-utilization conclusion.

Minimum sampled free VRAM was 277 MiB on the RTX 3090 and 1,637 MiB on the
CMP 170HX (1,716,518,912 bytes); minimum host `MemAvailable` was 15.18 GiB. Admission passed before
startup with margins of 220,824,340 B on the RTX and 1,733,296,128 B on the
CMP after the required headroom. No compute process remained after the
diagnostic. The API restart was then verified healthy at
`127.0.0.1:8080`; it remains loopback-only with the tested K36 settings.

This was a deliberately synthetic token sequence, not natural user text; its
random n-grams may cause unusually cold PLE-table reads. Its 4,152 ms prompt
time is therefore not comparable to the earlier 1,409 ms natural-fixture
median, and the event timers add instrumentation overhead. Treat the 2,737 ms
PLE wait as a strong follow-up lead for this input, not a general request
latency claim.

### Second bounded capture: historical short fixture

After a second explicit approval, the API was stopped and the instrumented
standalone engine ran one fresh request from the exact historical `large`
1,977-token fixture. Its prompt-ID hash was
`953be64023898d51c4e513b40fefb66721ed2c643a7c18713c15edc81d4c014f`; all
1,977 input tokens were fresh, 22 generated IDs matched the historical
reference exactly, and no prompt or answer text was saved. The request took
1,874.1 ms for prefill and 246.0 ms for decode. No long-context workload ran.
The capture used zero engine reserves, no external GPU-free abort, and kept
the 6/5 GiB host guards and mandatory CUDA/admission checks.

The CMP primary prefill event timeline was 1,241 ms (1,337 ms wall), and the
RTX peer was 518 ms (533 ms wall). In the 1,976-token PLE chunk, row-key
generation took 0.063 ms, reader issue 6.861 ms, reader collect 345.870 ms,
and decode 2.028 ms; the whole gather took 354.775 ms. The asynchronous gather
future blocked the primary path for 146.456 ms, while H2D enqueue took
0.036 ms and the previous-upload completion wait was 0 ms. Thus most measured
reader time overlapped other work; the request's exposed PLE wait was 147 ms,
not the whole 355 ms gather. The comparison with the earlier synthetic
sequence (2,737 ms host PLE wait) confirms that PLE cost is strongly
input/cache-state-sensitive, but does not isolate caching as the cause.

N-gram read latency can be high for initial reads and lower after caching.
That is an important qualification: on Linux this
direct PLE reader opens the table with `O_DIRECT`, bypassing the OS page
cache. Its bounded in-process row cache can still hit repeated rows during a
request, and SSD/controller caching may persist across engine restarts; no
caches were flushed. The engine was restarted, so this run began with a fresh
in-process row cache, but this capture did not enable `--stats` and therefore
did not record row-cache hit percentage or per-read p50/p99. The harness now
enables the existing `--stats` output for future approved captures. Do not
interpret this run as a cold-disk result or claim a measured warm-cache speedup.
The engine and monitor both exited cleanly and the exact-ID check passed; the
harness's final check initially returned nonzero only because it required the
omitted `ple io` summary. The harness was corrected to launch with `--stats`.

The one-second samples overlapping the request saw GPU 0 (RTX) at 2% then 0%
and GPU 1 (CMP) at 30% then 65%; each sample found one runnable engine thread.
They are too sparse to describe whole-request utilization or SM occupancy.
Minimum sampled free VRAM was 296,747,008 bytes (283 MiB) on the RTX and
1,714,421,760 bytes (1,635 MiB) on the CMP; minimum host `MemAvailable` was
15.0 GiB. Startup admission passed, including the 512 MiB prompt-init
headroom. The engine and monitor exited with no remaining compute process.

### Bounded CUDA timeline attempt

After the separately approved exclusive pause, Nsight Systems 2026.3.2 traced
one fresh request using the exact historical 1,977-token fixture, cap 32, and
unchanged K36 2048/512 runtime. The request completed in 1,673.1 ms prefill
and 250.2 ms decode, with MTP accepting 9 of 12 proposals. All 22 generated
token IDs were compared offline with the historical fixture and matched
exactly. Only the prompt-ID hash was retained; the temporary output-ID file
was removed after comparison.

Nsight stopped collection but remained in report generation beyond the
120-second wait. No `.nsys-rep` or SQLite timeline was produced, so this
attempt does not separate CUDA kernels, CPU launch gaps, copies, or stage
waits, and it supports no optimization choice or speedup claim. It was not
rerun. The sampled minimum free memory was 277,872,640 bytes (265 MiB) on
physical GPU 0 (RTX 3090), 1,685,061,632 bytes (1,607 MiB) on physical GPU 1
(CMP 170HX), and 13,432,557,568 bytes (12.51 GiB) host `MemAvailable`. The
capture session and its orphaned Nsight agent were cleared. The ordinary API
was restored using `api/start-api.sh` and verified loaded/healthy at
`127.0.0.1:8080`; it remains on the tested K36 configuration. The timeline
harness restores the API in its cleanup path and records cleanup errors. A
later bounded capture resolved the report-finalization problem; see below.

## Decision and ranked follow-up

1. If further work is approved, enable `--stats` to capture the reader's
   row-cache hit rate and direct-read latency distribution, then compare a
   repeat of this exact fixture only after a separate pause approval. Do not
   flush storage caches; cold-disk behavior needs a separately designed,
   safe test.
2. Direct gather reads took 346 ms but the exposed primary wait was 146 ms,
   which bounds the possible prefill improvement from completely hiding this
   gather on this request. Consider read-ahead or a larger bounded row cache
   only if reader stats show repeatable missed-read stalls; both trade host
   memory or I/O pressure for less GPU starvation. Do not change the layer
   split before that evidence.
3. Only after PLE is understood, measure stage service with a longer bounded
   prompt and fresh admission. A different layer split could raise RTX work
   but spends its very small post-load VRAM margin; multi-request concurrency
   could improve aggregate throughput but costs additional KV/scratch memory
   and is not a path to 100% utilization for one request.

No kernel, scheduler, layer split, or default setting was changed. The
original synthetic capture and the historical-fixture capture each have
their own directory: `gpu-utilization-capture/` and
`gpu-utilization-capture-ple-phases/`. Each `capture.json` has the sanitized
phase summary, `monitor.jsonl` has per-second host/GPU/thread samples, and
`dmon-samples.txt` has timestamped utilization/power/clocks/PCIe data.
`SHA256SUMS` covers each directory's three artifacts. Reproduce only during
an approved exclusive pause, after the API is stopped and fresh admission
passes:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/capture_gpu_timing.py --prompt-fixture historical-large --capture-dir gpu-utilization-capture-ple-phases-repeat
sha256sum -c bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/gpu-utilization-capture-ple-phases-repeat/SHA256SUMS
```

Existing K36/K37 evidence and checksums remain unchanged. This does not
validate near-limit attention or establish full-context safety.

### Third bounded capture: CMP kernel attribution

The previous Nsight attempt had produced no report because finalization did not
finish. A tiny CUDA probe established a working collection/export path when
compiled explicitly for sm80 and run with CPU instruction sampling, symbol
resolution, and automatic summary-stat generation disabled. The resulting
SQLite export contained a real kernel record. The K36 runner now uses those
bounded settings while retaining CUDA, cuBLAS, OS runtime, and process-tree
thread-scheduling events.

With the loopback API idle (`requests=0`, `in_flight=0`), one standalone engine
was run under Nsight with the unchanged K36 2048/512 settings and the historical
1,977-token fixture. The input was fresh; all 22 output IDs matched the
historical reference, and MTP accepted 9 of 12 proposals. The profiled request
reported 1,643.6 ms prefill and 249.3 ms decode. No warmup, extra variant,
long-context, or near-limit workload ran. These are instrumented timings, not
an unprofiled speed comparison.

The capture directory is
`gpu-utilization-capture-cuda-timeline-cmp-sm80/`. It contains the native
Nsight report, SQLite event export, sanitized capture summary, monitoring
samples, and `SHA256SUMS`. The profiler used
`--trace=cuda,cublas,osrt --sample=none --cpuctxsw=process-tree
--resolve-symbols=false --stats=false`; it did not collect CPU instruction
samples. No prompt or response text was retained.

The report maps CUDA device 0 to the CMP 170HX (physical GPU 1) and CUDA device
1 to the RTX 3090 (physical GPU 0). During the 1,643.6 ms prefill window,
Nsight recorded 7,544 CMP kernels with 748.4 ms of non-overlapping kernel
activity (45.5% of the prompt window), from about 12 ms to 1,050 ms after the
request began. The RTX recorded 7,598 kernels and 426.3 ms of kernel activity
(25.9%), from about 1,145 ms to 1,636 ms. The CMP prefix therefore has about
2.1 times the measured kernel-activity span of the RTX suffix. The timeline
also shows a 77.2 MiB CMP device-to-host copy lasting 95.0 ms between the last
recorded CMP prefill kernel and the first RTX prefill kernel. This is
consistent with a stage handoff on the host-staged path, but the separate
Nsight peer-copy records do not agree with the machine's known no-P2P topology;
their labels and durations are not a trustworthy PCIe bandwidth measurement.

On the CMP, `mul_mat_q` kernels consumed 442.8 ms across 2,775 launches, about
59% of recorded kernel time. The MMQ `copy16_group_kernel` consumed 56.3 ms
across 945 launches; quantization and prompt-attention kernels consumed 10.9
and 24.5 ms. This makes CMP MMQ compute the leading kernel family to examine,
not the RTX. It does not establish SM occupancy, memory-bandwidth saturation,
or which MMQ tile is inefficient.

The decode interval contains 24 CMP and 41 RTX CUDA-graph activity records,
with 160.5 ms and 59.0 ms of unioned graph intervals, respectively. The export
has no graph-node kernel records, so decode is visible only at graph granularity;
it cannot identify individual decode kernels. During prefill the trace also
records 97 `cudaStreamSynchronize` calls totaling 1,314 ms, while 14,624
`cudaLaunchKernel` calls total 29.9 ms. Across the request, the corresponding
totals are 1,536 ms for 167 stream synchronizations and 33.1 ms for 17,874
kernel-launch calls. Synchronization durations overlap device work and other
threads, so they are not additive wall time; they indicate substantial waiting,
not a measured GPU-idle duration. Launch API time is much smaller and is not
the first target.

The trace includes 12 `cuLibraryLoadData` calls totaling 219.9 ms in the first
263 ms, likely first-use library/module loading rather than steady-state kernel
cost. It also records 10,665 `pread` calls totaling 3,275.6 ms across threads.
Those calls are consistent with direct PLE reader activity, but their summed
time overlaps across threads and the run did not flush or measure storage
caches. The user-noted faster later reads remain a cache-state caveat. PLE
settings and cache behavior were left unchanged; no PLE optimization was
attempted.

The approved memory policy was retained: 0 MiB engine reserves, no external
GPU-free abort, host start/stop thresholds of 6/5 GiB, mandatory CUDA
admission, and the 512 MiB prompt-init headroom. Startup margins after that
headroom were 1,733,296,128 bytes on the CMP and 220,824,340 bytes on the RTX.
Minimum sampled free VRAM was 1,685,061,632 bytes (1,607 MiB) on the CMP and
277,872,640 bytes (265 MiB) on the RTX; minimum host `MemAvailable` was
12,910,485,504 bytes (12.02 GiB). No CUDA engine or Nsight process remained
after the capture. The API was restored and verified healthy by the capture
cleanup, then stopped again when the user asked; it is now intentionally off
and port 8080 is unbound.

The captured runtime used the pre-merge executable SHA-256
`6b4e64e6fee8d6aff5a208d37579f507dd050c81d5d7030b2f0e05a515f2e643` and
K-quant library SHA-256
`0a93ccd8f52a329da586a36e76cf5865a4c3ce0256b80e3da4b28ce937bdf27d`, with
pre-merge source revision `1319480d1683d6f83de78413222a45e1d2058dd3`. The
older `ef991b2f` run provenance records a different executable hash
(`c13681c5...`) and earlier source revision (`448bce27...`). The profiled
executable is the exact binary that had been running in the tested API, but
this trace is attribution only, not a byte-identical comparison against that
older provenance artifact. The merged source was not built or deployed for
this capture.

**Decision:** no optimization was implemented. The trace prioritizes CMP MMQ
over launch-overhead reduction, but does not select a specific
`mul_mat_q` tile; several tile variants contribute, and the graph decode path
has no kernel-node detail. A bounded Nsight Compute follow-up was attempted;
the host stop guard interrupted it during the short request before any kernel
metrics or exact-output check were available. No optimization is supported by
that incomplete attempt. Do not tune the RTX, change PLE, or claim 100%
simultaneous utilization from these captures.

### Bounded Nsight Compute attempt

After the Intel-only measurement reservation was released, a standalone
Nsight Compute attempt targeted one CMP CUDA-device-0 launch of the prominent
Q4_K `mul_mat_q<(ggml_type)12, (int)128, ...>` variant from the exact tested
K36 2048/512 binary. It used the historical 1,977-token fixture, cap 32,
unchanged PLE/cache settings, 0 MiB engine reserves, no external GPU-free
abort, mandatory CUDA admission and 512 MiB prompt-init headroom, and host
start/stop thresholds of 6/5 GiB. The API remained stopped throughout.

The engine loaded and reported `READY`; the fixture request was then sent.
`MemAvailable` fell from 12.29 GiB at request start to 7.16 GiB and then
3.72 GiB over the next three one-second samples. The required 5 GiB stop
guard terminated the profiled process. The request did not report completion,
generated IDs were unavailable, and no Nsight Compute report was finalized;
there are no usable kernel metrics or parity result from this attempt.
Minimum sampled free VRAM was 257 MiB on the RTX 3090 and 255 MiB on the CMP.
The saved samples do not attribute the host-memory decline to a process:
the harness recorded only aggregate `MemAvailable`, not per-process RSS/PSS,
and no disk cache was flushed. The installed Nsight Compute 2026.3.1
Profiling Guide says kernel replay saves all GPU memory accessible to the
selected kernel and may spill that snapshot to CPU host memory when device
memory is insufficient. The sharp host decline coincided with the profiled
request, when free device memory was already close to the admission floor.
That makes profiler replay spill a plausible explanation, not a proven cause.
Nsight Compute also warned that GPU caches were uncontrolled because cache
flushing was disabled. Do not repeat this kernel-replay capture unchanged.

Application replay avoids kernel memory save/restore, but reruns the full
application for every profiling pass. In this setup that entails reloading
the model and reissuing the request, so it is not a safe substitute without
a separately bounded admission plan and host-process telemetry.

Cleanup was verified: no inference, Nsight, or API process remained, and
`127.0.0.1:8080` stayed unbound. The failed capture metadata, monitor samples,
and sanitized profiler log are retained in
`gpu-utilization-capture-ncu-cmp-sm80/`; no prompt or response text was saved.
The next step, if profiling continues, is a telemetry-only harness revision
that records per-process RSS/PSS and relevant host memory counters before any
new model load. Keep the 5 GiB stop threshold; do not retry kernel replay
unchanged or switch to full-application replay without a separate memory plan.
