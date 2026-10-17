# K36 GPU utilization investigation

## Result

The bounded K36 timing capture found a large first-chunk host-side PLE gather
wait before the GPU stages can proceed; it did not establish a kernel or PCIe
bandwidth limit. “100% on both GPUs” is not a reliable target for a single
dependent request: the RTX suffix can only process activations after the CMP
prefix produces them, and sampled utilization is not SM occupancy or useful
throughput. The useful objective is to reduce end-to-end latency or raise
sustained throughput while keeping both stages productively occupied.

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

Minimum sampled free VRAM was 277 MiB on the RTX 3090 and 1,598 MiB on the
CMP 170HX; minimum host `MemAvailable` was 15.18 GiB. Admission passed before
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

## Decision and ranked follow-up

1. Split the first-chunk PLE wait into row-key generation, reader queue/collect,
   row decode, and H2D enqueue/sync. If direct table I/O dominates, measure
   whether additional read-ahead or a bounded host row cache hides that wait;
   both trade more host memory / I/O pressure for less GPU starvation. Do not
   change the 170HX/3090 split before this is separated.
2. Compare one natural-language 1,977-token prompt with the same timing hooks
   under an approved exclusive pause, then a second identical request only if
   memory and prompt-freshness policy allow. This distinguishes cold random
   PLE behavior from ordinary API text. The current request was synthetic, so
   this control is still missing.
3. Only after PLE is understood, measure stage service with a longer bounded
   prompt and fresh admission. A different layer split could raise RTX work
   but spends its very small post-load VRAM margin; multi-request concurrency
   could improve aggregate throughput but costs additional KV/scratch memory
   and is not a path to 100% utilization for one request.

No kernel, scheduler, layer split, or default setting was changed. The
instrumented capture harness and sanitized monitor/phase evidence are retained
beside this report: `gpu-utilization-capture/capture.json` has the phase
summary, `monitor.jsonl` has per-second host/GPU/thread samples,
`dmon-samples.txt` has timestamped utilization/power/clocks/PCIe data, and
`SHA256SUMS` covers those three artifacts. Reproduce only during an approved
exclusive pause, after
the API is stopped and fresh admission passes:

```sh
python3 -B bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/capture_gpu_timing.py
sha256sum -c bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/gpu-utilization-capture/SHA256SUMS
```

Existing K36/K37 evidence and checksums remain unchanged. This does not
validate near-limit attention or establish full-context safety.
