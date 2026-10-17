# K36 GPU utilization investigation

## Result

The available evidence shows an imbalanced two-stage prefill, not a demonstrated
kernel or link bottleneck. “100% on both GPUs” is not a reliable target for a
single dependent request: the RTX suffix can only process activations after the
CMP prefix produces them, and the measured utilization is coarse whole-device
activity rather than SM occupancy or useful throughput. The useful objective
is to reduce end-to-end latency or raise sustained throughput while keeping
both stages productively occupied.

## Evidence

The local API was left running and untouched. At `2026-10-17T19:24:41Z`,
`/health` returned `status=ok`, `loaded=true`; `/v1/status` reported 12
requests and `in_flight=0`. No request was sent and no prompt or answer content
was read. The RTX 3090 had 249 MiB free and the CMP 170HX had 1,611 MiB free;
host `MemAvailable` was 15.82 GB (14.74 GiB).

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

## Decision and next experiment

No controlled API request, engine restart, profiler attachment, or competing
GPU workload was started. The live RTX free memory (249 MiB) is below the
previous 512 MiB prompt-init headroom, and the user API must not be disrupted.
The existing memory policy remains: zero discretionary engine reserves and no
external GPU-free abort, with the 6 GiB host-start / 5 GiB host-stop guard and
mandatory CUDA/admission checks preserved.

The smallest useful causal test is an **exclusive, user-approved maintenance
pause**: after the API is stopped cleanly and normal admission headroom is
available, relaunch the exact K36 2048/512 configuration with
`STRATA_PREFILL_TIMING=1`; run one fresh 1,977-token prompt capped at 32 output
tokens; align the engine's phase events with per-GPU `nvidia-smi dmon` SM,
memory, power, clocks and PCIe samples, plus CPU/thread scheduling. Then stop
the diagnostic engine and restore the ordinary API. This short test can
separate phase timing and likely host/device waits, but cannot validate
near-limit attention or establish full-context safety. Do not move layers to
the RTX or add concurrency until this trace and a fresh admission check show
that the configuration fits.

No optimization or runtime setting was changed. This investigation adds no
new inference evidence, and the previously committed K36/K37 evidence and
checksums remain unchanged. The API remains bound to loopback at
`127.0.0.1:8080`.
