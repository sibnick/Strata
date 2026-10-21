# K35 MMQ with resident-RAM adaptive promotion

2026-10-16, CMP 170HX 64 GiB plus RTX 3090 24 GiB, CUDA only. This evidence bundle compares two guarded K35 short-input arms: a CPU-computed resident-RAM miss control and synchronous adaptive promotion of those misses into the GPU cache. MMQ was enabled in both. Both arms and the single approved 32,788-token sanity request passed.

The short measurements show no meaningful wall-time difference between these two arms: the adaptive/control median wall ratios were 0.998, 0.995 and 1.003 for 502, 994 and 1,977 input tokens. This is descriptive evidence for these prompts, not a general performance, quality or best-split claim.

## Placement and admission

`--gpu 1,0 --layer-split 35` put layers `[0,35)` on logical CUDA0 / physical GPU 1 (CMP) and `[35,48)` on logical CUDA1 / physical GPU 0 (RTX). The engine used automatic profile-ranked cache sizing, not a fixed `--expert-cache` count and not `--gpu-owned-experts`.

Both arms used the same tested CUDA executable, MMQ library, prompt IDs, cache reserve and resident complement. Both used `STRATA_MMQ_KQ=1`, fixed prefill chunk 256, INT8 KV with allocated context 262,144, MTP/spec2, `--no-prefill-borrow`, mmap experts, profile ranking, 4 GiB resident budget and pinned CPU experts. The only scheduler difference was control `--adapt-every 0 --adapt-swaps 0 --adapt-async 0` versus adaptive `--adapt-every 4 --adapt-swaps 96 --adapt-async 0`.

The RTX later-stage allocation reserve was 2,049 MiB; the CMP allocation reserve was 1,024 MiB. External stop thresholds were 256 MiB free on the RTX and 640 MiB on the CMP. Host MemAvailable had to start at 6 GiB and stop below 5 GiB, retaining the 4 GiB floor plus 1 GiB reaction margin. Pinning remained enabled with 4 GiB headroom and pin reserve.

At startup, both arms reported 17,920 primary-cache slots and an RTX AUTO cache of 5,659 slots / 16.59 GiB, holding 5,659 of the 6,656 profiled stage-1 pairs. The 997 remaining stage-1 pairs were assigned to resident RAM; the runtime reported a 2.94 GiB mapped, page-locked complement in both arms. Exact logical expert payload is **3,157,504,000 bytes** (2.94065 GiB), calculated from the profile-ranked cold pairs and the per-layer payload sizes in `../2026-10-16-cmp170hx-rtx3090-k37/ledger-chunk256.csv`. `analysis.json` records the pair counts and bytes by layer, profile SHA-256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`, and ledger SHA-256 `142e1fe37cd5ffde74e05ed9f4c79dfada9974510036aef7548978aa7786d01e`. The runtime prints resident and pinned sizes rounded to 0.01 GiB; the byte-exact payload is derived from the profile and ledger.

The short prompts were all fully fresh, with zero prefix reuse. Each arm ran one 1,977-token warmup and three measured repetitions of the identical 502-, 994- and 1,977-token fixtures. Each short answer matched the independently expected three-code retrieval answer and corresponding generated token IDs matched between arms. All measured short requests reported 0.0 MB of expert reads from files. OS storage reads were nonzero and are retained in the raw per-request counters.

## Short-request measurements

Medians exclude the warmup. `adaptive/control` is the ratio of median wall time to `DONE`; a value near 1 means the measured times were similar.

| Input tokens | Control prefill ms | Adaptive prefill ms | Control wall s | Adaptive wall s | Adaptive/control |
|---:|---:|---:|---:|---:|---:|
| 502 | 676.3 | 676.8 | 0.9240 | 0.9219 | 0.998 |
| 994 | 1,227.7 | 1,233.9 | 1.4634 | 1.4564 | 0.995 |
| 1,977 | 2,357.8 | 2,362.0 | 2.5897 | 2.5977 | 1.003 |

All short outputs were 22 tokens including EOS, below the 32-token cap. MTP remained enabled; the exact accepted/offered counts are in both `results.json` files.

For context only, previous K37 MMQ and K36 MMQ short wall medians were 0.9054 / 1.4978 / 2.6348 s and 0.8776 / 1.4481 / 2.5728 s, respectively, at the same three input lengths. Those are separate runs with different stage splits, cache ownership and reserves; they are not controlled comparisons with K35. No K36 profile was run for this experiment.

## Cache misses and adaptive exchanges

The median per-request decode-cache statistics over the three measured repetitions were:

| Input tokens | Control GPU hit rate | Adaptive GPU hit rate | Control GPU hits | Adaptive GPU hits | Control resident-RAM blob reads | Adaptive resident-RAM blob reads |
|---:|---:|---:|---:|---:|---:|---:|
| 502 | 94.9% | 98.8% | 3,009 | 3,208 | 837 | 662 |
| 994 | 94.7% | 98.8% | 2,758 | 2,950 | 1,472 | 1,260 |
| 1,977 | 95.1% | 99.0% | 2,785 | 3,216 | 2,924 | 2,660 |

The exact hits/lookups, other-GPU reads and counters for each individual request are in `analysis.json`. MTP acceptance differed slightly between some repeated requests, so these counts are descriptive per-request totals, not a route-normalized estimate of algorithmic savings.

Per-request resident-RAM blob reads and exchange counts are preserved in `analysis.json`. The runtime's `resident RAM` counter is cumulative, so the analysis reports deltas from adjacent request-end counters. In the order warmup, three small/medium/large repetitions, then the 32K sanity request:

| Arm | Resident-RAM blobs per request | RAM-to-GPU exchange events per request |
|---|---|---|
| Control | 2,924; 837; 1,472; 2,924; 837; 1,472; 2,924; 837; 1,472; 2,924 | 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 |
| Adaptive | 2,914; 705; 1,282; 2,680; 662; 1,260; 2,660; 655; 1,258; 2,656; 52,996 | 28; 20; 5; 2; 1; 1; 0; 2; 0; 0; 15 |

The adaptive arm recorded 59 exchange events through its warmup and nine measured short requests (31 during the measured repetitions), then 15 during the one 32K sanity request. Control recorded none. These are exchange-counter deltas, not adaptation-round counts. The unchanged runtime did not emit per-request round counts, accepted-round histograms, layer IDs or per-stage swap-byte totals. No tracing, profiling or runtime changes were added, so round counts and per-stage swap bytes are not inferred.

The first warmup's file-tier line includes 73,867.1 MB of startup-cumulative expert reads; it is not a warmup-only delta. After warmup, every short request and the 32K request reported 0.0 MB of expert reads from files. Other OS reads remain visible in `file tier I/O this request`.

## The 32K sanity request

After the control passed and all adaptive short requests passed, the adaptive arm ran the single approved existing 32,788-token fixture with a 192-token output cap. It returned the exact expected answer in 37 tokens, with zero prefix reuse and 39,150.0 ms prefill / 39.570 s to `DONE`. MTP accepted 18 of 22 offered drafts. This is one working-behavior sanity request, not a long-context performance claim.

## Memory and cleanup

| Arm | Minimum RTX free MiB | Minimum CMP free MiB | Minimum host available GiB | Maximum RSS GiB | Maximum process swap |
|---|---:|---:|---:|---:|---:|
| Control | 1,457.6875 | 4,422.3125 | 7.637 | 8.703 | 0 |
| Adaptive | 1,431.6875 | 4,400.3125 | 7.022 | 9.085 | 0 |

Maximum sampled system swap use was 60 MiB for control and 132.25 MiB for adaptive; process swap remained zero. Maximum sample gaps were 1.100 s and 1.101 s. These are sampled extrema, not a hard OOM guarantee. Neither arm crossed a guard or reported an OOM.

Both engines exited zero. `cleanup.json` records no remaining NVML compute process, absent engine PIDs, no matching runner/helper process, returned GPU allocations, driver/device identity and host memory after cleanup. At the final check the RTX used 28 MiB and CMP used 16 MiB, both at 0% utilization; host MemAvailable was 17,812,131,840 bytes.

The first, no-inference startup attempt is retained in `startup-gate-abort-1/`. It was stopped because an earlier checker expected a literal reserve string the engine did not print. No prompt was sent. The corrected startup gate validates the argv reserve, actual AUTO capacity and ownership, and pinned complement; it also passes against the captured rejected-startup log. The attempt is not an experiment result.

## Identity and reproduction

`control/results.json` and `adaptive/results.json` preserve the exact argv and environment overrides, runtime/source commit, executable, library and build-cache hashes, artifact manifest identities, prompt hashes, request outputs, adaptive diagnostics, memory samples and timestamps. `build-identity.json` records the unchanged CUDA Release sm80+sm86 build; HIP and SYCL were not built or tested. Large model files were not freshly rehashed; recorded hashes and file size/mtime checks were reused. No model, MTP, pack or runtime source was changed.

From the repository root, offline checks are:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive -p 'test_k35.py' -v
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive/analyze.py > bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive/analysis.json
sha256sum -c bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive/SHA256SUMS
```

The eight CPU safety tests and offline analysis passed. The guarded commands used for the two arms are recorded in their JSON:

```sh
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive/run-k35.py --mode control
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k35-mmq-adaptive/run-k35.py --mode adaptive
```

The runner refuses to overwrite an existing arm directory. No additional split, K36 profiling, alternate prompt, runtime algorithm, push or merge was part of this K35 experiment.
