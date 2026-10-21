# K36 MMQ near-limit context: one successful request

2026-10-16, CMP 170HX 64 GiB + RTX 3090 24 GiB, CUDA. The user requested one repeat of the successful K37 MMQ test with K36. **The original 261,682-token prompt passed exact beginning/middle/end retrieval, with zero prefix reuse and 37 completed output tokens.**

Near-limit prefill was **336,360.8 ms / 777.980 tokens/s**, versus preceding K37 MMQ **341,240.9 ms / 766.854 tokens/s**: 1.0145x throughput, or 1.43% less prefill time. Request wall time fell from 341.763s to 336.818s (4.945s). This small sequential-run difference is not a controlled best-placement result. RTX free memory reached **315.6875 MiB**, only **59.6875 MiB above its 256 MiB guard**.

## Configuration and identity

K37 evidence was committed as `deec4d5c2363888d31f6af211e3c33037b88886f` and checksums verified before starting. Exact original argv/env/artifact identities are retained in `mmq/results.json`; only split and suffix allocation reserve differ from K37. CMP logical CUDA0/physical1 owns layers [0,36); RTX logical CUDA1/physical0 owns [36,48). No P2P.

| Policy | CMP | RTX |
|---|---:|---:|
| K36 allocation reserve, MiB | 1024 | 256 |
| K36 external free-memory guard, MiB | 640 | 256 |
| Previous K37 allocation reserve, MiB | 1024 | 1024 |
| Previous K37 external guard, MiB | 640 | 640 |

The deliberate RTX exceptions were previously user-authorized; they were not reduced further. Both units are binary MiB. Host start minimum6 GiB, reaction threshold5 GiB (4 GiB floor plus1 GiB) remained unchanged. Allocation reserves and sampled external guards are separate policies.

The same executable `build/cuda-short-profile/strata` SHA256 `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214` and K-quant library `f0476d2db4fc631d77b2923f35dacdc7dcbb80462e583356c85d773768a12ef4` were used. CUDA13.4.92 Release sm80+86; build identity is in `../2026-10-16-cmp170hx-rtx3090-short-profile/build-identity.json`. Runtime/build sources remain unchanged from0327ba27. Model/MTP/pack identities match K37; large weights were **not freshly rehashed**, retaining original hashes with size/mtime checks and fresh small-file hashes.

`STRATA_MMQ_KQ=1`, phase timing OFF, max_context262144, INT8 KV, fixed chunk256, MTP/spec2, prompt cache0, trim-stage weights, mmap/direct expert reads, direct PLE and no prefill borrowing remain unchanged. No runtime/layout/system/model-artifact modification.

| Owner | Expert slots | Resident expert bytes |
|---|---:|---:|
| CMP [0,36) | 18432 | 57,619,251,200 |
| RTX [36,48) | 6144 | 19,398,656,000 |

Startup confirms all24,576 experts resident, CPU complement/cache loans disabled, one3,993,600-byte load staging buffer.

## Requests, output and timing

One resident process ran the1,977-token warmup, original32,788-token sanity fixture, then **one** original261,682-token near-limit fixture. Every request freshly processed its full input, with zero prefix reuse/checkpoints. No repeated near-limit or MTP-off run.

| Request | Input | Output/cap | Prefill ms | Tokens/s | Observed TTFT s | Wall to DONE s | Decode ms | MTP accepted/offered |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Warmup | 1977 | 22/32 | 2608.5 | 757.907 | 2.635 | 2.838 | 225.9 | 10/11 |
| 32K | 32788 | 37/192 | 39590.0 | 828.189 | 39.611 | 39.989 | 391.2 | 18/23 |
| Near-limit | 261682 | 37/192 | 336360.8 | 777.980 | 336.411 | 336.818 | 424.4 | 18/24 |

Startup **167.713s** is separate (K37:172.172s). Following the32K rate828.189tokens/s, the bounded near-limit timeout was1800s using `max(1800,4*input_tokens/preceding_rate+300)`. Durable PP logs show sustained progress; no disk-progress watchdog. Monitoring continues during threaded request submission. Final PP261681 plus final seed-token path gives DONE full261682.

Read-only original fixtures and metadata are referenced with hashes in `prompt-references.json`, together with the preceding K37 manifest hash. They were generated with the actual model tokenizer from natural-text records in the original chat wrapper, not repeated-token proxies. Near-limit IDs SHA256 `67a876781b4fde7323b1b340a5e55700acb3f6741dc19d953d32229c42cf55ed`; text SHA256 `10c7eb3f6a8a385458ecbaee8d71e00f2bae6995b6d1664ba57f3d0d5ea6ecbc`. Needle offsets59/130846/261559; independent expected answer:

```text
ORIGIN: IRIS-4Q7-LANTERN
CENTER: CEDAR-8M2-QUARRY
FINAL: ORBIT-6V9-KETTLE
```

Both long answers match exactly, including generated IDs identical to K37 MMQ and original non-MMQ. Input+cap261874 fits262144; actual input+output261719. Allocated capacity does not mean262144 input tokens were processed. Recall is not broad quality.

32K prefill was1.0141x K37. Near-limit decode424.4ms versus K37 421.7ms did not improve; MTP18/24 matches K37. Historical non-MMQ prefill821160ms gives2.4413x versus K36, but that older run used a different rebuilt executable and uncontrolled clock/cache conditions. The primary K36/K37 comparison uses identical current executable/settings except the documented placement and RTX policies. Neither comparison controls order, caches, thermals, clocks, desktop or host variation.

## Memory, reads, PCIe and cleanup

Overall/near-limit sampled minima: RTX **331,022,336 bytes /315.6875MiB**; CMP **2,644,836,352 bytes /2522.3125MiB**. RTX warmup343.6875MiB,32K319.6875MiB, near-limit315.6875MiB demonstrate why short success alone did not establish long fit. CMP margin above640 was1882.3125MiB. No guard/admission/correctness failure or OOM occurred.

Minimum host available **12,269,457,408 bytes /11.4268GiB**, max RSS **5,119,123,456 bytes /4.7676GiB**. Process swap0; system swap maximum1,310,720 bytes. About1s sampling plus forced post-request observations; max gap1.100020s. These are **sampled extrema, not guaranteed peaks or hard OOM protection**. The narrow RTX margin is a material limitation, not a general recommendation to reduce reserves.

First warmup file-tier line includes startup-cumulative77,024.6MB expert reads. Both32K and near-limit request intervals explicitly report **0.0MB expert reads**; cumulative expert bytes do not increase. Near-limit OS read10,336.2MB/0 major faults, so other file-backed/PLE traffic remains; no zero-storage claim.

Whole-device PCIe telemetry: near-limit309 samples/device, printed MB/s. RTX RX mean54.72/max645, TX10.92/max87; CMP RX84.56/max238, TX54.00/max557. This is not phase-isolated/model-only transfer attribution or proof of saturation. Full raw telemetry and per-stage counters/extrema are retained.

Engine exited0, telemetry helper stopped, and post-run compute list was empty. `cleanup.json` records live post-run memory/process checks. No new GPU workload was launched for analysis.

## Reproduction and limits

From repository root:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-16-cmp170hx-rtx3090-k36-mmq-long-context -p test_long.py -v
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k36-mmq-long-context/run-long.py --mode mmq
```

Five CPU tests and the guarded GPU run passed. The runner refuses overwrite. From this directory `python3 -B analyze.py > summary.json` regenerates the offline checks/comparison; `sha256sum -c SHA256SUMS` checks evidence. Analysis asserts exact permitted argv changes, artifact/env identity, full fresh prefill, exact recall/IDs, residency, expert-read intervals and sampled guards.

One successful K36 MMQ near-limit test, not repeated-nearmax, general-quality, best-split or expert-service acceptance. CUDA host only; HIP/SYCL untested. No new optimization, agents, pushes or merges. Broader Fossil ticket5f8e4f2509 remains open.
