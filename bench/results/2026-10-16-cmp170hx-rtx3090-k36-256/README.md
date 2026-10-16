# User-authorized K36 attempt: RTX monitor 256 MiB

2026-10-16. **K36 was attempted but the engine refused full-residency admission before READY.** No prompt, answer or performance comparison completed. Lowering the external monitor threshold alone does not change the engine's allocation/admission reserves.

The user's request was: `lets move 3090 threshold to 256Mb and run K36`. This was interpreted as binary MiB, consistent with previous monitors. RTX physical index 0 had a 256 MiB stop threshold; CMP physical index 1 retained 640 MiB. Host start remained 6 GiB available; stop remained 5 GiB, preserving the 4 GiB floor plus 1 GiB reaction margin. Engine `--vram-reserve-mib 1024 --vram-reserve-later-mib 1024` were unchanged. No permission to lower those separate allocation reserves was inferred.

## Actual attempt

From the repository root:

```sh
python3 -B -m unittest discover -s bench/results/2026-10-16-cmp170hx-rtx3090-k36-256 -p test_short.py -v
python3 -B bench/results/2026-10-16-cmp170hx-rtx3090-k36-256/run-short.py --mode mmq
```

The four pre-run guard tests passed. The GPU command failed, engine exit **1**, during startup after loading CMP's 18,432 experts in 141.6 seconds. Exact engine diagnostic:

```text
CUDA1 needs 19398656000 bytes for 6144 assigned experts, but 20813905920 bytes are free (18844749876 after the stage reserve, exact owned-prefill scratch and draft binding); refusing partial ownership
```

The RTX required **19,398,656,000 expert bytes**, but usable capacity after existing reserves was **18,844,749,876 bytes**: a **553,906,124-byte deficit (528.246 MiB)**. The current monitor never fired; this was the engine's independent admission check (`src/program/generate.cpp:5798-5812`). Raw free before suffix expert allocation was 20,813,905,920 bytes, with 1,969,156,044 bytes reserved by that calculation. The preceding log explicitly prices 155,189,248 bytes of exact owned-prefill scratch and 293,531,596 bytes for MTP binding, in addition to the stage reserve which already includes the verifier.

Thus the previous short-run estimate of roughly 336 MiB remaining after K36 was **not a startup admission guarantee**. A 256 MiB external monitor would permit that estimate, but the unchanged engine reserve does not. No bypass, partial expert ownership, reserve reduction, second GPU attempt or runtime modification was made. The request cannot reach a K36 timing result with these admission settings.

The monitor also encountered an exit-time `/proc/PID/io` `PermissionError` as the engine exited. This secondary race is retained verbatim in `progress.txt` and `mmq/results.json`; it must not obscure the engine's concrete capacity diagnostic. `summary.json` distinguishes both. The new harness was subsequently corrected to record an explicit exited return code when `/proc` becomes unreadable **only if the process has exited**; a permission failure for a live process still raises. Two CPU tests cover this correction. No GPU rerun was needed to reconfirm the deterministic admission rejection. The checked-in runner therefore includes that post-run correction; all actual raw logs/results are untouched.

## Identity, safety and limits

`mmq/results.json` retains exact argv/environment, source parent `435fdd1c`, executable/cache/library hashes, model/pack/MTP identities, guards and timestamps. Compared with the successful K37 MMQ arm, only `--layer-split 37` becomes `36` in engine argv; environment remains identical (`STRATA_MMQ_KQ=1`, timing disabled). Binary SHA-256 remains `cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214`. Model weights were not rehashed; original manifest identities and size/mtime checks were reused, small metadata checked freshly.

The intended schedule was identical 502/994/1,977-token natural-text retrieval prompts, one warmup plus three repetitions each, output cap 32. Exact tokenizer-derived prompt artifacts are included, but **none was submitted** because READY was never reached. CMP ownership was confirmed; all 24,576 experts were **not** resident because suffix admission failed.

Before launch there were no compute users, RTX 24,096 MiB / CMP 64,897 MiB unloaded free, and host available 18,791,596,032 bytes. Minimum sampled host available was 16,797,921,280 bytes; process swap zero; max system swap 1,310,720 bytes. Maximum sample gap was about 1.100 seconds. Samples cannot guarantee against transient OOM. No OOM or threshold event occurred; no decode or long prompt ran. `cleanup.json` confirms no compute process and released allocations.

Offline `analyze.py` checks exact binary/model/config identity against preserved K37, parses the admission bytes and retains the failed outcome. Six CPU tests pass, including independent asymmetric thresholds, unchanged host guard, exited-process race and live-process error handling. New and prior evidence checksums pass. CUDA host only; HIP/SYCL untested. Broader Fossil ticket `5f8e4f2509` remains open. No push, merge, split optimization claim or scratch-layout change.
