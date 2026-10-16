# CMP 170HX + RTX 3090: preserved K37/chunk256 baseline

This is the working baseline for Fossil ticket `d423fa`, not the fastest configuration.
Do not replace its artifacts or logs with later tuning results. Compare future configurations against a
same-day rerun of this baseline; the four-token diagnostic is not a performance or quality benchmark.

## Launch and artifacts

`launch.sh` retains the exact engine arguments, explicit environment overrides and stdin used by the successful
run on 2026-10-16. Paths are the original local paths, not downloadable artifacts. Run from the repository root.
The original wrapper removed `CUDA_VISIBLE_DEVICES`; `--gpu 1,0` selects CMP first, RTX last.
Unrecorded inherited environment variables cannot be reconstructed retrospectively.

`guarded-launch.py` preserves the session's monitor after its exit-time `/proc/PID/io` race fix. It contains the
same engine invocation and input, a 900-second limit, a 120-second no-disk-progress limit, 100 ms RAM polling,
and a 5 GiB stop threshold above the user-authorized 4 GiB floor. It is a polling guard, not a hard memory
limit or an OOM guarantee. The successful run used that monitor before the race fix. `launch.sh` by itself
does **not** provide those guards; use the guarded wrapper for a rerun. Its output goes to
`/tmp/strata-k37-guarded.log`, not to these immutable evidence files.

`artifacts.json` records SHA-256 and sizes of all four GGUF shards, every generated pack file (including
tokenizer metadata), every MTP file, the routing profile and the engine executable. The GGUF revision is
`38bb39ee97821de2c9009abb7e93950eec396e66`; the shard hashes are checked against `docs/UNSLOTH_Q4.md`.
Hashes were captured after the successful run, with no intervening artifact regeneration. Large artifacts
are not committed. Hashing uses bounded O_DIRECT reads so the model is not pulled into the OS file cache.
The source parent was `296b59d2f73c8f04a805df29c2fa97655adace78`; this evidence's commit preserves the implementation.
`build-config.txt` records relevant build settings and compiler/driver versions at preservation time.
The local binary is additionally retained in the originating session's `files/k37-baseline/` archive.

## Raw evidence and limits

- `engine.txt`: unedited engine stdout/stderr from the successful run, including `READY`, `T` and `DONE`.
- `monitor.txt`: unedited monitor output through process exit, with ten-second RAM, disk and GPU samples.
- `earlier-timeout.txt`: the preceding four-minute attempt; it timed out before startup completed.
- `ledger-chunk256.csv` and `ledger-chunk2048.csv`: sizing-command results, not measured physical allocations.
- `SHA256SUMS`: hashes of the retained evidence files (excluding itself).

Both cache verification checks passed: 18,944 experts on CMP and 5,632 on RTX, all 24,576 resident. The engine
reported `READY 262144 stop` with INT8 KV capacity. It processed `GEN 4 1,2,3` and emitted four tokens, then
received `QUIT`. With everything loaded it reported 1,110 MiB free on CMP and 1,713 MiB on RTX. One-/two-token
graph captures succeeded. Although `--spec 2` was requested, suffix lookup expanded verify capacity to four;
MTP remained capped at two.

Primary fill took 232.9 seconds at 254 MB/s; startup and the request took about seven minutes. Printed host
RAM samples stayed above 20 GiB. The monitor encountered an exit-time `/proc/PID/io` permission race, so no
final subprocess return code or exact 100 ms RAM minimum was saved. `engine.txt` contains `DONE` without an
engine error, and the process exited with GPU allocations released. Do not infer a recorded zero exit code.

This establishes full-residency startup and short decode at 262,144-cell capacity, **not a full 262K-token
prompt test**, arbitrary-request peak-memory bound, quality result or fastest-split result. The measured
selection rule and component estimates/reserves are in [MULTI_GPU.md](../../../docs/MULTI_GPU.md).
