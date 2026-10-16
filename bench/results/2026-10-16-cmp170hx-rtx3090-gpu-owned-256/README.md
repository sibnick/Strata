# Static GPU-owned expert run: CMP 170HX + RTX 3090

This is a guarded, real-tokenized 256-token prefill and decode run for the
opt-in `--gpu-owned-experts` implementation in source commit `6694c246`.
The preserved K37/chunk256 startup baseline is in
[`../2026-10-16-cmp170hx-rtx3090-k37/`](../2026-10-16-cmp170hx-rtx3090-k37/);
its evidence was not changed.

## Artifacts and build

The model is Qwen3.8-Flash-Next UD-Q4_K_XL, GGUF revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, using the nested shard path
`/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/`.
The four shard sizes and SHA-256 values, plus the pack, MTP and profile
identities, match the preserved baseline's
[`artifacts.json`](../2026-10-16-cmp170hx-rtx3090-k37/artifacts.json).
The local `build/pack/ud-q4-k-xl` link points to the existing pack; no model,
pack, profile or MTP source artifacts were modified.

The CUDA engine was built with CUDA 13.4.92 for `sm_80` and `sm_86`:

```sh
cmake -S . -B /tmp/strata-gpu-owned-cuda \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES='80;86'
cmake --build /tmp/strata-gpu-owned-cuda --target strata -j2
```

The host is a 64 GiB CMP 170HX and 24 GiB RTX 3090, with CUDA device order
1 = CMP and 0 = RTX. The guarded runner and `monitor.txt` preserve the full
engine command and environment. The run used CMP-first `--gpu 1,0`,
`--layer-split 37`, `--max-context 262144`, `--kv int8`, `--prefill 256`,
`--no-prefill-borrow`, `--mmap-experts`, `--gpu-owned-experts`, the complete
profile, direct PLE reads, a 1,024 MiB per-card reserve, and one CPU worker.
It used 4 GiB host pin/headroom settings, checked for other GPU compute
processes before launch, sampled host and GPU memory every two seconds, and
stopped at a 5 GiB host reaction threshold, a 1,024 MiB GPU free threshold,
or 20 minutes. The prompt IDs are preserved in `prompt.ids` (SHA-256
`703319b41b3ae612f36af6f2c9084a8c0228f79f5015e195269f8e0c9024093a`).

## Result

The monitored process exited successfully (`RESULT 0 completed`) in 430.4
seconds. Startup allocated one reusable 3,993,600-byte host staging buffer
and synchronously filled all 24,576 expert slots:

| CUDA stage | Device | Layers | Experts | Resident bytes |
|---|---|---:|---:|---:|
| 0 | CMP 170HX | `[0,37)` | 18,944 | 59,192,115,200 |
| 1 | RTX 3090 | `[37,48)` | 5,632 | 17,825,792,000 |

The primary fill took 312.4 seconds at 190 MB/s. The engine logged that all
experts were resident, the CPU complement and prompt-cache loans were
disabled, startup file-tier reads accounted for 77,024,563,200 bytes, and
no file fallback was expected after startup.

The engine reached `READY 262144 stop` and processed all 256 prompt tokens
in 1,146 ms (about 223 tokens/s). The configured maximum was four generated
tokens; generation stopped after one token, `248046` (`<|im_end|>`), in
23.2 ms (about 43 tokens/s). This verifies one real decode iteration, not a
non-empty answer or a quality result. The request-end tier counter reported
zero file blobs. The 77,024.6 MB file-byte counter is the startup fill; the
first-request OS-I/O delta starts at process launch and therefore includes
startup I/O, not just request-time I/O.

During the run, sampled `MemAvailable` reached a minimum of
17,894,141,952 bytes (about 16.67 GiB). At the lowest sampled GPU-free point,
the RTX 3090 had 1,700 MiB free (22,425 MiB used) and the CMP 170HX had
1,075 MiB free (63,837 MiB used). The CMP minimum was only 51 MiB above the
configured 1,024 MiB stop floor; these are sampled values, not a hard
resource bound. After process exit, the GPUs returned to 23,762 MiB and
64,897 MiB free, respectively; host `MemAvailable` was 21,381,348 KiB.

The earlier sm_86-only attempt is retained in
`attempt-sm86-only-engine.txt` and `attempt-sm86-only-monitor.txt`; it safely
stopped before cache allocation because the CMP 170HX requires sm_80. The
successful run used the subsequent sm_80 + sm_86 build.

## Checks and limits

`gpu_ownership_test` and `split_rules_test` passed in the CPU helper build;
`file_expert_source_test` passed in the CUDA build. The CUDA engine build
passed for both architectures. HIP and SYCL compilers were unavailable, so
those backends were not built. The run exercised a 256-token prompt, not a
262,144-token prompt; full-context correctness and performance remain
separate work. No speedup or full-context claim is made.
