# Same-prompt GPU-owned comparison

This guarded follow-up compares `--gpu-owned-experts` with the same build
and launch settings with that flag omitted. It uses the CMP-first K37 split,
the preserved UD-Q4_K_XL artifacts and a meaningful 256-token Qwen chat
prompt. It does not replace or modify the synthetic three-token run in the
preserved baseline directory.

## Prompt and candidate identity

`prompt.txt` is the user message. The exact formatted input is
`"<|im_start|>user\n" + prompt.txt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"`.

This is the model pack's chat-template form for a user message followed by
an assistant generation prompt with thinking disabled. It was encoded with
`tools/strata_tokenizer.py` from the pack tokenizer using
`parse_special=True`; `prompt.ids` contains exactly 256 tokens and its
SHA-256 is `f22c0a1e3197ca995477fae9c638c3c8267837b4025f082ae9f320f5724cb042`.
The 64 generated token IDs are in `generated.ids`; their decoded text is in
`generated.txt`.

The owned and flag-off processes used the identical CUDA executable,
SHA-256 `06a89cd0a9e8725ead0da3b55cc85c4cc11e8d33e4864d2e97b3aeec354ad1eb`,
built from source commit `0327ba279ba50a34894f6aac16e5fe3c21106611`.
Build directory: `/tmp/strata-gpu-owned-cuda`; CUDA 13.4.92, Release,
`sm_80;sm_86`. The CMake cache SHA-256 is
`fee6be1a4eefc5da96c3b06f9adceb2f88e1f4b21485ec213c47aac95bce86c7`.
Both GPUs ran NVIDIA driver 610.43.03. `build-identity.txt` records these
values and artifact references; `run-manifest.txt` records the exact paired
commands, environment overrides, prompt identity and guard settings.

The four model shards, native pack, MTP files and routing profile are the
same paths and identities recorded in the preserved baseline's
[`artifacts.json`](../2026-10-16-cmp170hx-rtx3090-k37/artifacts.json)
(manifest SHA-256
`de3b5214d60af2709d19baf8deed4f1c89e35006e26023882794437011d2ddd6`).
The profile SHA-256 is
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
The tokenizer vocab, merges, token types and chat template also match the
manifest. The original artifact files were not modified.

## Paired result

`compare-run.py` records each exact command and environment and runs the
owned mode first, then the same binary with only `--gpu-owned-experts`
omitted. Both used `--gpu 1,0 --layer-split 37 --max-context 262144 --kv
int8 --prefill 256 --no-prefill-borrow --mmap-experts`, the same MTP,
profile, prompt IDs, worker count, PLE settings, reserves and runtime
environment. Each run was independently guarded by a 5 GiB host-memory
reaction threshold, a 1,024 MiB GPU-free threshold and a 20-minute limit;
the runner refused to start if another GPU compute process was present.
Monitoring sampled every two seconds, so the thresholds are not hard
resource guarantees.

Both processes reached `READY 262144 stop`, read all 256 prompt tokens and
generated the same 64 token IDs, with no token divergence. The committed
runner checks the token count and equality before reporting success; replaying
that check against the saved engine logs also passed. The shared text
is a coherent answer explaining the distinct likely effects of checking
soil moisture and adding straw, followed by a limitation. Generation ended
at its 64-token length cap after “the volunteers did not”; this is a
meaningful partial continuation, not a completed answer, quality judgment
or full-context test.

| Mode | Elapsed | Host minimum available | RTX 3090 minimum free | CMP 170HX minimum free |
|---|---:|---:|---:|---:|
| Owned flag enabled | 296.3 s | 17,043,320,832 bytes | 1,679 MiB | 1,053 MiB |
| Same binary, flag omitted | 162.4 s | 15,713,546,240 bytes | 1,686 MiB | 1,075 MiB |

The CMP's owned-mode minimum was 29 MiB above the monitor stop threshold.
These sampled minima are close to the configured guard; do not treat them
as extra free capacity or an OOM guarantee. The process exited and released
the GPU allocations; host availability returned above 20 GiB.

The owned run logged 18,944 experts resident on CUDA0/CMP for layers
`[0,37)` and 5,632 on CUDA1/RTX for `[37,48)`, all 24,576 resident. It used
one reusable 3,993,600-byte host staging buffer with synchronous profile
fills. Its last-stage capacity preflight withheld 83,886,080 additional
bytes for exact owned-prefill scratch beyond the split estimate and
293,531,596 bytes for the MTP bind; the stage reserve also withheld 96 MiB
for verifier buffers. The primary-stage preflight uses exact owned-prefill
sizing plus its allocator margin, the configured safety reserve, 96 MiB for
the verifier and applicable MTP/pipeline allocations. Prompt cache loans
and the CPU expert complement remain disabled. These checks establish the
tested placement and its startup allocations, not a hard bound on unrelated
driver use or arbitrary future request shapes.

At the request summary, the expert-tier line reports `files 0 blobs` and a
77,024.6 MB read counter labelled “since the start”; those bytes are the
startup profile fill, not a prompt-path reread. The separate “file tier I/O
this request” line compares counters from process start to the first
request, so its OS read delta also includes startup and cannot be interpreted
as request-only storage I/O. The engine's startup log reports all assigned
slots resident and no expected file fallback after startup.

The elapsed times and tokens/s are retained only as run diagnostics, not a
performance comparison: these were sequential cold-start processes and
their filesystem-cache state differed. The flag-off run uses the candidate
binary without the new flag; it is not a separate pre-change executable.
This verifies identical output on this prompt and setting, not universal
bit-for-bit parity or a full-context result.

## Checks

- `cmake --build /tmp/strata-gpu-owned-cuda --target strata -j2` — passed
  for `sm_80` and `sm_86`.
- `/usr/bin/ctest --test-dir /tmp/strata-gpu-ownership --output-on-failure -R 'gpu_ownership|split_rules'`
  — 2 passed.
- `/usr/bin/ctest --test-dir /tmp/strata-gpu-owned-cuda --output-on-failure -R file_expert_source`
  — 1 passed.
- `python -m py_compile bench/results/2026-10-16-cmp170hx-rtx3090-gpu-owned-comparison/compare-run.py`
  — passed; saved-log check found 64 IDs in each run, exact equality, and a
  match with `generated.ids`.
- `git diff --check` — passed.

HIP/SYCL were not built because their compilers were unavailable. No
262,144-token prompt was run. `SHA256SUMS` covers the prompt, runner,
generated output, build/run identity, logs, monitors and this README
(excluding the checksum file itself).
