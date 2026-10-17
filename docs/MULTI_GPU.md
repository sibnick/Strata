# Strata on two or more GPUs (layer split)

One model can run across several NVIDIA cards in one PC. The layers are split into contiguous ranges, one per GPU:
the first card runs layers 0 to K-1, the next card runs K onward, and so on; the last card also runs the output head
and the draft (MTP) layer. Each card keeps an expert cache for **its own layers only**, so two cards hold about twice
the experts one card holds - for the Coder model on a 16 GB + 24 GB pair, nearly all of them, which is where the
speed comes from (decode then barely touches the CPU pool).

This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per
verify window (a few hundred KB through pinned RAM), not twice per layer. No NVLink or peer-to-peer access is
needed; cards on x4 or x1 slots work, and the PCIe share of each card is probed on its own link. A `--pcie-frac` you give
is every card's share and skips those probes; there is no per-card setting yet.

## Sizing UD-Q4_K_XL on a CMP 170HX and RTX 3090

The optional CUDA target `strata-split-memory` reads the GGUF headers and native pack index, not the expert
payload. It compares CMP-first boundaries 36, 37 and 38 using the engine's session and owned-prefill sizing
functions. It does not change inference defaults. Build and run it from the repository root:

```sh
cmake --build build --target strata-split-memory -j2
build/strata-split-memory --pack build/pack/ud-q4-k-xl \
  --native /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp /path/to/mtp/rt --context 262144 --chunk 256 > split-memory.csv
python tools/test_split_memory.py
```

This is a sizing command, not a CPU-only command: the existing metadata loader initializes CUDA and temporarily
allocates 24 MiB of pinned staging buffers even when every payload is skipped. A run on this machine reduced
available host RAM by about 151 MiB at its sampled low point. It allocates no expert cache, weights or full-context
session. Check host headroom before running it.

The CSV reports bytes, with stage 0 = CMP and stage 1 = RTX. The stage -1 rows contain shared file sizes and
individual layers' expert payloads. The canonical arena and session rows are exact logical allocation sizes;
the projection, head and prefill rows are allocation estimates, including the sizing functions' rounding and
slack. Each total additionally includes a 64 MiB projection-padding allowance, a 96 MiB verifier allowance,
384 MiB for MTP state/workspaces beyond its files and subset head, and 1 GiB for additional CUDA use and safety.
These allowances are **not measurements or proofs of fit**. MTP files and subset-head costs are only added on
the last stage; the `mtp_files_payload` row is informational, already included in `mtp_total_with_runtime_reserve`.
The context and chunk numbers are capacities, not a completed prompt measurement.

For the four UD-Q4_K_XL shards at revision `38bb39ee97821de2c9009abb7e93950eec396e66`, the expert payload is
77,017,907,200 bytes. At 262,144 cells with INT8 KV, the two session arenas total 4,000,221,184 bytes. With owned
prefill buffers and a 96-blob ring, the estimated per-card requirements, including the above allowances, are:

| Boundary | CMP bytes, chunk 256 | RTX bytes, chunk 256 | CMP bytes, chunk 2048 | RTX bytes, chunk 2048 |
|---|---:|---:|---:|---:|
| 36 | 65,785,561,088 | 25,436,781,804 | 66,777,513,984 | 26,428,734,700 |
| 37 | 67,456,991,232 | 23,761,157,356 | 68,448,944,128 | 24,753,110,252 |
| 38 | 69,130,518,528 | 22,087,630,060 | 70,122,471,424 | 23,079,582,956 |

On this machine with NVIDIA driver 610.43.03, a bounded 16 MiB pinned-buffer DMA probe found CUDA/PCI order
0 = RTX 3090, 1 = CMP 170HX. Use engine `--gpu 1,0` for CMP-first. The RTX had 24,687,935,488 bytes free and
copied 1 GiB H2D at 24.54 GB/s (Gen4 x16 sampled); the CMP had 67,727,917,056 bytes free and copied at
0.790 GB/s (Gen1 x4 sampled). Peer access was unavailable in both directions. These are small DMA-load
measurements, **not full-model under-load measurements**.

The selection rule is to reject a boundary if either card's requirement exceeds its measured free memory;
never reduce the safety allowances merely to make it fit. Against that free-memory snapshot only K37 with
chunk 256 passes; its remaining margins after the listed allowances are 270,925,824 bytes on CMP and
926,778,132 bytes on RTX. All three boundaries fail with a uniform chunk 2048. K37/256 was therefore selected for a
guarded test, **not as a measured best split**. Its accounting assumes
`--trim-stage-weights --kv int8 --max-context 262144 --spec 2 --prefill 256 --no-prefill-borrow`,
`STRATA_SPLIT_RING=96`, the default 32,768-cell MTP window, and no batch, vision or KV streaming.

### Different prefill chunks per stage

For an explicit, fully owned multi-GPU `--serve` layer split, the opt-in
`--prefill-stage-chunks N[,N...]` sets one chunk per GPU stage, in GPU order.
Its first value must equal `--prefill`. For example,
`--prefill 2048 --prefill-stage-chunks 2048,1024` keeps a 2,048-token chunk on stage 0 and
uses 1,024-token chunks on stage 1. Each stage's device scratch and host
handoff buffers are sized for its own chunk, and startup admission prices each
stage separately. If a producer hands off more rows than the next stage's
chunk, the consumer processes that handoff in its own sub-chunks. The option
does not apply to borrowed prompt buffers; omitting it retains the existing
uniform `--prefill` behavior.

This changes prompt-prefill scheduling and the cadence of the last-stage MTP
prompt-KV updates. It does not change session/KV capacity, model ownership or
decode settings. Verify the handoff subdivision, fresh-input accounting and
historical output IDs on the exact workload before relying on the mixed
configuration; a sizing pass is not a runtime or memory-safety guarantee.

**K36/K37 source sizing (2026-10-17).** With a 262,144-cell INT8 session,
2,048/1,024 stage chunks, zero discretionary CUDA reserve and the sampled
idle free memory below, the source ledger reports:

| Split | Stage / GPU | Chunk | Owned prefill estimate | Required before model load | Margin after mandatory 512 MiB prompt-init headroom |
|---|---|---:|---:|---:|---:|
| K36 | CMP 170HX | 2,048 | 1,430,257,664 B | 65,779,269,632 B | 1,733,296,128 B |
| K36 | RTX 3090 | 1,024 | 1,025,507,328 B | 25,025,740,012 B | -297,172,204 B |
| K37 | CMP 170HX | 2,048 | 1,430,257,664 B | 67,450,699,776 B | 61,865,984 B |
| K37 | RTX 3090 | 1,024 | 1,025,507,328 B | 23,350,115,564 B | 1,378,452,244 B |

The free-memory snapshot was 68,049,436,672 B on the CMP and 25,265,438,720 B
on the RTX. Although both K36 raw ledger totals are below that snapshot, the
RTX does not meet the unchanged mandatory 512 MiB prompt-initialization
headroom check for 2,048/1,024. That bounded K36 GPU run was not started. K37
passes this offline check but was not run with those mixed chunks. A separate
2,048/512 K36 attempt and its measured results are recorded below.

The requested test policy is zero reserves on both GPUs and no external
free-VRAM abort, while retaining the host start-at-6/stop-below-5 GiB guards
and mandatory CUDA/allocation admission. Removing the discretionary reserves
does not override the failed 2,048/1,024 prompt-init check above.

**K36 2,048/512 follow-up (2026-10-17).** Reducing only stage 1 to 512 lowered
its source-priced scratch from 1,025,507,328 B to 507,510,784 B. The same
512 MiB prompt-init admission then passed on both stages; at the idle snapshot
the RTX margin beyond that headroom was 220,824,340 B. K37 also passed offline
admission with these sizes, but was not run.

The bounded K36 run passed one warmup, three 1,977-token requests with a
32-token generation cap, and one 32,788-token request. It matched the
historical output-ID fixtures and recall checks, consumed the complete fresh
input on each request with zero prompt reuse, and the larger first-stage
handoff was consumed as 512-token sub-chunks. After the warmup, all four
measured requests reported 0.0 MB of expert source reads. The warmup itself
reported 77,024.6 MB of expert reads; this is not counted as a measured
post-warmup request and is retained in the raw counters.

The three short-prefill times were 1,410.7, 1,396.8 and 1,409.0 ms (median
1,409.0 ms); the 32,788-token prefill was 15,529.8 ms. The historical uniform
1,024-token K37 run reported a 1,368.6 ms short median and 20,237.2 ms for its
32,788-token request. This is descriptive only: the split differs and the
runs were not contemporaneous.

The minimum sampled free VRAM during the K36 run was 256,835,584 B
(245 MiB) on RTX 3090 and 1,675,952,128 B (1,598.5 MiB) on CMP 170HX. Minimum
host `MemAvailable` was 16,599,871,488 B (15.46 GiB), above the 5 GiB stop
threshold. The engine and PCIe monitor exited with code 0, and no compute
process remained. The small observed RTX margin is a real risk; this one
successful run is not a memory-safety guarantee. The discretionary GPU
reserves were zero on both cards, the external free-VRAM abort was disabled,
and the mandatory 512 MiB prompt-init and CUDA allocation checks remained
enabled. The guarded run did not use near-limit, 128K or 262K prompts.

Raw logs, monitor samples, request outputs, source-ledger data, checks,
provenance and checksums are in
`bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/`.
CUDA targets `strata`, `strata_kq` and `strata-split-memory` were built.
HIP and SYCL were not tested.

### Starting the tested K36 configuration as a local API

The committed launcher and server config are in that evidence directory's
`api/` folder. Start the OpenAI- and Anthropic-compatible API with
`bash bench/results/2026-10-17-cmp170hx-rtx3090-k36-stage-prefill-2048-512/api/start-api.sh`
and stop it with the sibling `stop-api.sh`. It binds only to
`127.0.0.1:8080`; no API key is configured or needed for loopback clients.
The OpenAI base URL is `http://127.0.0.1:8080/v1` and the listed model ID is
`qwen3.8-flash-next-unsloth-ud-q4_k_xl`. The launcher requires 6 GiB host
`MemAvailable` before startup and supervises the service, asking it to stop
below 5 GiB. The engine's mandatory CUDA allocation/admission checks remain
active with zero discretionary reserves.

The first real API smoke test passed: `/health` reported the model loaded,
`/v1/models` listed the expected ID, and a short OpenAI-compatible completion
returned `API_READY`. After full engine startup it reported 330 MiB free on
the RTX 3090 with a low-free-memory warning; a later snapshot was 317 MiB.
The smoke request passed, but this leaves little room for competing GPU work.
The supervisor and API server PIDs and sanitized smoke result are recorded
in the evidence directory.

The server loads the model and may allocate its full 262,144-cell KV cache
at startup. Runtime validation so far covered only short prompts and one
32,788-token prompt; near-limit, 128K and 262K input handling remains
untested. The lowest sampled RTX free memory was 245 MiB, so do not treat
this as a full-context safety guarantee. Logs and PID are under the launcher's
`api/` directory (`server.log`, `engine.log`, `server.pid`).

### Root runner for the current K36 source

`run-170hx-3090.sh` at the repository root builds the current checkout for
CUDA sm_80 and sm_86, checks the K36 memory ledger using the Cyrillic MTP
draft vocabulary, and starts a guarded loopback API. The recipe preserves the
bounded-tested K36 settings above: CMP first, split 36, INT8 KV at 262,144
cells, prefill chunks 2,048/512, resident GPU-owned experts, MMQ K-quant and
batched gather. This is the best validated recipe for this machine, not a
claim that every setting is globally optimal.

Cyrillic support here is **MTP draft-vocabulary coverage**, not n-gram/PLE
processing. The runner uses the checked-in `data/draft_vocab_cyrillic.bin`
with `--mtp-draft-vocab`; it does not alter model tokenization, the n-gram
table, or PLE settings.

From the checkout root, configure the model data paths if they are not at the
defaults. The defaults are `build/pack/ud-q4-k-xl` for the current checkout
and `$HOME/projects/Strata-data` for the external model data. The expert
profile defaults to `data/expert-profile.bin` in the checkout; set
`STRATA_EXPERT_PROFILE` if it is stored elsewhere. The runner fails with the missing path
instead of silently using an older binary or another worktree's files:

```sh
export STRATA_PACK="$PWD/build/pack/ud-q4-k-xl"
export STRATA_NATIVE="$HOME/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
export STRATA_MTP="$HOME/projects/Strata-data/mtp/rt"
export STRATA_EXPERT_PROFILE="$PWD/data/expert-profile.bin"
./run-170hx-3090.sh build
./run-170hx-3090.sh check
./run-170hx-3090.sh start
curl http://127.0.0.1:8080/health
./run-170hx-3090.sh status
./run-170hx-3090.sh stop
```

`build` compiles `strata`, `strata_kq`, and `strata-split-memory` into the
ignored `build/cuda-170hx-3090/` directory from the current source, with
`STRATA_MMQ_TAB=OFF`. `check` runs the source ledger with the same Cyrillic
draft-vocabulary file passed to inference, requires at least 6 GiB host
`MemAvailable`, verifies physical GPU order, and checks each K36 stage against
its free VRAM plus the mandatory 512 MiB prompt-init headroom. The ledger
initializes CUDA and allocates bounded metadata/pinned staging memory; it
does not load model weights or start inference. Check and startup refuse to
overlap an existing NVIDIA compute process. The runner also rejects an engine
binary missing the recipe's required flags, with an instruction to rebuild it.
`start` repeats admission using a fresh snapshot. It reports the API ready only after `/health` returns
`loaded: true`; until then it reports startup in progress. Engine stderr is
saved separately from the API/supervisor log. Runtime config, ledger output,
PID and logs stay under the ignored build directory.

The runner keeps discretionary CUDA reserves at 0 MiB, adds no external
free-VRAM abort, binds only to `127.0.0.1:8080`, and refuses startup below
6 GiB host availability; its supervisor asks the API to stop below 5 GiB.
Engine-side CUDA allocation/admission checks remain on. The RTX 3090 had only
245 MiB free at the low point of the pre-merge bounded K36 run, and upstream
changes have not been validated with inference on this machine. The ledger
must pass on the actual machine before starting; a pass is not a guarantee
against later memory pressure. Start with short prompts. Near-limit, 128K and
262K prompt handling are not validated. The merged-source runner has not yet
been smoke-tested, so use `check` and review its report before `start`.

On the measured machine snapshot (2026-10-18), the current-source ledger
including the 58,963-token Cyrillic subset passed with 254,568,116 B of RTX
margin after mandatory prompt-init headroom; CMP margin was 1,733,296,128 B.
The RTX margin is only about 243 MiB. This is an offline estimate, not a
loaded-engine measurement, so do not treat the pass as a guarantee that
startup or inference will fit.

Host RAM is a separate constraint. `--resident-budget-gib` caps only the resident expert copy. This model's
PLE file is 28,800,138,240 bytes: use `--ple-io direct` to avoid loading it into RAM. The mapped token embedding
is 675,430,400 bytes; its accessed pages can become resident. Expert assembly normally retains about 257
largest-blob buffers (1,026,355,200 bytes here), but that is not a hard pool cap. Direct expert reads under the
budgeted mode require `STRATA_UNBUFFERED_LOAD=1`: the default heuristic can choose buffered reads when a
recurring subset fits, even when the entire model does not. File-cache growth, CUDA/driver allocations and
other runtime caches are **not** bounded by the expert budget.

A guarded K37 retry with a 1 MiB expert budget, direct reads, 1,024 PLE cached rows and one CPU worker was
stopped during dense-weight loading when available host RAM crossed a 25 GiB reaction threshold
(minimum sampled 26,775,670,784 bytes, above the requested 24 GiB floor). No expert-cache fill, complete
startup or decode was validated. The external monitor is not a hard cgroup limit. The remaining host-memory
loss is unexplained; driver-pool retention is only a hypothesis. Recover host headroom before another full
attempt rather than lowering the floor or resetting a desktop GPU without approval.

**Subsequent live result (user-authorized 4 GiB host floor).** With the same K37/256 configuration, a
5 GiB reaction threshold and a longer startup limit, both profile fills completed: 18,944 primary experts
(55.13 GiB) and 5,632 secondary experts (16.60 GiB), all 24,576 experts resident. The primary fill took
232.9 seconds at 254 MB/s; startup and the short request took about seven minutes in total. The earlier
four-minute timeout was insufficient, not evidence of a stalled fill.

The engine reached `READY 262144 stop` and completed a three-token diagnostic prompt followed by four generated
tokens, with no expert-file reads during the request. With everything loaded, its `cudaMemGetInfo` trace
reported 1,110 MiB free on CMP and 1,713 MiB on RTX. The request captured one- and two-token graphs successfully;
the last logged capture began with 1,100 MiB free on CMP and 1,721 MiB on RTX. These readings have MiB
granularity, not byte precision. Host `MemAvailable` samples printed every ten seconds stayed at or above
21,780,000,768 bytes (20.28 GiB), and swap free was unchanged before/after. The process exited and both GPUs
released their model allocations. The monitor hit a `/proc/<pid>/io` permission race at exit, so its final
subprocess exit code and 100 ms minimum were not recorded; the engine log contains `DONE` and no engine error.

Allocations missing from the first expert-plus-session model are now visible: 1,097 MiB for the loaded MTP
files/state/workspace group, a later 275.7 MiB draft subset head, 84.4 MiB verifier buffers on **each** card,
and 518 MiB consumed on CMP between the before-prefill and after-head/prefill trace marks. That last group
exceeds the standalone 346 MiB prefill estimate: do not mistake the component estimates for exact physical
driver accounting. The conservative total still leaves the measured above-1-GiB margins in this run.
Although `--spec 2` was requested, suffix lookup expanded the verify capacity to four; MTP stayed capped at
two. The live evidence therefore covers that larger verifier configuration, not just a two-token allocation.

For this hardware/artifact/configuration, **K37/256 is a demonstrated full-residency startup and short-decode
configuration**, with at least 1 GiB reported free per card in the observed path. Keep that minimum margin,
remeasure free memory if other GPU applications change, and reject a run if either card falls below it.
This is not a fastest-split claim, a peak-memory bound for all requests, a 262K-token prompt test, or a quality
benchmark. Full-context correctness/performance remains separate work. The 4 GiB host floor was explicitly
authorized for the retry; it is not a new default or an OOM guarantee.
The [preserved K37/chunk256 baseline](../bench/results/2026-10-16-cmp170hx-rtx3090-k37/README.md)
retains the exact launch command, artifact hashes and raw logs. Use it as the working baseline, not as the
fastest configuration.

For the existing profile-based `--layer-split auto` search, `STRATA_SPLIT_LEDGER=1` also prints its selected
stages' usable bytes, session carve, later-stage weight estimate and profile-ranked cache capacities. This
describes that cache-based prediction, not the all-expert-resident rule above; the primary's weight carve is
zero there because its weights were already loaded. With the variable unset, the default path is unchanged.

## Experimental static GPU ownership

`--gpu-owned-experts` is an opt-in mode for an explicit `--layer-split` across GPUs. Use it with
`--mmap-experts` and a complete `--expert-profile`. Each device must fit every expert in its assigned layers, in
addition to the session, prompt buffers and runtime reserve. The engine reports the required and available bytes
and stops rather than starting with a partial cache. It does not affect the default cache or split behavior.

For example, the K37 split uses CUDA1 first and CUDA0 second on the machine described above:

```sh
STRATA_UNBUFFERED_LOAD=1 build/strata generate \
  --gpu 1,0 --serve --layer-split 37 --trim-stage-weights \
  --mmap-experts --resident-budget-gib 0.001 --gpu-owned-experts \
  --expert-profile data/expert-profile.bin --prefill 256 --no-prefill-borrow
```

The profile must rank every `(layer, expert)` pair exactly once. At startup the engine assembles one expert at a
time into a reusable buffer no larger than the largest expert blob, then uses a blocking host-to-device copy before
reusing it. Native GGUF expert rows are copied from their original tensor spans into that same buffer. The mode
does not build the CPU expert complement or use the ordinary growing blob-staging pool; mapped files still reserve
virtual address space, and the OS file cache and other runtime allocations are not hard-bounded by
`--resident-budget-gib`.

Prompt buffers remain separately allocated for the session. Prompt cache loans, head-aliasing, adaptive swaps,
elastic cache resizing, automatic splits, and peer/remote expert caches are not supported in this mode. The
runtime checks every expert slot after filling each device and does not continue if any assigned layer is missing.

## Using it

**Nothing to type.** `START-HERE.bat` (Linux: `./setup.sh`) lists your NVIDIA cards and says for each one whether
Strata can use it:

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) + GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) together   (recommended)
  2) GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) only
  3) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) only
Which GPUs? [1]:
```

When two or more cards can share the model, the two best together are recommended (the newest generation first:
it becomes the main card). A model installed on one card asks once, at its next start, whether to use both from
now on; the answer is kept.

**Choosing yourself** (at setup or at any start):

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; the first is the main one. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

**Not supported** (setup says so and names the cards that can be used instead):
- a card older than the RTX 20 series (compute capability below 7.5: GTX 10 and older), unless you name it: Pascal
  and Volta cards (Tesla P100 / P40, GTX 10, V100) are admitted when you choose them with `--gpus` (or `--gpu`,
  `--cuda 12`) and then run the experimental CUDA 12 engine, see [OLDER_GPUS.md](OLDER_GPUS.md). Two Tesla P40s
  (`"gpu": [0, 1]` in the config) ran the layer split in a community benchmark: IQ2_XS decode 19.6 tok/s on one card,
  34.8 on both (#1028, experimental, one report);
- a card with less than 8 GB of VRAM, together with others (each card holds a copy of the dense weights and its
  own prompt buffers) - unless you name it with `--gpus`: then setup says the risk and asks (`--yes` with the named
  cards goes ahead);
- Intel GPUs, and a mix of NVIDIA and AMD cards. (AMD cards share a model among themselves: `./setup.sh --backend
  hip --gpus 1,0`, see [AMD_HIP.md](AMD_HIP.md).)

Or edit an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

`"layer_split"` is `"auto"` (placed by each card's free VRAM) or the **first layer of each later card**: one rising
number per card after the first, not a count of layers per card. With 4 cards and a 48-layer model, `"24,36,42"`
(or `[24, 36, 42]`) puts layers 0-23 on the first card, 24-35 on the second, 36-41 on the third and 42-47 on the last.
The server checks it before the start and says what is wrong (0.1.39, #644).

**Card order with `"auto"`** (NVIDIA, #1352): the faster card (multiprocessors x max clock) goes **last** - the last stage runs the head, the draft layer and the verify, and a prompt chunk waits on it (a 4070 Ti SUPER + 5060 Ti read a 6K prompt in 50 s one way round and 15 s the other). Cards whose scores are within 5% of each other (two of the same model) keep your order; a manual split keeps it too. `"gpu_order": "as_given"` keeps the order you wrote under `"auto"` as well. The engine log line `layer split: card order ...` says when it changed.

**An unequal pair under `"auto"` (#2644).** The search prices a decode window, and it can give almost every layer to the card it
rates faster. A card with one or two layers holds a handful of experts and clamps the prompt chunk of the whole pipeline
(#1616), so the search then asks for at least 3 layers per card - which on a 4060 Ti 16 GB + 3060 12 GB (48 layers, IQ3_S)
was K=3: the 3060 ran 3 layers, 12 GB idle, 452 tok/s of prefill against 802 at 28 layers on the 3060 (the reporter's sweep of
`"layer_split"` 12, 18, 24, 28, 32: 582, 693, 762, 802, 803 tok/s; the 3060's 360 GB/s of memory bandwidth against the 4060 Ti's
288 counts for more than its SMs). On two cards the engine now first asks for a placement in which each card owns at least its
share of the layers by its room for experts, and the log line says which rule chose the split. Measured on 2x RTX A4000 made
unequal (stage 0 priced at 0.57 of the speed with 3.7 GiB less room, `STRATA_SPLIT_SPEED_SCALE=0.57` and `--vram-reserve-mib 4400`; 14.5K-token
prompt, IQ3_XXS, 5 interleaved pairs): the old rule chose K=3 (3 layers on stage 0), the new one K=20; prompt reading 945 -> 1,441 tok/s (+52%,
5/5 pairs), decode 50.2 -> 65.9 tok/s (+31%, 5/5). Equal cards and three or more GPUs are unchanged.
On x16 + x4 rigs the link-miss scaling (#2007, from a 4x gap) can also move `auto` off the balanced placement (2x RTX 3090, x16 + x4:
K=31 with miss cost x4.20, 114-119 tok/s decode against 130-136 at K=27, #2418): `STRATA_SPLIT_UNIFORM=1` (opt-in, with `"gpu_order": "as_given"`)
prices every link the same, as before 0.1.41.
Prefill that is flat in the prompt length (no gain from 4K to 16K) is the sign of an idle card; **on an unequal pair sweep
`"layer_split"` once** - a 4K-only benchmark cannot see it.

**Skip the split when the first card holds everything** (opt-in, 0.1.31): `"split_skip_if_fits": true` in the config
(engine flag `--split-skip-if-fits`, with `--layer-split auto`) runs on the first card alone when it holds every
profiled expert plus the context's KV, the draft layer and the reserve, and says so in the log; otherwise the split
stays. On an R9700 32 GB + RX 9070 XT the R9700 holds all of the Coder's experts: with the flag 4K prompts read at
1,776 tok/s instead of 1,244 (split) and decode runs at ~60 tok/s instead of ~51 (16K prompts ~5% slower than split).

**Short prompts on a split (0.1.32, #340).** 0.1.30 gave each card's prompt path a loan from its own expert cache,
refilled after every request; on cards that hold nearly all their experts that cost short prompts up to a third of
their speed. 0.1.32 refills all cards at once, uses a smaller streaming ring on a split, and lets a card with free
VRAM keep its own prompt buffers - the same output as 0.1.31, measured on an R9700 + RX 9070 XT: 2K prompts 993 ->
1,265 tok/s, 16K 1,852 -> 1,950, decode unchanged. `STRATA_SPLIT_OWN=1` (opt-in) gives every card its own buffers:
2K 1,450 and 16K 2,227 tok/s there, but a full card then keeps a different set of experts resident, so the output
differs from the default's (stable and coherent); `STRATA_SPLIT_OWN=auto` does that only where the buffers are at
most 12% of each card's VRAM.

**The idle card can help one-chunk prompts (opt-in, `STRATA_PREFILL_HELP=1`).** A prompt that fits one chunk runs the stages one after the other, so while
one card reads its layers the other idles. With it on, each stage hands a share of its streamed experts to the idle card: it
streams them over its own PCIe link into its own (lent) prompt buffers, computes their rows on the MMQ path and sends
them back - `--peer-device`'s peer streaming, without P2P (the activations and the rows go through mapped host memory,
read by copy kernels, so they do not queue behind the expert blobs on either card's copy engine). The share falls with
the prompt (0.41 of the streamed experts at 1.5K tokens, 0.32 at 3K) and is off from ~3.3K tokens, where the stages
overlap anyway and no share paid. Measured on 2x RTX 3090 (UD-Q4_K_XL, no P2P), prompt tok/s without / with: 1.1K 496 /
723, 1.5K 674 / 833, 2K 878 / 1,128, 2.5K 1,017 / 1,296, 3K 1,260 / 1,440; 4K and 8K unchanged, decode unchanged. It
costs no VRAM (the idle card's own prompt buffers) and ~110 KB of mapped host memory per token of the largest chunk it
helped (~360 MB at 3.3K tokens). The rows it computes round like a different MMQ grouping, so the output is not
bit-identical to the default's (it is repeatable: same prompt, same output), which is why it is **opt-in**:
`STRATA_PREFILL_HELP=1` turns it on. Native packs on the MMQ prompt path only (not with the fused prompt kernels,
`STRATA_PF_FUSED=1`, nor with `--peer-device`). With it on, `STRATA_PREFILL_HELP_FRAC=f` fixes the share.

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**The draft layer's prompt K/V in batches on a split (opt-in, `STRATA_SPLIT_MTP_BATCH=1`).** With `--mtp` the draft
layer sits on the last card, and each prompt chunk ends by filling that layer's K/V for the chunk's rows. On one card the
prompt path does it in batches (`Prefill::draft_kv`, a few large matrix products per chunk); on a split the drafter's own
pass always did it, one graph launch and four device copies per group of 8 rows (1,024 groups per 8,192-token chunk).
With the variable set, the last card's prompt path batches it as it does on one card. Restart-only; default off, so the
default start is unchanged. It needs a `--native` pack (the GGUF-form token table) and a paged or ring (`--kv-resident`)
draft K/V (`STRATA_MTP_BATCH_RING=0` sends a ring back to the old pass); otherwise it says why once and keeps the old pass:
`strata serve: draft layer prompt K/V batched on CUDA<d> ...` or `strata serve: STRATA_SPLIT_MTP_BATCH=1 declined, the
drafter's own pass runs: <reason>`. The draft layer's K/V come out of Q8_1 x Q8_0 MMQ instead of its own mmvq, so the
drafts, and how many are accepted, can move; the target's tokens are decided by the verify window.
`tools/split_mtp_batch_parity.py` compares the greedy texts of the two arms. Measured on one machine (2x RTX 3080 20 GB
at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL, `--layer-split 23`, `--batch-mtp`, two slots, `--prefill auto:16384`), two full
restarts per arm in the order off, on, off, on, the same binary in every arm: pooled medians of the prompt read were 1,996
tok/s without and 2,129 with at 25K tokens (+6.7%), 2,448 and 2,553 at 51K (+4.3%), 2,846 and 2,940 at 104K (+3.3%); all 8
on/off ratios of medians are above 1 (1.03 to 1.08), and the two off restarts differ by at most 2% at 25K and 0.3% at the
other sizes. The host's "after each chunk" time per request fell from 0.9-1.3 s to 0.1-0.25 s (one restart per arm). Greedy
output (`--adapt-every 0 --suffix-draft 0`, five prompts of 256 tokens and one 25.5K-token prompt) was identical in 6 of 6
prompts in all five comparisons, the two runs inside one process (A/A) included, and the drafts accepted were 826 of 1,210
in every run, so no difference in acceptance was visible. Decode is not shown to be unchanged: the medians with the variable
set were about 3% lower (pooled solo 82.6 -> 80.2 tok/s, two streams 93.2 -> 90.8), and the same configuration restarted
twice moved up to 4.7%, so two restarts per arm can neither confirm nor exclude it; the change touches only the prompt path.

**Each card loads only its own layers' dense weights** (0.1.39, PR #639) with explicit split points (`--layer-split
27`, not `auto`): every card used to keep a full copy (~3.4 GB for the Coder) though its stage reads only its own
layers, and the VRAM it frees goes to that card's expert cache (2x MI50 16 GB, Coder: 8,819 -> 10,626 experts in
VRAM, decode 39.2 -> 41.7 tok/s). Opt-in for now, on AMD and NVIDIA alike: `STRATA_STAGE_TRIM=1` (please report how
it goes).
A card holding more experts can change which experts run on the GPU, so the output can differ slightly from a run
without it.

**The resident RAM mode works on a split** (`--resident-experts`, the low-RAM mode). The RAM copy of the experts
leaves out the ones every card's cache holds, not only the first card's, and when the rest does not fit whole it keeps
the hottest by the expert profile over all the layers. An adaptive swap copies the evicted expert back into RAM from
the card that owns its layer. Before, `--resident-experts` with a split ran as `--mmap-experts`, and setup recommended
one card in the low-RAM mode; with an engine that has it (`RESIDENT_SPLIT_ENGINE` in setup.py), setup keeps the cards
together and the experts no card holds in RAM. The same copy, sized by a RAM budget (`--resident-budget-gib`, the
Unsloth UD-Q4_K_XL and UD-IQ4_XS configs), is kept on a split as well ([UNSLOTH_Q4.md](UNSLOTH_Q4.md)). Swift 1.5 IQ3_XXS at
160K (q4_0 KV, `--prefill 4096`, `--spec 4` with the stock draft layer), RTX 4060 Ti (layers 0-19) + RTX 5080 (20-47),
i9-14900KF, 32 GB of RAM, Windows 11, four greedy prompts at a time, decode tok/s:

| | first four prompts | after three more rounds |
|---|---|---|
| 5080 alone, `--resident-experts` | 25 | 29 |
| split, `--mmap-experts` (what `--resident-experts` became on a split) | 32 | 64 |
| split, `--resident-experts` (22 GiB of experts locked in RAM) | 71 | 69 |

The split with `--mmap-experts` catches up once the OS file cache holds the experts, on a PC with nothing else
running; the resident copy is there from the first request and stays locked when other programs need the RAM. With
`--pcie-frac 0 --adapt-every 0` the split's greedy output is the same with either mode.

**The prompt path's loan on a split and its streamed ring (`STRATA_SPLIT_RING`).** Under a split the prompt path borrows
the tail slots of each card's expert cache for its buffers (they are given back after the prompt), and the resident RAM
copy keeps those experts too, so a borrowed expert is not read back from the model file during a prompt. How many slots
are borrowed depends on the prompt path's streamed ring: a split uses 96 ring slots when at least 75% of the (layer,
expert) pairs sit in some card's cache, and `STRATA_SPLIT_RING=N` sets N slots (`0` goes back to the pinned-share rule).
The ring used to be chosen after the RAM copy's regions were sized, so the regions were sized for a different ring than
the prompt path then used: with `STRATA_SPLIT_RING=384` it borrowed 5,396 slots and the copy kept 4,662, and the
other 734 were read from the model file in every chunk. The ring is now chosen first (`Prefill::set_ring_override`
says it must be set before the buffers are counted). The start-up line `strata serve: lend sizing: the prompt path
borrows N slots (ring R slots at chunk C), K of them keep their experts in RAM too` shows both numbers, and a
`WARNING` follows when K is below N. Without a split, or without `--resident-experts`, nothing changes. With a split
the order of the two steps changed for every ring choice, the 96-slot rule included: its regions used to be sized for
the larger default ring, so there the copy probably kept more slots than were borrowed (not measured). The 5,396 /
4,662 figures were measured on the test machine (2x RTX 3080 20 GB at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL,
`--layer-split 23`, resident RAM mode, `--prefill auto:16384`, 2 slots), one restart per arm, with `STRATA_SPLIT_RING=384`
and `STRATA_PREFILL_RING` unset. Before the change a 104K-token prompt made 413 blob reads and read 10.3 GB of experts
from the model file per request; after it, 0, with all 5,396 borrowed slots kept in RAM, for 2.15 GiB more pinned RAM
(13.61 -> 15.76 GiB). 25K and 51K prompts read nothing from the file in either case. With the override unset and
`STRATA_PREFILL_RING=384` the lend lines are identical before and after (5,396 / 5,396, 15.76 GiB). No speed gain is
shown: the medians of three 104K reads were 2,803 tok/s before and 2,907 after, but the spread before is 2,540-2,897
(its first read faulted 4.2 GB in from NVMe) and there is no A/A restart. Here the page cache absorbed the 10 GB per
request; with less spare page cache it would be disk traffic. Not measured: both ring variables unset.

**A separate VRAM reserve for the later cards:** `--vram-reserve-later-mib N` (default: `--vram-reserve-mib`'s value).
The card that drives the monitors needs more headroom than one that drives none; with the display on the last card,
`--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card's cache that VRAM.

**auto** tries every placement (all of them for two, three or, since 0.1.40, four cards; proportional to the free VRAM beyond that) and
keeps the one whose caches would hold the most of the expert profile, hottest pairs weighted most; ties go to the
placement that leaves the fullest card the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

## What each card holds

- **every card**: a copy of the dense weights (~3.4 GB for the Coder), its own session state (the KV cache of the full
  context), its verify window and its prompt-path buffers, and an expert cache for its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. (0.1.44, AMD gfx12: the auto chunk on a split stays at 8192, not the single-card 16384: a 16K prompt is then two chunks
that overlap instead of one, a 32K one four instead of two. Same tokens; 2x / 3x / 4x R9700 prompt tok/s at 16K +36% / +45% / +53%,
32K +18% / +21% / +29%, 64K +4% / +12% / +17%, 8K unchanged. `STRATA_SPLIT_CHUNK16K=1` keeps 16384.) Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Overlapped hand-off (`STRATA_SPLIT_OVERLAP=1`, opt-in)

By default each verify window runs the stages one after another: the host syncs stage 1, then stages and launches
stage 2's graph, and so on. `STRATA_SPLIT_OVERLAP=1` (stages on different cards only) gives the hand-off a ready
flag: the earlier stage's GPU writes the hand-off with volatile stores and raises the flag (`handoff_publish`), the
next stage's graph is staged and launched on a helper thread while the earlier stage is still running and waits for
the flag on its GPU (`wait_flag_ge`), and the host goes straight from serving one stage's layers to the next, syncing
the earlier stage only once the chain is done. It takes one stream sync and one graph launch per window off the
critical path. Batch windows (several conversations) keep the serial order. It is off beside `--pipeline-windows`, whose loop overlaps the stages its own way.

Measured on 2x RX 7900 XTX (gfx1100, ROCm, Linux), Ryzen 9 9900X, 52 GB RAM, Swift 1.5 IQ3_XXS, 128K context, 8-bit
KV, layers 0-25 / 26-47; 512-token greedy answers, median of 8 runs:

| | serial (default) | `STRATA_SPLIT_OVERLAP=1` | one card |
|---|---|---|---|
| decode, code prompt | 53 tok/s | 70.1 tok/s | ~70 tok/s |
| decode, prose prompt | 45 tok/s | 55.9 tok/s | ~68 tok/s |

Greedy output is byte-identical with the flag at 0 and 1; the quiz set scored the same (47/51). Not measured on
NVIDIA yet, where a sync and a graph launch cost less.

The reading stage's wait for the flag is bounded on the GPU: `STRATA_SPLIT_WAIT_MS` (default 30000, 0 = no bound).
If the flag does not arrive in time, the wait marks an error word and returns, the window's graph completes, and the
host fails that window with an error naming the setting (the server then restarts the engine) instead of using a
stale hand-off. Normally the host's own per-layer wait (20 s, #267) ends a stalled window first and releases every
GPU wait, the hand-off's included. Test hook: `STRATA_TEST_HANDOFF_DROP=N` withholds the flag in the first stage's
N-th window, as if the publish were lost.
## Prompt chunks sized for the pipeline (`--prefill-pipe`, opt-in; setup's `--calibrate` sets it)

`--prefill auto` takes the largest chunk the buffers allow (up to 8192), which is what one card wants: a chunk costs a
fixed pass over its layers' experts (b: dequantized, or streamed over PCIe where they are not in VRAM) plus a part per
token (a), so few large chunks are cheapest. A layer split reads a prompt as a pipeline - stage s reads chunk c while
stage s+1 reads chunk c-1 - and then takes about sum_i (b + a C_i) + (S - 1) max_i (b + a C_i) for chunks C_i over S
stages: large or uneven chunks keep the later cards waiting (a 10.7K-token prompt in a 7680 and a 3011 chunk over
four stages: the first stage carried 24 of the 36 s), small ones pay b again and again.

`--prefill-pipe R` reads each prompt in the chunk count with the least pipeline time for b/a = R tokens, when that
beats today's split by 3% or more: every count from the fewest the buffers allow is tried, each chunk evened out on the
256-token grid (at least 512, at most the buffers' chunk; keep `--prefill auto` so long prompts may use large chunks).
b/a differs by an order of magnitude between rigs - about 700 tokens on four GP100 dies (every expert in VRAM, slow
arithmetic), about 5000 on two RTX 3090 Ti (part of the experts streamed per chunk, fast arithmetic) - and a wrong one
costs: a fixed 1024 read 19-24% slower on the two RTX 3090 Ti and 22-36% slower on two RTX 4090 (reported in the PR by
MistyMoonR). So it is neither guessed nor learned from the prompts the engine serves (that made a prompt's chunks, and
so its output bits, depend on how busy the PC happened to be): like the PCIe share and the draft floor, it is a
setting of the PC that setup's `--calibrate` measures once (`tools/calibrate.py`: on a layer split, the prompt read
speed of a 2K- and an 8K-token prompt - the same text every time, after a header of its own so nothing is reused -
with the chunks off, evened out and b/a 768, 1536, 3072 and 6144; the best is kept only when it reads faster than off
in each of three paired rounds and by more than 3% in the median) and writes into the config. Measured so: four GP100
dies keep b/a 768 (355-356 against 219-220 tok/s in every round, +62%); two RTX 3090 Ti keep it off (nothing read
faster than off, b/a 768 20% slower). A first version that read a different stretch of text per value and confirmed in
two rounds picked 1536 on the RTX 3090 Ti pair from one fast read - where the experts are streamed, the read speed
depends on the text. `--prefill-pipe 1` needs no
b/a: it is `STRATA_PREFILL_EQUAL=1`'s rule (#693) - today's chunk count, evened out on the 256-token grid, and only
when today's last chunk has 1024 tokens or more (3072 with `STRATA_PREFILL_CPU_SHARE` set). A chunk that size streams
every expert its card does not hold, a shorter one only the experts it routes to, so a prompt just past the chunk size
reads as one full chunk and a nearly free tail, and evening that out would add a second full pass over the experts:
the first version evened every split out, and on two RTX 4090 MistyMoonR's calibration read 1,601 tok/s with it
against 2,129 off. 0 (the default) is today's split;
`STRATA_PREFILL_PIPE=<R>` is the same as the flag where the config cannot carry it. The chunk geometry changes the
rounding, which is why it is opt-in.

A prompt segment (the prompt between two of its cuts: the system-prompt root, the last turn, a `pin=`) is planned once,
on its whole length and R, and every run of it reads in that chunk. So its chunks, and the loan its buffers are laid
out for, are the same whether it is read in one run, in pieces between the windows of decoding `--batch` slots, or goes
on after a `BYIELD` or from a periodic checkpoint taken inside it (the slot and the checkpoint keep the plan; the read
goes on in it when it continues the same segment, else it plans again and says so in the log). The output bits then
match as far as the experts the cards hold do (`--adapt-swaps 0` for byte-identical repeats, as without the pipe).
Before this, a read beside decoding slots re-planned every piece, and one that went on after a `BYIELD` re-planned what
was left. A chunk the pipe planned takes no `STRATA_PREFILL_CPU_SHARE`: the stages read
such chunks at the same time, and which stage got the CPU pool first would decide which experts the CPU computed. The
DONE line ends with `chunk=` (the first chunk of the request's longest batched segment) and `experts_streamed=` (the
expert blobs its prompt copied to the cards), and the INFO line carries `prefill_cap`, `prefill_stages`,
`stream_all_min` and `prefill_pipe`. One line in the log says what a plan chose:

    strata prefill: 10058 tokens in 1280-token chunks over 4 stages (--prefill-pipe 706: b/a)

Four GP100 dies (PH402 SKU 200, `--layer-split 12,24,36`, Flash-Next IQ3_XXS, 64K context), prompt tok/s, fresh
prompts, two interleaved rounds (each within 0.2%), with b/a 1024 (the same chunks as 700-890, measured on this rig):

| Prompt tokens | `--prefill auto` | `--prefill 2048` | `--prefill auto` + `--prefill-pipe 1024` (chunk) |
| ---: | ---: | ---: | ---: |
| 2,013 | 180.9 | 181.4 | 228.0 (1024) |
| 3,865 | 207.6 | 276.5 | 312.4 (1024) |
| 10,041 | 261.1 | 433.0 | 432.6 (2048) |
| 28,998 | 475.7 | 554.1 | 559.7 (3328) |

Two RTX 3090 Ti (CUDA 12.9 build for sm_86, two stages, Swift 1.5 IQ3_XXS, 1M context, 13,371 of 24,576 experts in
VRAM), prompt tok/s, fresh prompts, two starts per arm, each reading the four prompts twice (the cold first pass after
a card was freed is left out); b/a 5836 and 6432 (timed on the rig), against 1024:

| Prompt tokens | `--prefill auto` | + b/a 5836 / 6432 (chunk) | for comparison: b/a 1024 |
| ---: | ---: | ---: | ---: |
| 2,035 | 694-815 | 698-816 (today's) | 560-562 |
| 3,887 | 1,127-1,207 | 1,115-1,227 (today's) | 880-885 |
| 10,063 | 1,437-1,553 | 1,654-1,681 (2 x 5120) | 1,109-1,113 |
| 29,019 | 2,532-2,603 | 2,566-2,620 (today's) | 2,064-2,089 |

## Limits (for now)

- **Works across cards** (bench/results/2026-09-29-layer-split-limits):
  - images (`--vision`): each card keeps its own image-position table;
  - control vectors and the experimental speed projection: each card holds the vector's tables, switched on and
    off per request on all of them;
  - KV streaming (`--kv-resident`): each card streams the KV of its own session;
  - mid-prompt checkpoints (`--prompt-cache-every`): each card saves its part of a checkpoint when it has read that
    chunk;
  - the older helper-GPU caches (`--expert-cache-remote`, docs/SECOND_GPU.md): they take the visible GPUs no stage
    runs on, and hold only experts no stage's cache holds. On the test rig, a 2080 Ti helper made decoding slower,
    as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`), with or without a split; a native (IQ) pack says so at
  start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024`
  halves that) instead of borrowing cache slots as one card does. An explicit `--expert-cache` on the first card is
  capped to leave room for them.
- Under WDDM (Windows, and WSL2) only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts,
  leaves WDDM refusing allocations); the rest streams through the pinned staging ring. A Linux driver has no such
  limit, so there the whole arena is pinned (since 0.1.31; the cap cost a 4090 + 3060 split two thirds of its
  prompt speed, #253). `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.
- Every card needs compute capability 7.5 (RTX 20 or newer). The pre-sm_80 QSA scorer path is fp32 FMAs, so a
  Turing card runs the same kernels instead of the tensor-core prompt attention.

## Measured

The Coder on an RTX 5080 + RTX 3090 (Ryzen 9 9950X3D), 32K context; details in
`bench/results/2026-09-29-layer-split/`:

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
|---|---|---|
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

- **Prompts gain the most** (+18-20%): each card reads its own layers of the chunk while the other reads the next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **Correctness:** one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.

## One conversation with both cards busy (`--pipeline-windows`, default on two or more AMD gfx12 stages)

**Default on AMD RDNA4 (gfx1200 / gfx1201).** With `--serve`, the draft layer and a layer split of **two or more stages**,
`--pipeline-windows 2` is on unless you say otherwise (the log says "is on by default for a layer split of N stages"; two
stages since 0.1.44, three or more since 0.1.43). It is no quality trade: the tokens are the serial loop's up to CPU rounding
(below). `--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off; `--pipeline-windows 2` asks for it on any split.
The same options that switch it off when you ask for it (`--batch`, `--peer-device`, the helper caches, no draft layer,
`--adapt-async 1`) leave it off without a line when it is only the default. NVIDIA builds keep it opt-in.

**Where the draft layer sits (0.1.44).** The draft layer's chain used to run on the last card, beside the last stage's window.
On 2x / 4x R9700 the two streams then slowed each other, in most engine starts and for the whole life of that engine: the
last stage's window took 27 ms instead of 13 ms while a chain ran (14.9 ms in the lucky starts, which also decoded 25% faster;
the 0.1.43 notes' "128.9 does not reproduce" was this). With the pipeline on, the engine now loads the draft layer on the
**first** card, whose window has finished when the chain starts: the chain's source rows are copied across once per chain and the
draft head's rows once at load. Same tokens; `STRATA_MTP_STAGE0=0` keeps the old place. Exact-flags A/B, 5 pairs, decode tok/s
(serial loop -> now): 2 cards code 103 -> 125 (+21%), 4K prompt +8%, prose equal; 3 cards code +19%, 4K +12%, prose +2%; 4 cards
code +22%, prose +4%, 4K +15%.

With a split the cards take turns on a verify window: the first card runs its layers and hands off, then waits while
the last card runs the rest, the head and the draft. `--pipeline-windows 2` lets the first card start the **next**
window while the last card still verifies this one. The next window is a guess: that this window is accepted whole
and that its bonus token is the one the draft layer predicts (the draft layer is run on through the drafts of the
window in flight). When the guess holds, half of the next window is already done; when it does not, the first card
puts its state back (a copy of its recurrent state taken while the window ran) and the next window is built from the
real tokens. A guessed window is only started when the draft layer's estimate says it is likely to be kept. Windows
copied from earlier context (`--suffix-draft`) are guessed past as well, which is where edits that copy text gain
most. `--pipeline-windows 1` overlaps only the short prompt reads that go through the verify windows
(`--short-read`).

Two or more cards, `--serve`. With three or more stages every stage but the last is a front stage: the guessed
window follows the verified one through them one card behind (each front stage keeps its own pair of recurrent-state
copies and puts its state back on a wrong guess), and only the last stage waits for the verdict. `--pipeline-windows 1`
and the asynchronous tier stay two-stage. In the config:

```
"args": [ ..., "--pipeline-windows", "2" ],
"layer_split": "20"
```

- **Cost**: a second verify window on each card, 160 MiB more kept out of each card's expert cache, plus two copies
  of the first card's recurrent state (about 3 MiB per GDN layer it runs) on the first card with `2` (on every front
  card with three or more stages; only the first card's room is kept out of its cache, the others allocate theirs after).
- **Same text**: the last card only ever runs windows that are verified, and every window row computes what it
  would in any other window, so the tokens are the serial loop's. With `STRATA_IQ_MT_MIN=1 --pcie-frac 0
  --adapt-every 0` the greedy output is identical bit for bit to the serial loop's with the same expert caches. The pipeline keeps
  its VRAM out of the caches, so against a run without the flag a few experts move from a card to the CPU, which
  rounds them differently, and a near-tie can flip (a serial run given the same caches through `--vram-reserve-mib`
  matches it exactly).
- **Off, with one line in the log saying why**, with `--batch` slots, `--peer-device`, the helper caches
  (`--expert-cache-device1..3`, which `--remote-expert-opt` builds on), a split onto one GPU (`--split-device 0`), or no
  draft layer; on three or more stages also `--pipeline-windows 1` and `--adapt-async 1`. A request with repetition penalties (`penalty_last_n`) or coupled
  draft sampling decodes serially.
- **With the resident RAM mode's asynchronous swaps** (`--adapt-async 1`, [DETAILS.md](DETAILS.md)) a round's steps
  advance between the verified windows. Each card's copies are queued by the decode loop itself while that card has
  no window in flight, after every window that may still read what they overwrite has finished; the moves into RAM
  wait the same way.
- **Measured** (Swift 1.5 IQ3_XXS, 160K context, q4_0 KV, the stock draft layer, RTX 4060 Ti (layers 0-19) +
  RTX 5080 (20-47), i9-14900KF, 32 GB of RAM with the resident RAM mode on the split (#848); greedy, 500 tokens, two
  interleaved pairs of three rounds, decode tok/s): Python code 90.4 -> 103.1, C code 76.6 -> 81.6, English prose
  63.0 -> 71.1, Italian prose 41.6 -> 46.3, a copy-heavy edit (a 5 KB file back with a rename) 90.5 -> 117.9; mean
  72.4 -> 84.0 (+16%). It pays when the windows are GPU-bound: with the experts read through the OS file cache
  (`--mmap-experts` on that 32 GB PC) the file reads dominate and it measured no faster.
- **Measured on four stages** (Flash-Next GSQ-RCO IQ3_S, 262K context, int8 KV, layer split 12,24,36 on 4x RX 7900 XT,
  every expert in VRAM; greedy, reasoning off, 256 tokens, repository-text prompts): decode 59.4 -> 67.6 tok/s at 4K
  (median of 3) and 51.8 -> 65.6 at 32K. A guessed window that holds costs ~15 ms against ~43 ms for a fresh one; about
  a third of the guesses held. The greedy text was identical to the serial loop's on all four prompts (`STRATA_IQ_MT_MIN=1
  --pcie-frac 0 --adapt-every 0`), and with every guess forced wrong (`STRATA_PIPELINE_FORCE_MISS=1`).
- **HIP**: the draft layer's per-row `gr_read` used the last stage's session scratch (`ss.block.gr`), which that stage's
  verifier also uses; a chain launched at a verdict and the next window's last stage then ran on the card at once and
  the window's head read a clobbered mix (wrong tokens after a guess that held). The draft layer now has its own scratch.

The `STRATA_PIPELINE_*` tuning and test variables (THETA, FORCE_MISS, SWITCH, LOG, TRACE, SPEC_DEPTH and the like) are read only with
`STRATA_PIPELINE_DEBUG=1`. `--pipeline-windows` and `--adapt-async 1` combine: the engine turns the asynchronous tier
off beside `--pipeline-windows 2` only when `STRATA_PIPELINE_ADAPT_ASYNC=0` is set. What switches either one off is
printed once at start ("is off: ..."). `--remote-expert-opt` does something only with a helper cache
(`--expert-cache-device1..3`); on a plain layer split it is inert, and setup no longer writes it there (#1447).

Measured on 2x RTX 3090 (sm_86, 250 W limits; GPU0 PCIe 4.0 x16, GPU1 x4, no NVLink; Ryzen 7 9800X3D, 32 GB RAM),
Qwen3.8-Flash-Next GSQ-RCO IQ3_S, `--resident-experts`, KV int8, `--spec 4 --mtp`, `"layer_split": "29"` for the
`--pipeline-windows` rows (without `--remote-expert-opt`); the others are setup's config (`auto`). Decode is the mean of
runs 2-6 of 1,500-token coding replies at temperature 0.6, prefill one cold 19.9K-token prompt; one run per arm unless
noted (reported by adambenhassen, #1447; not repeated on our boxes):

| Arm | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| setup defaults (two runs) | 137.0 / 136.9 | 1798 |
| `--adapt-async 1` (two runs) | 145.2 / 143.7 | 1800 / 1797 |
| `--adapt-async 1`, no `--remote-expert-opt` | 145.0 | 1807 |
| `STRATA_ADAPT_LAG=2` | 142.8 | 1800 |
| `STRATA_EXCHANGE_ROTATE=1` | 139.6 | 1800 |
| `STRATA_PF_FUSED=1` | 137.2 | 1913 |
| `--pipeline-windows 2` | 136.7 | 2097 |
| `STRATA_SPEC_COUPLED=1` | 137.6 | 1804 |
| `STRATA_SPEC_PROB=1` | 137.2 | 1788 |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1` | 136.5 | 1805 |
| async + lag 2 + rotate + pf_fused | 145.0 | 1829 |
| pw2 + lag 2 + rotate + pf_fused | 135.9 | 2296 |
| pw2 + async + lag 2 + rotate + pf_fused | 142.0 | 2302 |

The last row keeps most of the asynchronous tier's decode gain and the pipeline's +28% prefill; no stalls in any arm.
These are opt-in settings on one rig, not defaults.

## Several conversations at once

With a layer split, `--batch N --batch-groups G --trim-stage-weights` decodes several conversations together and
pipelines them through the cards: see [BATCHING.md](BATCHING.md).
