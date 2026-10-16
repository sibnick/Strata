import os
import subprocess
import time
from pathlib import Path


def available():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:"))) * 1024


if available() < int(5.5 * 2**30):
    raise SystemExit("Insufficient host headroom to start safely.")
env = os.environ.copy()
env.pop("CUDA_VISIBLE_DEVICES", None)
env.update(STRATA_UNBUFFERED_LOAD="1", STRATA_RESIDENT_HEADROOM_GIB="4",
           STRATA_PIN_RESERVE_GIB="4", STRATA_TRACE="1", STRATA_SPLIT_RING="96", STRATA_STAGE_PIN="0")
args = [
    "build/strata", "--pack", "build/pack/ud-q4-k-xl",
    "--native", "/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--gpu", "1,0", "--layer-split", "37", "--trim-stage-weights",
    "--max-context", "262144", "--kv", "int8",
    "--mtp", "/home/nikolay/projects/Strata-data/mtp/rt", "--spec", "2", "--prefill", "256",
    "--no-prefill-borrow", "--resident-budget-gib", "0.001",
    "--expert-profile", "data/expert-profile.bin", "--ple-io", "direct",
    "--ple-row-cache", "1024", "--vram-reserve-mib", "1024", "--vram-reserve-later-mib", "1024",
    "--pool-workers", "1", "--serve", "--prompt-cache", "0",
    "--conversation-cache-min-free-mib", "4096",
]
print("COMMAND", " ".join(args), flush=True)
start = time.monotonic()
minimum = available()
reason = "completed"
last_progress = start
last_io = -1
with open("/tmp/strata-k37-guarded.log", "w") as log:
    process = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=log,
                               stderr=subprocess.STDOUT, text=True)
    process.stdin.write("GEN 4 1,2,3\nQUIT\n")
    process.stdin.flush()
    next_sample = 0
    while process.poll() is None:
        now = time.monotonic() - start
        free = available()
        minimum = min(free, minimum)
        io_path = Path(f"/proc/{process.pid}/io")
        try:
            io = dict(line.split(": ", 1) for line in io_path.read_text().splitlines())
            read_bytes = int(io["read_bytes"])
            if read_bytes != last_io:
                last_io = read_bytes
                last_progress = time.monotonic()
        except (FileNotFoundError, PermissionError):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
                raise
            read_bytes = last_io
        stalled = time.monotonic() - last_progress > 120
        if free < 5 * 2**30 or now > 900 or stalled:
            reason = ("5 GiB reaction threshold (4 GiB floor)" if free < 5 * 2**30
                      else "120 seconds without disk-read progress" if stalled else "900 second time limit")
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        if now >= next_sample:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            print(f"{now:.1f}s available={free} disk_read_bytes={read_bytes} GPU free/used MiB: "
                  f"{gpu.stdout.strip().replace(chr(10), '; ')}", flush=True)
            next_sample = now + 10
        time.sleep(.1)
    print("RESULT", process.returncode, reason, "minimum_host_available_bytes", minimum,
          "elapsed_seconds", time.monotonic() - start, flush=True)
print("\n".join(Path("/tmp/strata-k37-guarded.log").read_text().splitlines()[-35:]))
