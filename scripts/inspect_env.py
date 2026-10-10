"""Inspect CPU/RAM/OS/GPU and installed packages; save to results/env.json."""
import json, os, platform, shutil, subprocess, sys

info = {
    "platform": platform.platform(),
    "python": sys.version,
    "cpu_count": os.cpu_count(),
    "gpu": None,
    "ram": None,
}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["threads"] = torch.get_num_threads()
except Exception as e:
    info["torch"] = f"missing ({e})"
try:
    import transformers
    info["transformers"] = transformers.__version__
except Exception as e:
    info["transformers"] = f"missing ({e})"
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                info["ram"] = line.strip()
                break
except OSError:
    pass
if shutil.which("nvidia-smi"):
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        info["gpu"] = f"nvidia-smi failed: {e}"
else:
    info["gpu"] = "none (nvidia-smi not found -> CPU-only)"

print(json.dumps(info, indent=2))
os.makedirs("results", exist_ok=True)
with open("results/env.json", "w") as f:
    json.dump(info, f, indent=2)
