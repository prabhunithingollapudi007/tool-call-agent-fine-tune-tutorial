"""
hello.py — Quick environment check. Run this first to verify your setup.

Usage:
    python hello.py
"""

import torch

print("=" * 50)
print("Environment Check")
print("=" * 50)

# GPU check
gpu_available = torch.cuda.is_available()
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {gpu_available}")
if gpu_available:
    print(f"GPU device      : {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"VRAM            : {vram_gb:.1f} GB")

# Check key dependencies
deps = {
    "transformers": "transformers",
    "datasets": "datasets",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "accelerate": "accelerate",
}

print(f"\nDependencies:")
for name, module in deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        print(f"  {name:20s} {version}")
    except ImportError:
        print(f"  {name:20s} NOT INSTALLED")

print("\nRun: pip install -r requirements.txt")
print("=" * 50)