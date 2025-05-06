import importlib
import platform
import torch
import transformers
import peft
import datasets

print("\n✅ Environment Validation Report")
print("------------------------------")

print(f"🖥️  Platform      : {platform.system()} {platform.machine()}")
print(f"🐍 Python Version: {platform.python_version()}")
print(f"🔥 Torch Version : {torch.__version__}")
print(f"💾 Device        : {'MPS (Metal)' if torch.backends.mps.is_available() else 'CPU'}")

# Check module versions
modules = {
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
}

for name, version in modules.items():
    print(f"📦 {name:<12}: v{version}")

# Check if CUDA or MPS is enabled (optional)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor = torch.tensor([1.0]).to(device)
print(f"✅ Torch can run tensor on: {tensor.device}")

print("\nAll core libraries are working correctly ✅")
