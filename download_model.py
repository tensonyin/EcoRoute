import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed. Run 'pip install huggingface_hub'")
    sys.exit(1)

print("Starting model download (unsloth/gemma-4-E2B-it-GGUF)...")
os.makedirs("model", exist_ok=True)
model_path = hf_hub_download(
    repo_id="unsloth/gemma-4-E2B-it-GGUF",
    filename="gemma-4-E2B-it-Q4_K_M.gguf",
    local_dir="model",
    local_dir_use_symlinks=False
)
print(f"Model successfully saved to: {model_path}")
