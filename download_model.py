import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed. Run 'pip install huggingface_hub'")
    sys.exit(1)

print("Starting model download (Qwen/Qwen2.5-1.5B-Instruct-GGUF)...")
os.makedirs("model", exist_ok=True)
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir="model",
    local_dir_use_symlinks=False
)
print(f"Model successfully saved to: {model_path}")
