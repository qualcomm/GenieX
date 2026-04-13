import os

from huggingface_hub import HfApi

# Enable hf_transfer (if installed) for faster transfers :contentReference[oaicite:0]{index=0}
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_TOKEN"] = "REDACTED_HF_TOKEN"

api = HfApi()
api.upload_large_folder(
    repo_id="nexaai/bridge-test-modelfiles",
    folder_path="./modelfiles",
    repo_type="model",
    ignore_patterns=[
        "**/.cache/**",
        "**/assets/**",
        ".gitignore",
        "**/DS_Store",
    ],
)
