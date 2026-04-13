import argparse
import os
import platform

from huggingface_hub import snapshot_download

# Enable hf_transfer (if installed) for faster transfers
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_TOKEN"] = "REDACTED_HF_TOKEN"

# Get the project root directory (one level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Centralized HuggingFace repository
HF_REPO = "nexaai/bridge-test-modelfiles"

# Platform-specific folder mappings
PLATFORM_FOLDERS = {
    "windows_arm64": ["llama_cpp", "qnn"],
    "linux_arm64": ["llama_cpp", "qnn"],
}

# Download mode mappings: defines which 2nd level folders to download for each plugin
# "core" mode: minimal set of essential models
# "broad" mode: expanded set including specialized models
# "full" mode: download everything (no filtering)
MODE_MAPPINGS = {
    "qnn": {
        "core": [
            "OmniNeural-4B",
            "Llama3.2-3B-NPU-Turbo",
        ],
        "broad": [
            "OmniNeural-4B",
            "Llama3.2-3B-NPU-Turbo",
            "EmbedNeural",
            "jina-rerank-npu",
            "parakeet-tdt-0.6b-v3-npu",
            "paddleocr-npu",
            "Pyannote-NPU",
            "AutoNeural",
        ],
    },
    "llama_cpp": {
        "core": [
            "Qwen3-0.6B-Q8_0.gguf",
            "SmolVLM-256M-Instruct-Q8_0.gguf",
        ],
    },
    # Add more mappings as needed
}


def detect_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if machine in ["aarch64", "arm64"]:
            return "windows_arm64"
        raise ValueError(f"Unsupported Windows architecture: {machine}")
    elif system == "linux":
        if machine in ["aarch64", "arm64"]:
            return "linux_arm64"
        raise ValueError(f"Unsupported Linux architecture: {machine}")
    raise ValueError(f"Unsupported platform: {system} {machine}")


def get_allow_patterns(folders: list[str], mode: str = "full") -> list[str]:
    """
    Generate allow patterns based on folders and download mode.

    Args:
        folders: List of top-level folder names (plugins)
        mode: Download mode - "core", "broad", or "full"

    Returns:
        List of patterns for snapshot_download allow_patterns
    """
    if mode == "full":
        # Download everything for specified folders
        return [f"{folder}/*" for folder in folders]

    # For core/broad modes, filter by 2nd level folder names
    patterns = []
    for folder in folders:
        if folder in MODE_MAPPINGS and mode in MODE_MAPPINGS[folder]:
            # Get the allowed 2nd level folders for this plugin and mode
            second_level_folders = MODE_MAPPINGS[folder][mode]
            for second_level in second_level_folders:
                patterns.append(f"{folder}/{second_level}/*")
        else:
            # If no mapping exists for this folder/mode, download everything for it
            patterns.append(f"{folder}/*")

    return patterns


def download_modelfiles(
    folders: list[str], mode: str = "full", source: str = "platform"
):
    """
    Download specified folders from the centralized HuggingFace repo.

    Args:
        folders: List of folder names to download
        mode: Download mode - "core", "broad", or "full"
        source: Description of how folders were determined (for logging)
    """
    if not folders:
        raise ValueError("No folders specified for download")

    target_dir = os.path.join(PROJECT_ROOT, "modelfiles")
    os.makedirs(target_dir, exist_ok=True)

    # Create allow patterns based on mode
    allow_patterns = get_allow_patterns(folders, mode)

    print(f"Download mode: {mode}")
    print(f"Source: {source}")
    print(f"Downloading folders: {', '.join(folders)}")
    print(f"Target directory: {target_dir}")
    print(f"Repository: {HF_REPO}")
    if mode != "full":
        print(f"Filtered patterns: {len(allow_patterns)} pattern(s)")

    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=target_dir,
            allow_patterns=allow_patterns,
        )
        print(f"Successfully downloaded modelfiles in {mode} mode")
    except Exception as e:
        print(f"Error downloading modelfiles: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download modelfiles from HuggingFace repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download based on detected platform (full mode by default)
  python download_models.py
  
  # Download with core mode (minimal models)
  python download_models.py --mode core
  
  # Download with broad mode (expanded model set)
  python download_models.py --mode broad
  
  # Download specific folders
    python download_models.py --folders llama_cpp qnn
    python download_models.py -f qnn --mode core
        """,
    )
    parser.add_argument(
        "--folders",
        "-f",
        nargs="+",
        help="Specific folder(s) to download. If not specified, folders are determined by platform.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["core", "broad", "full"],
        default="full",
        help='Download mode: "core" (minimal), "broad" (expanded), or "full" (everything). Default: full',
    )

    args = parser.parse_args()

    if args.folders:
        # Use specified folders
        download_modelfiles(
            args.folders, mode=args.mode, source="command-line arguments"
        )
    else:
        # Use platform detection
        platform_key = detect_platform()
        folders = PLATFORM_FOLDERS.get(platform_key)
        if not folders:
            raise ValueError(f"Unknown platform key: {platform_key}")
        download_modelfiles(
            folders, mode=args.mode, source=f"platform detection ({platform_key})"
        )


if __name__ == "__main__":
    main()
