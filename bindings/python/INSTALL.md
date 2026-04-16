# Python Binding Development Guide

This guide explains how to configure and build the Python bindings for GenieX-Bridge from source.

## Prerequisites

### System Requirements

- **Python**: Python 3
- **CMake**: Version 3.16 or later
- **Build Tools**:

  - `build` package for Python (install via `pip install build`)

  ```bash
  pip install build
  ```

  - Platform-specific build tools (see main [INSTALL.md](../../INSTALL.md))

## Development Workflow

When developing locally, you need to configure the environment to use locally built binaries instead of downloading pre-built ones. There are two development approaches:

### Prerequisites for Local Development

Both development modes require the following environment variables to be set:

**Linux (ARM64):**

```bash
# Skip automatic binary download
export GENIEX_SKIP_DOWNLOAD=1

# Point to locally built plugins (from project root directory)
export GENIEX_PLUGIN_PATH=$(pwd)/build/out
```

**Windows PowerShell:**

```powershell
# Skip automatic binary download
$env:GENIEX_SKIP_DOWNLOAD = "1"

# Point to locally built plugins (from project root directory)
$env:GENIEX_PLUGIN_PATH = "$PWD\build\out"
```

### Method 1: Development Mode (Editable Install)

This method installs the package in editable mode, allowing you to modify the Python code and see changes immediately without reinstalling.

1. **Build the project** (from project root):

   ```bash
   cmake -B build -DGENIEX_BINDING_PYTHON=ON -DGENIEX_BRIDGE_VERSION=v0.0.0
   cmake --build build -j
   ```

2. **Set environment variables** (as shown above)

3. **Navigate to Python binding directory:**

   ```bash
   cd bindings/python
   ```

4. **Install in editable mode:**

   ```bash
   pip install -e .
   ```


   ```bash
   ```

### Method 2: Local Testing (Using Built Package)

This method uses the built source distribution package for testing.

1. **Build the project** (from project root):

   ```bash
   cmake -B build -DGENIEX_BINDING_PYTHON=ON -DGENIEX_BRIDGE_VERSION=v0.0.0
   cmake --build build -j
   ```

2. **Set environment variables** (as shown above)

3. **Install from built package:**

   ```bash
   pip install build/dist/geniex-<version>.tar.gz
   ```


   ```bash
   ```

**Note:** After building, the Python source distribution (sdist) will be located at `build/dist/geniex-<version>.tar.gz` in the project root directory.

## Examples and Testing

### User Examples Repository

TODO: add the current examples repository link for the GenieX-Bridge Python binding.

These examples can be used for testing and updating during development. They demonstrate how to use various features of the Python bindings, including LLM, VLM, embeddings, and other model interfaces.

### Local Tests

Local test suites are available in the `tests/` directory. These pytest-based tests can be used to verify functionality during development:


**Run specific test file:**

```bash
# Test LLM functionality
pytest tests/llm.py -v

# Test other components
pytest tests/vlm.py -v
pytest tests/embedding.py -v
```

For detailed information about running tests, see [tests/README.md](tests/README.md).

## Python SDK Design

### Architecture Overview

The Python SDK follows a **two-layer architecture** that separates low-level C bindings from high-level Pythonic APIs:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                       │
│  (User Code: LLM, VLM, Embedding, etc.)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              High-Level Python API Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   LLM    │  │   VLM    │  │ Embedding│  │   ASR    │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         ModelLoaderMixin (Abstract Base)             │   │
│  │  • Model download from HuggingFace                   │   │
│  │  • Local path handling                               │   │
│  │  • Quantization selection                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Store   │  │ ModelHub │  │ Progress │                   │
│  │ (Cache)  │  │(Download)│  │ Tracker  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│           Low-Level C Bindings Layer (ctypes)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              geniex/geniex_sdk/                        │   │
│  │  • _lib.py      - Library loader                     │   │
│  │  • ml.py        - Core C functions                   │   │
│  │  • llm.py       - LLM C structures                   │   │
│  │  • vlm.py       - VLM C structures                   │   │
│  │  • error.py     - Error handling                     │   │
│  │  • types.py     - C type definitions                 │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    C ABI Interface (ml.h)                   │
│              (Unified C Library - geniex_bridge)              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **High-Level API Layer** (`geniex/`)

**Model Classes:**
- `LLM` - Large Language Model interface
- `VLM` - Vision-Language Model interface
- `Embedder` - Text embedding interface
- `ASR` - Automatic Speech Recognition
- `TTS` - Text-to-Speech
- `CV` - Computer Vision (OCR)
- `ImageGen` - Image Generation
- `Diarize` - Speaker Diarization
- `Reranker` - Document Reranking

**Core Infrastructure:**
- `core.py` - Logging, device/plugin discovery
- `models.py` - Model download and management API
- `internal/model_loader.py` - `ModelLoaderMixin` abstract base class
- `internal/store.py` - Local model cache management
- `internal/model_hub.py` - HuggingFace model download
- `internal/progress_tracker.py` - Download progress tracking

#### 2. **Low-Level Bindings Layer** (`geniex/geniex_sdk/`)

**Library Loading:**
- `_lib.py` - Dynamic library loader (finds `geniex_bridge.dll/so/dylib`)

**C Interface Bindings:**
- `ml.py` - Core C functions (`ml_init`, `ml_deinit`, etc.)
- `llm.py` - LLM C structures and functions
- `vlm.py` - VLM C structures and functions
- `error.py` - Error code definitions and exception mapping
- `types.py` - C structure definitions (ctypes)

#### 3. **Model Loading Flow**

```
User Code:
    LLM.from_("microsoft/Phi-3-mini-4k-instruct", quant="Q4_K_M")
         │
         ▼
ModelLoaderMixin.from_()
         │
         ├─▶ is_local_path()? 
         │   ├─ YES → _from_local_path()
         │   └─ NO  → _from_huggingface_repo()
         │
         ▼
_from_huggingface_repo():
    ├─▶ Store.get() - Get cache directory
    ├─▶ ensure_model_downloaded() - Download from HuggingFace
    ├─▶ _select_model_file() - Choose quantized file
    ├─▶ _extract_model_params_from_manifest() - Parse manifest
    └─▶ _create_instance() - Create LLM instance
         │
         ▼
LLM.__init__()
   └─▶ Use C interface via ctypes
```

#### 4. **Build System Integration**

**CMake Integration:**
```cmake
# bindings/python/CMakeLists.txt
- Finds Python3 interpreter
- Configures version file (_version.py)
- Builds Python sdist package
- Output: build/dist/geniex-<version>.tar.gz
```

**Python Build Tools:**
- Uses `build` package for creating source distributions
- `setup.py` - Legacy build configuration
- `pyproject.toml` - Modern Python packaging (PEP 517/518)

### Design Principles

1. **Separation of Concerns**
   - Low-level bindings isolated in `geniex_sdk/`
   - High-level APIs in `geniex/`
   - Internal utilities in `internal/`

2. **Model Abstraction**
   - `ModelLoaderMixin` provides unified model loading interface
   - All model classes inherit from this mixin
   - Consistent API across all modalities

3. **Automatic Resource Management**
   - Models cached in `~/.cache/geniex/geniex_sdk/models/`
   - Automatic download from HuggingFace
   - Progress tracking for large downloads

4. **Error Handling**
   - C error codes mapped to Python exceptions
   - Type-safe error propagation
   - Clear error messages

5. **Plugin Discovery**
   - Automatic plugin detection via `get_plugin_list()`
   - Device enumeration per plugin
   - Default plugin selection based on platform

### File Structure

```
bindings/python/
├── geniex/                    # High-level Python API
│   ├── __init__.py           # Public API exports
│   ├── core.py               # Logging, device/plugin discovery
│   ├── llm.py                # LLM class
│   ├── vlm.py                # VLM class
│   ├── embedding.py          # Embedder class
│   ├── asr.py                # ASR class
│   ├── tts.py                # TTS class
│   ├── cv.py                 # CV class
│   ├── image_gen.py          # ImageGen class
│   ├── diarize.py            # Diarize class
│   ├── rerank.py             # Reranker class
│   ├── models.py             # Model download API
│   ├── geniex_sdk/             # Low-level C bindings
│   │   ├── __init__.py
│   │   ├── _lib.py           # Library loader
│   │   ├── ml.py             # Core C functions
│   │   ├── llm.py            # LLM C structures
│   │   ├── vlm.py            # VLM C structures
│   │   ├── error.py          # Error handling
│   │   └── types.py          # C type definitions
│   └── internal/            # Internal utilities
│       ├── model_loader.py   # ModelLoaderMixin
│       ├── model_hub.py      # HuggingFace integration
│       ├── store.py          # Local cache management
│       ├── progress_tracker.py
│       └── types.py          # Internal type definitions
├── CMakeLists.txt            # CMake build configuration
├── setup.py                  # Python package setup
├── pyproject.toml            # Modern Python packaging
└── tests/                    # Test suite
```

## Additional Resources

- Main project [INSTALL.md](../../INSTALL.md) for platform-specific dependencies
- [README.md](README.md) for usage examples and API documentation
- [Tests README](tests/README.md) for testing guidelines
- TODO: add the current user examples repository link for testing and development.
