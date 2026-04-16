# ML Backend Tests

This directory contains tests for the ML interfaces defined in `ml.h`, organized by ML functionality rather than backend implementation.

## Structure

The tests are organized by ML interface:

- `llm.cpp` - Language Model tests
- `embedding.cpp` - Text Embedding tests
- `multimodal.cpp` - Multimodal Vision-Language Model tests
- `multimodal-multi-round.cpp` - Multi-round multimodal conversation tests
- `rerank.cpp` - Text Reranking tests
- `asr.cpp` - Automatic Speech Recognition tests
- `profiling.cpp` - Performance profiling tests across all interfaces
- `version.cpp` - Version print tests across all interfaces

## Downloading Models

Model weights needed for running tests are stored on huggingface. Before running tests, run `python scripts/download_models.py` to fetch the model weights from huggingface first. You shoud occassionally run this script to ensure local model file list is up to date.

### Uploading New Models

- If you ever add new models for testing, place them properly under `modelfiles` and run `python scripts/upload_models.py` to push them to huggingface. Note that this only add new files or overwrite existing ones, but it never cleans old files that do not exist in your local. If you need to clean up something, go to huggingface and do so manually.

## Building Tests

- Test program will build with cmake flags `-DGENIEX_TEST=ON`, make sure plugin flags (`GENIEX_PLUGIN_XXX`) is enabled
- then run with `ctest --test-dir build`.
- When running test executables directly (not via `ctest`), you need to set the `GENIEX_PLUGIN_PATH` environment variable to point to the plugin directory: `GENIEX_PLUGIN_PATH=/path/to/build/out ./build/tests/src/geniex_test_xxx`.
- you can filter test case by several ways:

  **Using CTest (recommended for CI/CD):**

  - Filter by modality: `ctest --test-dir build -R llm`
  - Filter by plugin: `ctest --test-dir build -R llama_cpp`
  - Filter by modality and plugin: `ctest --test-dir build -R llm_llama_cpp`

  **Running test executables directly:**

  Tests are organized using a **`"model_name/test_name"`** subcase naming convention, allowing flexible filtering:

  - **Plugin filtering** with `-ts` or `--test-suite`:

    ```bash
    .\build\tests\src\geniex_test_llm.exe --success -ts="qairt"              # Single plugin
    .\build\tests\src\geniex_test_llm.exe --success -ts="qairt,llama_cpp"   # Multiple plugins
    ```

  - **Subcase filtering** with `-sc` (include) or `-sce` (exclude) using wildcards:

    ```bash
    # All tests for a specific model
    -sc="model_name/*"              # Example: -sc="phi3.5-mini-npu/*"

    # Specific test for specific model
    -sc="model_name/test_name"      # Example: -sc="phi3.5-mini-npu/GenerateJson"

    # Specific test across all models
    -sc="*/test_name"               # Example: -sc="*/GenerateJson"

    # Pattern matching (matches anywhere)
    -sc="*pattern*"                 # Example: -sc="*phi3.5*"

    # Exclude patterns
    -sce="pattern"                  # Example: -sce="*phi*" or -sce="*/GenerateJson"
    ```

  - **Example usage**:

    ```bash
    # All phi3.5 tests
    .\build\tests\src\geniex_test_llm.exe --success -ts="qairt" -sc="*phi3.5*"

    # GenerateJson test across all QAIRT models
    .\build\tests\src\geniex_test_llm.exe --success -ts="qairt" -sc="*/GenerateJson"
    ```

  - **List available tests**:
    ```bash
    .\build\tests\src\geniex_test_llm.exe -ltc
    ```

- On Windows, you may need to append with `-C (Release|Debug)` to properly run via `ctest`.

## Running Tests by Platform

### Windows (MSVC)

```powershell
# Set environment variables
$env:GENIEX_PLUGIN_PATH="./build/out"
$env:GENIEX_TOKEN="key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE0YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA=="

# Run tests (QAIRT backend example)
.\build\tests\src\geniex_test_asr.exe --test-suite="qairt" --success
.\build\tests\src\geniex_test_embedding.exe --test-suite="qairt" --success
.\build\tests\src\geniex_test_llm.exe --test-suite="qairt" --success
.\build\tests\src\geniex_test_cv.exe --test-suite="qairt" --success
.\build\tests\src\geniex_test_vlm.exe --test-suite="qairt" --success
```

### Linux (ARM64 with QAIRT)

```bash
# Set environment variables
export GENIEX_PLUGIN_PATH="./build/out"
export GENIEX_TOKEN="key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE4YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA=="

# Run tests (QAIRT backend example)
./build/tests/src/geniex_test_asr --test-suite="qairt" --success
./build/tests/src/geniex_test_embedding --test-suite="qairt" --success
./build/tests/src/geniex_test_llm --test-suite="qairt" --success
./build/tests/src/geniex_test_cv --test-suite="qairt" --success
./build/tests/src/geniex_test_vlm --test-suite="qairt" --success
```

### Android NPU

```bash
bash scripts/run_android_tests.sh --mode test
adb shell "mkdir -p /data/local/tmp/geniex/"
adb push build-android/out/ /data/local/tmp/geniex/
adb push /Users/zackli/Downloads/lfm2/* /data/local/tmp/geniex/modelfiles/LFM2-1.2B-npu/
adb shell
cd /data/local/tmp/geniex
chmod -R 777 .
export GENIEX_PLUGIN_PATH=/data/local/tmp/geniex/out
export LD_LIBRARY_PATH=/data/local/tmp/geniex/out:/data/local/tmp/geniex/out/llama_cpp:/data/local/tmp/geniex/out/qairt:/data/local/tmp/geniex/out/qairt/htp-files:$LD_LIBRARY_PATH
# for running llama.cpp hexagon, set ADSP_LIBRARY_PATH to /data/local/tmp/geniex/out/llama_cpp
export ADSP_LIBRARY_PATH=/data/local/tmp/geniex/out/qairt/htp-files
export GENIEX_TOKEN="key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE0YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA=="

chmod +x ./out/tests/*

# For QAIRT test
./out/tests/geniex_test_llm --test-suite="qairt" --success
./out/tests/geniex_test_vlm --test-suite="qairt" --success
./out/tests/geniex_test_asr --test-suite="qairt" --success
./out/tests/geniex_test_cv --test-suite="qairt" --success
./out/tests/geniex_test_embedding --test-suite="qairt" --success
./out/tests/geniex_test_rerank --test-suite="qairt" --success

# For llama_cpp test, GenerateStream test case with one specific model on mobile device
adb push test.txt /data/local/tmp/geniex/modelfiles/test.txt
./out/tests/geniex_test_llm --test-suite="llama_cpp" -sc="Llama-3.2-3B-Instruct-Q4_0/GenerateStream" --success
```

## Debug Tests

- Run test, and get failed test case name.
- Run with `gdb --args ./build/tests/src/geniex_test_xxx -tc="XXXX"`, then press `r`. On Windows, you can use `cdb`.
- Progress will pause on failed check, you can start debug now.

## Test Organization Benefits

This refactored structure provides several advantages:

1. **Interface-focused testing**: Tests are organized by ML functionality, making it easier to verify that all backends implement the same interfaces correctly.

1. **Unified build system**: A single CMakeLists.txt handles all backend configurations, reducing duplication

1. **Easy backend comparison**: You can build and run the same test with different backends to compare behavior and performance.

1. **Simplified maintenance**: Adding new backends or new interface tests requires minimal changes to the build system.

1. **Clear separation of concerns**: Backend-specific configuration is isolated in the CMakeLists.txt, while test logic focuses on the ML interfaces.

## Adding New Tests

To add a new test:

1. Create a new test file (e.g., `tts.cpp`) that includes `ml.h` and tests the relevant ML interfaces.
1. Add `Setup` guard to save param, Add `TEST_MAIN()` at end of file
1. Add the test to the appropriate backend sections in `CMakeLists.txt` in `foreach` function.
1. The test will automatically be built for all enabled backends.

## Test Execution Model

The new test infrastructure optimizes model lifecycle management for efficiency:

### Model Lifecycle

1. **Create Model** → Run ALL tests for that model → **Destroy Model** → Next model
2. Models are **reset** (not destroyed) between individual test cases for efficiency
3. Only **one model** exists in memory at a time to prevent resource exhaustion
4. Switching to a different model automatically destroys the previous one

### Example Execution Flow

**Subcase Naming Convention**: Tests use `"model_name/test_name"` format (e.g., `"phi3.5-mini-npu/GenerateBasic"`).

**Single model, all tests** (`-sc="model/*"`):

```
TEST_CASE [enters] → Create model → Test1 (reset) → Test2 (reset) → ... → Destroy model
```

**Single test, all models** (`-sc="*/TestName"`):

```
TEST_CASE [enters] → Create model1 → TestName → Destroy → Create model2 → TestName → Destroy → ...
```

**Key behaviors**:

- Models are created lazily (only when their subcases are entered after filtering)
- Each model is created once and cached (singleton pattern)
- Models are reset (not destroyed) between tests for the same model
- Switching models destroys the previous one automatically

### Known Issues

- **QAIRT Plugin**: May experience destructor hangs when rapidly switching between different models. This is a plugin-level issue in the QAIRT implementation's cleanup code.

## Adding New Backends

To add support for a new backend:

1. Add backend in `CMakeLists.txt`, in `list(APPEND PLUGINS "xxxx")`.
2. Add backend in `tests/include/util.h`, use `PLUGIN_DEF`.
3. Add backend id to `TYPES` macros in each supported model test file.
4. Add backend settings in `setup_guard` on each test source file.
5. Add a `TEST_CASE` with test_suite tag for the new backend.

This structure makes it easy to ensure all backends implement the same ML interfaces correctly.
