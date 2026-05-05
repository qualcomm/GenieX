# geniex

Python bindings for the **GenieX SDK** — run LLMs and VLMs locally on
Qualcomm platforms (CPU, GPU, Hexagon NPU) with a single `pip install`.

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex
```

Supported platforms (wheels auto-provision the native SDK on install):

| OS      | Arch      |
|---------|-----------|
| Linux   | aarch64   |
| Windows | arm64     |

Python 3.10+ required.

## Library usage

```python
from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "qwen3",               # short alias, HF repo id, or local GGUF path
    device_map="auto",     # "auto" | "cpu" | "gpu" | "npu" | "hybrid"
                           # | "<plugin>" | "<plugin>:<device>"
)

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

# One-shot
output = model.generate(prompt, max_new_tokens=256)
print(output.text)
print(f"[{output.profile.generated_tokens} tok, "
      f"{output.profile.decode_speed:.1f} tok/s, stop={output.profile.stop_reason}]")

# Streaming
streamer = model.generate(prompt, max_new_tokens=256, stream=True)
for chunk in streamer:
    print(chunk, end="", flush=True)

model.close()
```

`from_pretrained` accepts any of: a short alias (resolved via the bundled
registry), a HuggingFace `org/repo` (optionally `org/repo:quant`), or a
local path to a `.gguf` file or a pre-downloaded directory.

### VLM

```python
from geniex import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("qwen3vl", device_map="auto")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "Describe the image."},
    ],
}]
prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
print(model.generate(prompt, images=["/path/to/image.jpg"]).text)
```

### Model management

The same model manager the CLI uses is available programmatically:

```python
from geniex import model_manager as mm

mm.pull("qwen3")                      # alias or "org/repo[:quant]"
paths = mm.get_paths("qwen3")         # resolved local paths
mm.list_models()                      # cached models
mm.remove("qwen3")
```

### Plugin / device enumeration

```python
from geniex import get_plugin_list, get_device_list

for plugin in get_plugin_list():
    print(plugin, get_device_list(plugin))
```

Friendly aliases accepted by `device_map`:

| Alias    | llama_cpp resolves to              | qairt resolves to | Notes                                                                                                                |
|----------|------------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------|
| `cpu`    | empty `device_id`, `ngl=0`         | `NPU` + warning   | Pure CPU for `llama_cpp`. QAIRT is NPU-only; other aliases are coerced with a stderr warning (no hard error).        |
| `gpu`    | `GPUOpenCL`                        | `NPU` + warning   | Adreno via `ggml-opencl`.                                                                                            |
| `npu`    | `HTP0`                             | `NPU`             | Pinned single-session HTP. Deterministic; slower than `hybrid` on LLMs (~30% TTFT).                                  |
| `hybrid` | empty `device_id`, `ngl=999`       | `NPU` + warning   | llama.cpp's per-tensor HTP+CPU scheduler — the fast path on Snapdragon. Used as the default when `device_map="auto"`. |

`device_map="auto"` (the default) picks `hybrid` for `llama_cpp` and
`npu` for `qairt`. Pass a concrete id as `device_map="<plugin>:<device_id>"`
for full control (e.g. `"llama_cpp:HTP0,HTP1,HTP2,HTP3"`). Run
`geniex-py devices` or `geniex._ffi.get_device_list(plugin)` to
enumerate what your host actually exposes.

## CLI

After install, the `geniex-py` console script is on your `$PATH` (the
name avoids clashing with the Go `geniex` binary when both are
installed):

```bash
geniex-py chat qwen3                       # interactive chat (auto-downloads)
geniex-py chat NexaAI/Qwen3-4B-GGUF --quant Q4_K_M
geniex-py chat /path/to/model.gguf --system "You are a concise assistant."

geniex-py pull qwen3                       # download into the cache only
geniex-py ls                               # list cached models (table)
geniex-py ls qwen3                         # show one model's geniex.json
geniex-py rm qwen3                         # remove from cache
geniex-py devices                          # list plugins and their devices
```

Inside chat: `/reset` clears history, `/exit` or Ctrl-D quits, Ctrl-C
interrupts the current reply. `geniex-py <cmd> --help` for all flags.

## Supported backends

| Plugin      | Accelerators                          | Notes                                                                                |
|-------------|---------------------------------------|--------------------------------------------------------------------------------------|
| `llama_cpp` | CPU / GPU / NPU (via ggml backends)   | Default. Devices available depend on the host build (e.g. CUDA, Metal, Vulkan, HTP). |
| `qairt`     | Hexagon NPU                           | Qualcomm Snapdragon only.                                                            |

## Environment variables

| Var                | Purpose                                              |
|--------------------|------------------------------------------------------|
| `GENIEX_DATADIR`   | Model cache directory (default: `~/.cache/geniex`).  |
| `GENIEX_HFTOKEN`   | HuggingFace token for gated repos.                   |
| `GENIEX_LIB_PATH`  | Point at a pre-built `libgeniex.so` / `geniex.dll`.  |

## Building from source

See [BUILD.md](BUILD.md) for dev-mode setup, building the SDK, building
the sdist, and the TestPyPI smoke-test workflow.

## License

Apache 2.0 — see [LICENSE](LICENSE).
