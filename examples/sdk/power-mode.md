# Controlling the HTP power mode

`geniex_ModelConfig.power_mode` sets the HTP DCVS/clock-management mode for a model, on
both the `qairt` and `llama_cpp` plugins. The field is a `geniex_PowerMode` enum, not a
string: resolve the user-facing alias (`low_power_saver`, `power_saver`, `high_power_saver`,
`low_balanced`, `balanced`, `high_performance`, `sustained_high_performance`, `burst`,
or `"default"`) with `geniex_power_mode_from_alias()` from
[`power_mode_alias.h`](../../sdk/include/power_mode_alias.h) before filling the struct — there's
no exported resolver in the SDK's C ABI itself, and the field has no implicit default (it's
`GENIEX_POWER_MODE_LOW_POWER_SAVER` if left zero-initialized). It's a no-op (logged, not an
error) on `cpu` / `gpu`.

```c
#include <stdio.h>
#include <string.h>

#include "geniex.h"
#include "power_mode_alias.h"

/* argv[1] = power mode alias (e.g. "low_power_saver"), argv[2] = <model>.gguf */
int main(int argc, char** argv) {
    if (argc < 3) return 2;

    if (geniex_init() != GENIEX_SUCCESS) return 1;

    geniex_LlmCreateInput in;
    memset(&in, 0, sizeof(in));
    in.model_path          = argv[2];
    in.plugin_id           = "llama_cpp";
    in.device_id           = "HTP0"; /* anything starting "HTP" -> NPU */
    in.config.n_gpu_layers = -1;     /* 0 forces CPU, which skips the HTP vote entirely */
    if (!geniex_power_mode_from_alias(argv[1], &in.config.power_mode)) {
        fprintf(stderr, "unknown power mode: %s\n", argv[1]);
        return 1;
    }

    geniex_LLM* llm = NULL;
    int32_t     rc  = geniex_llm_create(&in, &llm);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr, "create: %s\n", geniex_get_error_message(rc));
        return 1;
    }

    geniex_GenerationConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_tokens = 32;

    geniex_LlmGenerateInput gin;
    memset(&gin, 0, sizeof(gin));
    gin.prompt_utf8 = "Name three primary colors.";
    gin.config      = &cfg;

    geniex_LlmGenerateOutput out;
    memset(&out, 0, sizeof(out));
    rc = geniex_llm_generate(llm, &gin, &out);
    if (rc == GENIEX_SUCCESS) {
        printf("%s\n", out.full_text);
        geniex_free(out.full_text);
    }

    geniex_llm_destroy(llm);
    geniex_deinit();
    return rc == GENIEX_SUCCESS ? 0 : 1;
}
```

Build. `pkg-geniex/lib` ships only the DLL, so Windows links the import library from the
build tree:

```pwsh
clang --target=arm64-pc-windows-msvc -std=c11 -DGENIEX_SHARED `
  -I sdk/pkg-geniex/include power-mode.c sdk/build/src/geniex.lib -o power-mode.exe
```

```bash
clang -std=c11 -DGENIEX_SHARED -I sdk/pkg-geniex/include power-mode.c \
  -L sdk/pkg-geniex/lib -lgeniex -o power-mode
```

Run with the SDK on the library path, and `GENIEX_LOG=info` to see the mode actually voted:

```pwsh
$env:PATH = "sdk/pkg-geniex/lib;$env:PATH"
$env:GENIEX_LOG = "info"
./power-mode.exe low_power_saver C:\models\Qwen3-4B-Q4_0.gguf
```

```
ggml-hex: HTP0 power mode: low_power_saver
 Also, can you explain the difference between primary, secondary, and tertiary colors?
```

(the raw `prompt_utf8` above skips chat templating, so the base model free-associates instead of
just answering -- `geniex_llm_apply_chat_template` first is the fix, not a `power_mode` concern)

| | |
|---|---|
| Applies to | `qairt` and `llama_cpp`, NPU only |
| Default | `"" ` / `"default"` alias resolves to `burst` — always wins over the model bundle's own `htp_backend_ext_config.json` on qairt |
| Precedence (`llama_cpp`) | only affects HTP sessions created after this call — an already-open session (another model still loaded in the same process) keeps its old mode until released and reacquired |

CLI and binding equivalents: [notes/run.md](../../notes/run.md#power-mode).
