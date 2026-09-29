# Running against another QAIRT runtime

A QAIRT runtime ships with the `qairt` plugin. `geniex_set_qairt_runtime_path` points the
plugin at a different one. Call it **before `geniex_init`**.

```c
#include <stdio.h>
#include <string.h>

#include "geniex.h"

/* argv[1] = QAIRT runtime dir, argv[2] = <bundle>/genie_config.json */
int main(int argc, char** argv) {
    if (argc < 3) return 2;

    if (geniex_set_qairt_runtime_path(argv[1]) != GENIEX_SUCCESS) return 1;
    if (geniex_init() != GENIEX_SUCCESS) return 1;

    geniex_LlmCreateInput in;
    memset(&in, 0, sizeof(in));
    in.model_path = argv[2];
    in.plugin_id  = "qairt";
    in.device_id  = "NPU";

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
  -I sdk/pkg-geniex/include qairt-lib.c sdk/build/src/geniex.lib -o qairt-lib.exe
```

```bash
clang -std=c11 -DGENIEX_SHARED -I sdk/pkg-geniex/include qairt-lib.c \
  -L sdk/pkg-geniex/lib -lgeniex -o qairt-lib
```

Run with the SDK on the library path, and `GENIEX_LOG=info` to see which runtime was used:

```pwsh
$env:PATH = "sdk/pkg-geniex/lib;$env:PATH"
$env:GENIEX_LOG = "info"
./qairt-lib.exe C:\qairt\2.48.0 C:\models\Qwen3-4B\genie_config.json
```

```
Overriding the bundled QAIRT runtime from geniex_set_qairt_runtime_path: <path> (host libs: <dir>)
```

`host libs:` is the part that matters — for an SDK root it is the `lib/<triple>` subfolder,
not the root you passed.

| | |
|---|---|
| Accepted paths | a QAIRT SDK root, or a flat folder holding `QnnHtp.dll` / `libQnnHtp.so` |
| Precedence | this call → `GENIEX_QAIRT_LIB` → the bundled runtime |
| After `geniex_init` | `GENIEX_ERROR_COMMON_ALREADY_INITIALIZED`. One runtime per process; `geniex_deinit` does not reset it |

A bundle built for one QAIRT version may not load on another — that surfaces as a QNN context
error from `geniex_llm_create`, not from the setter.

CLI and binding equivalents: [notes/run.md](../../notes/run.md#using-a-custom-qnn-library).
