/*
 * Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * geniex-bench — single-cell C inference benchmark, public-API only.
 *
 * Flag naming follows llama.cpp's `llama-bench` (-r / --repetitions,
 * -n / --n-gen, -c / --ctx-size, -t / --threads, -m / --model,
 * -ngl / --n-gpu-layers, --no-warmup) so users moving between the two
 * read the same vocabulary.
 *
 * Defaults:
 *   - n_prompt=512 random tokens (matches llama-bench `pp512`), n_gen=128,
 *     temperature=0.0, seed=42
 *   - 1 warmup + 5 measured runs (configurable)
 *   - LLM prefill skips the tokenizer entirely: each cell rolls
 *     `rand() % vocab_size` for n_prompt positions (BOS at pos 0 when the
 *     model wants one) and feeds the array via input_ids — `pp` is
 *     therefore exactly N for every model
 *   - per-cell aggregation: median / min / max / mean / stdev for ttft_ms,
 *     prefill_tps, decode_tps; median-only for token counts
 *
 * The binary accepts either a raw filesystem path or a model-manager id
 * (e.g. `org/repo[:quant]` or `qualcomm/<aihub_repo>`) for the model
 * argument. When it isn't a filesystem path, the model-manager API is
 * used to download (resume-capable, multi-connection) and resolve the
 * model + mmproj + tokenizer paths, replacing the curl/IWR shell loops
 * the QDC bench run used to run on each device.
 */

#include <geniex.h>
#include <geniex_model.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
/* The MSVC CRT ships struct stat / stat() but not the POSIX type macros. */
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#else
#include <dirent.h>
#endif

/* Fixed VLM prompt. Multimodal cells still go through the bundle's chat
 * template (which needs real text), so the LLM/VLM paths are intentionally
 * asymmetric: LLM prefills random ids; VLM keeps a one-line text payload. */
static const char* const VLM_DEFAULT_PROMPT = "Describe the image.";

#define MAX_PATHS 16

typedef struct {
    const char* plugin;
    const char* device;
    const char* device_id;
    /* model_path is either an absolute filesystem path (treated as a local
     * model directory or .gguf file) or a model-manager id of the form
     * `org/repo[:quant]`. The model-id branch resolves via geniex_model_*
     * (pulling if missing) and the resulting paths are written back into
     * `mm_model_path` / `mm_mmproj` / `mm_tokenizer`. `mm_model_path` and
     * `mm_tokenizer` always shadow their `_path` siblings; `mm_mmproj` only
     * shadows `mmproj_path` when the user explicitly opted into VLM (via
     * --vlm or matrix col 8), so a passively-present mmproj in the bundle
     * cannot redirect a `-p N` LLM bench into the VLM run loop (#1090). */
    const char* model_path;
    const char* tokenizer_path;
    const char* mmproj_path;
    /* Heap-owned copies populated when the model is resolved through the
     * model manager; freed at the end of run_one_cell. */
    char*       mm_model_path;
    char*       mm_mmproj;
    char*       mm_tokenizer;
    bool        force_vlm; /* run VLM path even without an mmproj (QAIRT bundles) */
    bool        mm_is_vlm; /* manager classified the resolved model as VLM (geniex_ModelType) */
    const char* image_paths[MAX_PATHS];
    int32_t     image_count;
    const char* audio_paths[MAX_PATHS];
    int32_t     audio_count;

    int32_t n_prompt;   /* LLM random-ids prefill length (llama-bench -p), used when prompt_buf is NULL */
    char*   prompt_buf; /* heap-owned text prompt loaded via --prompt-file; NULL = use random-ids */
    int32_t max_new_tokens;
    float   temperature;
    int32_t seed;
    int32_t warmup;
    int32_t repeat;
    bool    reset_between_runs; /* true => geniex_llm_reset() before each run, freeing KV */
    bool    accuracy;           /* true => single run (warmup=0, repeat=1), print generated text */
    int32_t n_ctx;
    int32_t n_threads;
    int32_t ngl_override; /* -1 = use resolved alias default; >=0 overrides */

    const char* output_json;
    const char* output_md;
    const char* cell_id;

    /* Matrix mode: one process, one geniex_init, many cells */
    const char* matrix_file;
    const char* output_json_dir;

    /* Model-manager options. Apply to every cell whose `model_path` looks
     * like a model id rather than a filesystem path. */
    const char* mm_data_dir; /* cache root; NULL falls back to GENIEX_DATADIR / ~/.cache/geniex */
    const char* mm_chipset;  /* AI Hub chipset slug (e.g. "qualcomm-snapdragon-x-elite") */
    const char* mm_hub;      /* "auto" | "hf" | "aihub" | "modelscope" | "volces" — default auto */
} options_t;

/* Process-wide guard: geniex_model_init is one-shot per process (a second
 * call returns INVALID_INPUT). matrix mode copies options_t per cell so
 * the flag must live outside the struct. */
static bool g_mm_inited = false;

typedef struct {
    int32_t     run_idx;
    bool        is_warmup;
    int64_t     ttft_us;
    int64_t     prompt_time_us;
    int64_t     decode_time_us;
    int64_t     prompt_tokens;
    int64_t     gen_tokens;
    double      prefill_tps;
    double      decode_tps;
    const char* stop_reason; /* not freed; lifetime tied to SDK output */
    int32_t     status;      /* 0 ok */
    char        err[256];
} run_result_t;

static void die(int32_t code, const char* what) {
    const char* msg = geniex_get_error_message((geniex_ErrorCode)code);
    fprintf(stderr, "ERROR: %s: %s (code=%d)\n", what, msg ? msg : "?", code);
    exit(1);
}

static void check(int32_t code, const char* what) {
    if (code != GENIEX_SUCCESS) {
        die(code, what);
    }
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  Single cell:\n"
        "    %s --plugin {llama_cpp|qairt} --device {cpu|gpu|npu|hybrid|auto} \\\n"
        "                              -m <path> [options]\n"
        "\n"
        "  Matrix (one process, shared geniex_init/deinit):\n"
        "    %s --matrix-file <path> [--output-json-dir <dir>] [shared options]\n"
        "\n"
        "Required (single-cell mode):\n"
        "  --plugin            llama_cpp | qairt\n"
        "  --device            cpu | gpu | npu | hybrid | auto (default auto)\n"
        "  -m, --model VALUE   either a filesystem path (.gguf or bundle dir) or\n"
        "                      a model-manager id `org/repo[:quant]`. The id form\n"
        "                      pulls via geniex_model_pull on first use and reuses\n"
        "                      the cached copy thereafter. The manager's tokenizer\n"
        "                      shadows --tokenizer-path; the manager's mmproj is\n"
        "                      adopted ONLY when VLM is explicitly requested via\n"
        "                      --vlm (single-cell) or the matrix `vlm` column. An\n"
        "                      explicit --mmproj-path / matrix col 6 always wins.\n"
        "\n"
        "Required (matrix mode):\n"
        "  --matrix-file PATH  one cell per line, tab-separated:\n"
        "                      cell_id<TAB>plugin<TAB>device<TAB>model_path_or_id"
        "[<TAB>tokenizer_path][<TAB>mmproj_path][<TAB>image_paths][<TAB>vlm]\n"
        "                      column 4 is a model-manager id when it doesn't look\n"
        "                      like a path (no leading '/' / drive prefix and at\n"
        "                      least one '/'); otherwise it's used verbatim as a\n"
        "                      filesystem path.\n"
        "                      image_paths is comma-separated; vlm non-empty forces\n"
        "                      VLM mode (QAIRT bundles without an mmproj)\n"
        "                      lines starting with '#' are ignored\n"
        "\n"
        "Optional (llama-bench-style names):\n"
        "  -r, --repetitions N    default 5 (measured runs)\n"
        "  -p, --n-prompt N       LLM prefill length (random token ids); default 512.\n"
        "                         Mirrors `llama-bench -p N`: the bench tool fills N\n"
        "                         positions with `rand() %% vocab_size` (BOS at pos 0\n"
        "                         when the model wants one) and feeds them via\n"
        "                         input_ids, so `pp` is exactly N for every model.\n"
        "                         The qairt plugin currently rejects input_ids; the\n"
        "                         tool fails with a clear error pointing to the\n"
        "                         tracking issue.\n"
        "  -n, --n-gen N          tokens to generate per run; default 128\n"
        "  -c, --ctx-size N       model n_ctx (0 = from model, default 0)\n"
        "  -t, --threads N        generation threads (0 = SDK default)\n"
        "  -ngl, --n-gpu-layers N llama_cpp layers to offload; overrides the\n"
        "                         device alias default (needed for a real gpu run)\n"
        "  --warmup N             default 1\n"
        "  --no-warmup            equivalent to --warmup 0\n"
        "  --temperature F        default 0.0\n"
        "  --seed N               default 42; also seeds rand() for prompt ids\n"
        "  --prompt-file PATH     opt out of random-ids prefill: read a UTF-8 prompt\n"
        "                         from PATH and feed it via prompt_utf8 instead. The\n"
        "                         only way to bench plugins that don't support\n"
        "                         input_ids (today: qairt). With this flag, reported\n"
        "                         `pp` is the tokenizer's count, NOT --n-prompt.\n"
        "  --no-reset-between-runs\n"
        "                         keep KV cache across measured runs (default is\n"
        "                         to call geniex_llm_reset() before every run so\n"
        "                         each repetition does the full prefill, matching\n"
        "                         llama-bench semantics)\n"
        "  --accuracy             accuracy mode: force a single run (--warmup 0\n"
        "                         --repetitions 1) and print the generated text to\n"
        "                         stdout, for eyeballing output quality rather than\n"
        "                         speed. Overrides --warmup / --repetitions. Pair\n"
        "                         with --prompt-file for a real prompt; the default\n"
        "                         random-ids prefill produces meaningless text.\n"
        "\n"
        "Optional (multimodal):\n"
        "  --tokenizer-path PATH  explicit tokenizer file\n"
        "  --mmproj-path PATH     multimodal projector — switches to VLM mode\n"
        "  --vlm                  force VLM mode without an mmproj (QAIRT bundles)\n"
        "  --image PATH           image input (VLM); may be passed multiple times\n"
        "  --audio PATH           audio input (VLM); may be passed multiple times\n"
        "  --device-id ID         override resolved device id (e.g. HTP0, GPUOpenCL)\n"
        "\n"
        "Optional (output):\n"
        "  --output-json PATH     (single-cell) write per-cell JSON report\n"
        "  --output-md PATH       (single-cell) write per-cell Markdown row\n"
        "  --output-json-dir DIR  (matrix) write <DIR>/<cell_id>.json per cell\n"
        "  --cell-id ID           (single-cell) cell label used in reports\n"
        "\n"
        "Optional (model manager — applied when -m / matrix col 4 is a model id):\n"
        "  --mm-data-dir DIR      cache root for downloaded models;\n"
        "                         default: $GENIEX_DATADIR or ~/.cache/geniex\n"
        "  --hub NAME             auto | hf | aihub | modelscope | volces\n"
        "                         (default auto). hf reads $GENIEX_HFTOKEN.\n"
        "  --chipset SLUG         AI Hub chipset slug, e.g. qualcomm-snapdragon-x-elite\n"
        "                         (only consumed when the resolved hub is aihub)\n"
        "  --help / -h\n",
        argv0,
        argv0);
}

/* True when `s` looks like a filesystem path rather than a model-manager id.
 *
 * Heuristic: model-manager ids are always `org/repo[:quant]` shape — at
 * least one '/', no leading '/' or '\', no Windows drive prefix, no '.' in
 * the leading segment. Anything else (absolute path, ./relative, plain
 * filename like `model.gguf`, an existing directory) routes to the path
 * branch.
 *
 * Edge case: a bare `model.gguf` in the current working directory — a path
 * — has no '/' so this returns true, which is the correct branch.
 * Conversely an org/repo without quant (e.g. `unsloth/Qwen3-4B-GGUF`) has
 * '/' in the middle and falls through to the model-id branch. */
static bool looks_like_path(const char* s) {
    if (!s || !*s) return true;
    if (s[0] == '/' || s[0] == '.' || s[0] == '\\') return true;
#ifdef _WIN32
    if (s[1] == ':' && (s[2] == '\\' || s[2] == '/')) return true;
#endif
    /* No slash anywhere → treat as a filename in cwd, not a model id. */
    if (!strchr(s, '/')) return true;
    /* `org/repo` shape: keep as model id. */
    return false;
}

/* VLM detection for a local QAIRT bundle:
 * read metadata.json's genie.supports_vision. Lets geniex-bench route a VLM
 * bundle to the VLM run loop even when --vlm / matrix col 8 is unset.
 * `model_path` may be a bundle directory or a file inside it. Any read/parse
 * miss is treated as "not VLM". */
static bool local_bundle_is_vlm(const char* model_path) {
    if (!model_path || !*model_path) return false;

    /* If model_path is a regular file, use its parent; else treat as the dir. */
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", model_path);
    struct stat st;
    if (stat(dir, &st) == 0 && !S_ISDIR(st.st_mode)) {
        char* slash = strrchr(dir, '/');
#ifdef _WIN32
        char* bslash = strrchr(dir, '\\');
        if (bslash && (!slash || bslash > slash)) slash = bslash;
#endif
        if (slash) *slash = '\0';
    }

    char meta[1100];
    snprintf(meta, sizeof(meta), "%s/metadata.json", dir);

    FILE* f = fopen(meta, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (long)(8 * 1024 * 1024)) {
        fclose(f);
        return false;
    }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return false;
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';

    /* Anchor on the "genie" object so this matches the CLI's
     * meta['genie']['supports_vision'] rather than any top-level key.
     * Parser-free: metadata.json is small, machine-generated JSON. */
    bool        is_vlm = false;
    const char* genie  = strstr(buf, "\"genie\"");
    const char* p      = strstr(genie ? genie : buf, "\"supports_vision\"");
    if (p) {
        p += strlen("\"supports_vision\"");
        const char* colon = strchr(p, ':');
        if (colon) {
            const char* v = colon + 1;
            while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') v++;
            if (strncmp(v, "true", 4) == 0) is_vlm = true;
        }
    }
    free(buf);
    return is_vlm;
}

static geniex_HubSource parse_hub(const char* s) {
    if (!s || !*s || strcmp(s, "auto") == 0) return GENIEX_HUB_AUTO;
    if (strcmp(s, "hf") == 0 || strcmp(s, "huggingface") == 0) return GENIEX_HUB_HUGGINGFACE;
    if (strcmp(s, "aihub") == 0) return GENIEX_HUB_AIHUB;
    if (strcmp(s, "modelscope") == 0) return GENIEX_HUB_MODELSCOPE;
    if (strcmp(s, "volces") == 0) return GENIEX_HUB_VOLCES;
    fprintf(stderr, "ERROR: unknown --hub %s (expected auto|hf|aihub|modelscope|volces)\n", s);
    exit(2);
}

/* Split `org/repo[:quant]` into a NUL-terminated `name` slot and an
 * optional `quant` (NULL when absent). Stores both in `buf` so the caller
 * doesn't manage two allocations. */
static void split_id(char* buf, const char** name_out, const char** quant_out) {
    *name_out   = buf;
    *quant_out  = NULL;
    char* colon = strchr(buf, ':');
    if (colon) {
        *colon     = '\0';
        *quant_out = colon + 1;
        if ((*quant_out)[0] == '\0') *quant_out = NULL;
    }
}

/* Resolve a model-manager id to local paths, downloading if missing. On
 * success populates o->mm_model_path / mm_mmproj / mm_tokenizer (heap-
 * owned) and rewrites o->model_path / mmproj_path / tokenizer_path to
 * point at them. Returns 0 on success.
 *
 * Calls geniex_model_init() lazily (idempotent for the matrix-mode case
 * where many cells share one process). */
static int resolve_via_mm(options_t* o, const char* id_in) {
    if (!g_mm_inited) {
        int32_t rc = geniex_model_init(o->mm_data_dir);
        if (rc != GENIEX_SUCCESS) {
            const char* m = geniex_model_last_error_message();
            fprintf(stderr, "ERROR: geniex_model_init: %s (%d)\n", m ? m : "?", rc);
            return 1;
        }
        g_mm_inited = true;
    }

    /* Copy the id into a writable buffer for in-place split_id(). */
    size_t n   = strlen(id_in);
    char*  buf = (char*)malloc(n + 1);
    if (!buf) return 1;
    memcpy(buf, id_in, n + 1);
    const char* name;
    const char* quant;
    split_id(buf, &name, &quant);

    /* Try `get_paths` first — already cached is the common path on the
     * second cell of a matrix. Fall through to pull on file-not-found. */
    geniex_ModelPaths paths;
    memset(&paths, 0, sizeof(paths));
    int32_t rc = geniex_model_get_paths(name, &paths);
    if (rc != GENIEX_SUCCESS) {
        geniex_ModelPullInput in;
        memset(&in, 0, sizeof(in));
        in.struct_size = (uint32_t)sizeof(in);
        in.model_name  = name;
        in.quant       = quant;
        in.hub         = parse_hub(o->mm_hub);
        in.chipset     = o->mm_chipset;
        in.model_type  = GENIEX_MODEL_TYPE_AUTO;
        fprintf(stderr, "[mm  ] pulling %s%s%s ...\n", name, quant ? ":" : "", quant ? quant : "");
        rc = geniex_model_pull(&in);
        if (rc != GENIEX_SUCCESS) {
            const char* m = geniex_model_last_error_message();
            fprintf(stderr, "ERROR: geniex_model_pull(%s): %s (%d)\n", id_in, m ? m : "?", rc);
            free(buf);
            return 1;
        }
        rc = geniex_model_get_paths(name, &paths);
        if (rc != GENIEX_SUCCESS) {
            const char* m = geniex_model_last_error_message();
            fprintf(stderr, "ERROR: geniex_model_get_paths(%s): %s (%d)\n", id_in, m ? m : "?", rc);
            free(buf);
            return 1;
        }
    }
    free(buf);

    /* Take ownership of the heap strings the SDK handed us. */
    o->mm_model_path = paths.model_path;
    o->mm_mmproj     = paths.mmproj_path;
    o->mm_tokenizer  = paths.tokenizer_path;
    /* Capture the manager's LLM/VLM classification (geniex_ModelType) — the
     * CLI's _is_vlm() signal (3). */
    o->mm_is_vlm     = (paths.model_type == GENIEX_MODEL_TYPE_VLM);
    paths.model_path = paths.mmproj_path = paths.tokenizer_path = NULL;
    /* model_dir / model_name / plugin_id aren't consumed here; free via the
     * paths_free() helper to keep the allocator pairing intact. */
    geniex_model_paths_free(&paths);

    o->model_path = o->mm_model_path;
    /* QAIRT VLM bundles have no mmproj and the dispatcher has no LLM factory
     * for VLM model_ids, so a VLM bundle in run_llm hard-fails. Force
     * the VLM run loop when the manager classified it as VLM. */
    if (o->mm_is_vlm && o->plugin && strcmp(o->plugin, "qairt") == 0 && !o->force_vlm) {
        fprintf(stderr, "[mm  ] %s is a VLM bundle; forcing VLM run loop\n", id_in);
        o->force_vlm = true;
    }
    /* Only adopt the manager's mmproj when the user explicitly opted into VLM
     * (--vlm or the matrix `vlm` column). A passively-present mmproj sibling
     * in the manager bundle (e.g. unsloth/gemma-4-E2B-it-GGUF ships an mmproj
     * next to the LLM gguf) must NOT flip the bench into the VLM run loop —
     * that replaces random-ids prefill with a real chat-templated sampling
     * run, breaking the llama-bench contract that `-p N` runs N decode steps
     * regardless of model semantics (#1090). An explicit --mmproj-path or
     * matrix col 6 still wins, so VLM cells that name their projector keep
     * working. */
    if (o->force_vlm && o->mmproj_path == NULL && o->mm_mmproj) {
        o->mmproj_path = o->mm_mmproj;
    }
    if (o->mm_tokenizer) o->tokenizer_path = o->mm_tokenizer;
    fprintf(stderr, "[mm  ] resolved %s -> %s\n", id_in, o->mm_model_path);
    return 0;
}

/* If `path` is a directory, return a heap-allocated path to a regular file
 * inside it (preferring `tokenizer.json`, otherwise the lexicographically
 * first regular file). The SDK derives the model dir via `parent_path()`,
 * so it needs a *file* path, not a directory path. Mirrors
 * `_resolve_local_anchor` in bindings/python/geniex/auto.py:122.
 *
 * If `path` is a regular file (e.g. an explicit *.gguf), returns NULL —
 * the caller should keep using the original path. Callers must free the
 * returned string. */
static char* resolve_local_anchor(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return NULL;
    }

    size_t plen = strlen(path);
    /* Prefer tokenizer.json. */
    {
        const char* leaf = "/tokenizer.json";
        char*       buf  = (char*)malloc(plen + strlen(leaf) + 1);
        if (!buf) return NULL;
        snprintf(buf, plen + strlen(leaf) + 1, "%s%s", path, leaf);
        if (stat(buf, &st) == 0 && S_ISREG(st.st_mode)) {
            return buf;
        }
        free(buf);
    }

    /* Fallback: pick the lexicographically first regular file. */
    char* best = NULL;
#ifdef _WIN32
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA ffd;
    HANDLE           h = FindFirstFileA(pattern, &ffd);
    if (h == INVALID_HANDLE_VALUE) return NULL;
    do {
        const char* name = ffd.cFileName;
        if (name[0] == '.') continue;
        size_t need = plen + 1 + strlen(name) + 1;
        char*  cand = (char*)malloc(need);
        if (!cand) {
            free(best);
            FindClose(h);
            return NULL;
        }
        snprintf(cand, need, "%s/%s", path, name);
        if (stat(cand, &st) == 0 && S_ISREG(st.st_mode)) {
            if (!best || strcmp(cand, best) < 0) {
                free(best);
                best = cand;
            } else {
                free(cand);
            }
        } else {
            free(cand);
        }
    } while (FindNextFileA(h, &ffd));
    FindClose(h);
#else
    DIR* d = opendir(path);
    if (!d) return NULL;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        size_t need = plen + 1 + strlen(e->d_name) + 1;
        char*  cand = (char*)malloc(need);
        if (!cand) {
            free(best);
            closedir(d);
            return NULL;
        }
        snprintf(cand, need, "%s/%s", path, e->d_name);
        if (stat(cand, &st) == 0 && S_ISREG(st.st_mode)) {
            if (!best || strcmp(cand, best) < 0) {
                free(best);
                best = cand;
            } else {
                free(cand);
            }
        } else {
            free(cand);
        }
    }
    closedir(d);
#endif
    return best;
}

/* Load whole file into a heap buffer (caller frees). Used by --prompt-file
 * for plugins that don't support input_ids (qairt). */
static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "ERROR: oom slurping %s\n", path);
        exit(1);
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        fprintf(stderr, "ERROR: short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    buf[sz] = '\0';
    return buf;
}

static const char* arg_value(int argc, char** argv, int* i, const char* flag) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ERROR: %s requires a value\n", flag);
        exit(2);
    }
    *i += 1;
    return argv[*i];
}

/* Optional per-token sleep (for the on_token-overhead study). Read once at
 * startup; default 0 = no-op callback (the production case). Hot path stays
 * branch-predictable. */
static int g_token_callback_delay_us = 0;

static void parse_args(int argc, char** argv, options_t* o) {
    o->plugin             = NULL;
    o->device             = "auto";
    o->device_id          = NULL;
    o->model_path         = NULL;
    o->tokenizer_path     = NULL;
    o->mmproj_path        = NULL;
    o->mm_model_path      = NULL;
    o->mm_mmproj          = NULL;
    o->mm_tokenizer       = NULL;
    o->force_vlm          = false;
    o->mm_is_vlm          = false;
    o->image_count        = 0;
    o->audio_count        = 0;
    o->n_prompt           = 512;
    o->prompt_buf         = NULL;
    o->max_new_tokens     = 128;
    o->temperature        = 0.0f;
    o->seed               = 42;
    o->warmup             = 1;
    o->repeat             = 5;
    o->reset_between_runs = true;
    o->accuracy           = false;
    o->n_ctx              = 0;
    o->n_threads          = 0;
    o->ngl_override       = -1;
    o->output_json        = NULL;
    o->output_md          = NULL;
    o->cell_id            = NULL;
    o->matrix_file        = NULL;
    o->output_json_dir    = NULL;
    o->mm_data_dir        = NULL;
    o->mm_chipset         = NULL;
    o->mm_hub             = NULL;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(a, "--plugin") == 0) {
            o->plugin = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--device") == 0) {
            o->device = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--device-id") == 0) {
            o->device_id = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "-m") == 0 || strcmp(a, "--model") == 0) {
            o->model_path = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--tokenizer-path") == 0) {
            o->tokenizer_path = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--mmproj-path") == 0) {
            o->mmproj_path = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--vlm") == 0) {
            o->force_vlm = true;
        } else if (strcmp(a, "--image") == 0) {
            if (o->image_count >= MAX_PATHS) {
                fprintf(stderr, "ERROR: too many --image\n");
                exit(2);
            }
            o->image_paths[o->image_count++] = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--audio") == 0) {
            if (o->audio_count >= MAX_PATHS) {
                fprintf(stderr, "ERROR: too many --audio\n");
                exit(2);
            }
            o->audio_paths[o->audio_count++] = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "-p") == 0 || strcmp(a, "--n-prompt") == 0) {
            o->n_prompt = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--prompt-file") == 0) {
            o->prompt_buf = slurp(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-n") == 0 || strcmp(a, "--n-gen") == 0) {
            o->max_new_tokens = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--temperature") == 0) {
            o->temperature = (float)atof(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--seed") == 0) {
            o->seed = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--warmup") == 0) {
            o->warmup = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--no-warmup") == 0) {
            o->warmup = 0;
        } else if (strcmp(a, "-r") == 0 || strcmp(a, "--repetitions") == 0) {
            o->repeat = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--no-reset-between-runs") == 0) {
            o->reset_between_runs = false;
        } else if (strcmp(a, "--accuracy") == 0) {
            o->accuracy = true;
        } else if (strcmp(a, "-c") == 0 || strcmp(a, "--ctx-size") == 0) {
            o->n_ctx = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-t") == 0 || strcmp(a, "--threads") == 0) {
            o->n_threads = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-ngl") == 0 || strcmp(a, "--n-gpu-layers") == 0) {
            o->ngl_override = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--output-json") == 0) {
            o->output_json = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--output-md") == 0) {
            o->output_md = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--cell-id") == 0) {
            o->cell_id = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--matrix-file") == 0) {
            o->matrix_file = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--output-json-dir") == 0) {
            o->output_json_dir = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--mm-data-dir") == 0) {
            o->mm_data_dir = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--chipset") == 0) {
            o->mm_chipset = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--hub") == 0) {
            o->mm_hub = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--token-callback-delay-us") == 0) {
            g_token_callback_delay_us = atoi(arg_value(argc, argv, &i, a));
        } else {
            fprintf(stderr, "ERROR: unknown arg %s\n", a);
            usage(argv[0]);
            exit(2);
        }
    }

    /* Accuracy mode is about eyeballing the generated text, not timing: pin a
     * single measured run with no warmup regardless of --warmup / -r. */
    if (o->accuracy) {
        o->warmup = 0;
        o->repeat = 1;
    }

    if (o->matrix_file) {
        /* In matrix mode, --plugin/--device/--model come from each line of
         * the matrix file, not from argv. */
        if (o->repeat < 1) {
            fprintf(stderr, "ERROR: --repetitions must be >=1\n");
            exit(2);
        }
        if (o->n_prompt < 1) {
            fprintf(stderr, "ERROR: --n-prompt must be >=1\n");
            exit(2);
        }
        return;
    }

    if (!o->plugin) {
        fprintf(stderr, "ERROR: --plugin is required\n");
        exit(2);
    }
    if (!o->model_path) {
        fprintf(stderr, "ERROR: --model is required\n");
        exit(2);
    }
    if (o->repeat < 1) {
        fprintf(stderr, "ERROR: --repetitions must be >=1\n");
        exit(2);
    }
    if (o->n_prompt < 1) {
        fprintf(stderr, "ERROR: --n-prompt must be >=1\n");
        exit(2);
    }
}

static void busy_wait_us(int us) {
    if (us <= 0) return;
#ifdef _WIN32
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    LONGLONG target_ticks = (LONGLONG)us * freq.QuadPart / 1000000LL;
    do {
        QueryPerformanceCounter(&t1);
    } while ((t1.QuadPart - t0.QuadPart) < target_ticks);
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long target_ns = (long)us * 1000L;
    do {
        clock_gettime(CLOCK_MONOTONIC, &t1);
    } while (((t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec)) < target_ns);
#endif
}

/* Streaming callback. By default a no-op (the C consumer's expected pattern).
 * --token-callback-delay-us N inflates each call to N microseconds via
 * monotonic-clock busy-wait, simulating the ~per-token overhead a Python
 * binding pays for ctypes wrapper + GIL acquire/release. */
static bool on_token(const char* token, void* user_data) {
    (void)token;
    (void)user_data;
    busy_wait_us(g_token_callback_delay_us);
    return true;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static void summarize(const double* values, int n, double* median, double* lo, double* hi) {
    double* tmp = (double*)malloc(sizeof(double) * (size_t)n);
    memcpy(tmp, values, sizeof(double) * (size_t)n);
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    *lo     = tmp[0];
    *hi     = tmp[n - 1];
    *median = (n % 2 == 1) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
}

/* Same shape as summarize() but also yields mean / sample stdev. n=1 emits
 * stdev=0 (sample stdev with one observation is undefined; surface 0 to
 * match llama-bench's `123.4 ± 0.0` rendering). */
static void summarize_full(
    const double* values, int n, double* median, double* lo, double* hi, double* mean, double* sd) {
    summarize(values, n, median, lo, hi);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += values[i];
    *mean = sum / (double)n;
    if (n < 2) {
        *sd = 0.0;
        return;
    }
    double sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = values[i] - *mean;
        sq += d * d;
    }
    *sd = sqrt(sq / (double)(n - 1));
}

/* Total bytes occupied by the model on disk:
 *  - regular file (.gguf): st_size of the file
 *  - directory (qairt bundle): recursive sum of S_ISREG children
 * Returns 0 on stat failure or unknown layout. */
static int64_t compute_model_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (S_ISREG(st.st_mode)) return (int64_t)st.st_size;
    if (!S_ISDIR(st.st_mode)) return 0;

    int64_t total = 0;
    size_t  plen  = strlen(path);
#ifdef _WIN32
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA ffd;
    HANDLE           h = FindFirstFileA(pattern, &ffd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        const char* name = ffd.cFileName;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
        size_t need = plen + 1 + strlen(name) + 1;
        char*  cand = (char*)malloc(need);
        if (!cand) {
            FindClose(h);
            return total;
        }
        snprintf(cand, need, "%s/%s", path, name);
        total += compute_model_size(cand);
        free(cand);
    } while (FindNextFileA(h, &ffd));
    FindClose(h);
#else
    DIR* d = opendir(path);
    if (!d) return 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        size_t need = plen + 1 + strlen(e->d_name) + 1;
        char*  cand = (char*)malloc(need);
        if (!cand) {
            closedir(d);
            return total;
        }
        snprintf(cand, need, "%s/%s", path, e->d_name);
        total += compute_model_size(cand);
        free(cand);
    }
    closedir(d);
#endif
    return total;
}

/* Build a VLM prompt by running the bundle's chat template over a single user
 * message holding the text plus one image/audio content part per media file.
 * Both plugins need this: QAIRT's genie pipeline relies on the template's
 * vision tokens to place the image, and llama_cpp's apply_chat_template emits
 * the mtmd media marker. Returns heap text the caller frees with geniex_free,
 * or NULL on failure. */
static char* build_vlm_prompt(geniex_VLM* vlm, const options_t* o, const char* base_prompt) {
    geniex_VlmContent contents[1 + 2 * MAX_PATHS];
    int32_t           nc = 0;
    contents[nc].type    = "text";
    contents[nc].text    = base_prompt;
    nc++;
    for (int32_t i = 0; i < o->image_count; ++i) {
        contents[nc].type = "image";
        contents[nc].text = o->image_paths[i];
        nc++;
    }
    for (int32_t i = 0; i < o->audio_count; ++i) {
        contents[nc].type = "audio";
        contents[nc].text = o->audio_paths[i];
        nc++;
    }

    geniex_VlmChatMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.role          = "user";
    msg.contents      = contents;
    msg.content_count = nc;

    geniex_VlmApplyChatTemplateInput  tin;
    geniex_VlmApplyChatTemplateOutput tout;
    memset(&tin, 0, sizeof(tin));
    memset(&tout, 0, sizeof(tout));
    tin.messages      = &msg;
    tin.message_count = 1;

    int32_t rc = geniex_vlm_apply_chat_template(vlm, &tin, &tout);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr,
            "ERROR: geniex_vlm_apply_chat_template: %s (%d)\n",
            geniex_get_error_message((geniex_ErrorCode)rc),
            rc);
        return NULL;
    }
    return tout.formatted_text;
}

/* Print generated text with a `[gen ]` prefix on every line, so multi-line
 * output stays greppable and visually attributed (used by --accuracy). */
static void print_gen_text(const char* text) {
    const char* line = text;
    while (*line) {
        const char* nl  = strchr(line, '\n');
        size_t      len = nl ? (size_t)(nl - line) : strlen(line);
        fprintf(stdout, "[gen ] %.*s\n", (int)len, line);
        if (!nl) break;
        line = nl + 1;
    }
}

/* ----------------------------- LLM run loop ----------------------------- */

static void fill_sampler(geniex_SamplerConfig* s, const options_t* o) {
    memset(s, 0, sizeof(*s));
    s->temperature        = o->temperature;
    s->top_p              = 1.0f;
    s->top_k              = 0;
    s->min_p              = 0.0f;
    s->repetition_penalty = 1.0f;
    s->seed               = o->seed;
}

static void fill_gen_config(geniex_GenerationConfig* g, geniex_SamplerConfig* s, const options_t* o, bool with_media) {
    memset(g, 0, sizeof(*g));
    g->max_tokens     = o->max_new_tokens;
    g->sampler_config = s;
    if (with_media && o->image_count > 0) {
        g->image_paths = (geniex_Path*)o->image_paths;
        g->image_count = o->image_count;
    }
    if (with_media && o->audio_count > 0) {
        g->audio_paths = (geniex_Path*)o->audio_paths;
        g->audio_count = o->audio_count;
    }
}

static void fill_model_config(geniex_ModelConfig* c, const options_t* o, int32_t ngl) {
    memset(c, 0, sizeof(*c));
    c->n_ctx        = o->n_ctx;
    c->n_threads    = o->n_threads;
    c->n_gpu_layers = ngl;
    c->max_tokens   = o->max_new_tokens;
}

static void run_llm(const options_t* o, const char* device_id, int32_t ngl, run_result_t* out) {
    geniex_LlmCreateInput cin;
    memset(&cin, 0, sizeof(cin));
    cin.model_name     = "benchmark";
    cin.model_path     = o->model_path;
    cin.tokenizer_path = o->tokenizer_path; /* may be NULL */
    cin.plugin_id      = o->plugin;
    cin.device_id      = device_id; /* may be NULL */
    fill_model_config(&cin.config, o, ngl);

    geniex_LLM* llm = NULL;
    check(geniex_llm_create(&cin, &llm), "geniex_llm_create");

    /* Two prefill modes, picked by whether --prompt-file was passed:
     *   - prompt_buf != NULL: feed prompt_utf8 verbatim (the plugin tokenizes).
     *     `pp` is the tokenizer's count, NOT n_prompt. Required for plugins
     *     that don't accept input_ids (today: qairt).
     *   - prompt_buf == NULL: random-ids mode (mirrors llama-bench
     *     test_prompt) — query vocab + BOS via geniex_llm_get_model_info,
     *     fill n_prompt positions with rand() % vocab_size, overwrite pos 0
     *     with BOS when add_bos. `pp` is exactly n_prompt. */
    int32_t* tokens = NULL;
    if (!o->prompt_buf) {
        geniex_LlmModelInfo info;
        int32_t             rc_info = geniex_llm_get_model_info(llm, &info);
        if (rc_info != GENIEX_SUCCESS || info.vocab_size <= 0) {
            const char* msg = geniex_get_error_message((geniex_ErrorCode)rc_info);
            fprintf(stderr,
                "ERROR: %s plugin does not support random-ids prefill "
                "(geniex_llm_get_model_info: %s, code=%d). "
                "Pass --prompt-file PATH to use text-prompt mode instead, "
                "or see https://github.com/qualcomm/GenieX/issues/1008.\n",
                o->plugin,
                msg ? msg : "?",
                rc_info);
            geniex_llm_destroy(llm);
            exit(1);
        }

        tokens = (int32_t*)malloc((size_t)o->n_prompt * sizeof(int32_t));
        if (!tokens) {
            fprintf(stderr, "ERROR: oom allocating %d prompt tokens\n", o->n_prompt);
            geniex_llm_destroy(llm);
            exit(1);
        }
        srand((unsigned)o->seed);
        for (int32_t k = 0; k < o->n_prompt; ++k) {
            tokens[k] = (int32_t)(rand() % info.vocab_size);
        }
        if (info.add_bos && info.bos_token >= 0 && o->n_prompt > 0) {
            tokens[0] = info.bos_token;
        }
    }

    geniex_SamplerConfig    sampler;
    geniex_GenerationConfig gconfig;
    fill_sampler(&sampler, o);
    fill_gen_config(&gconfig, &sampler, o, /*with_media=*/false);

    int32_t total = o->warmup + o->repeat;
    for (int32_t i = 0; i < total; ++i) {
        bool    is_warmup = (i < o->warmup);
        int32_t run_idx   = is_warmup ? i : (i - o->warmup);

        if (o->reset_between_runs) {
            check(geniex_llm_reset(llm), "geniex_llm_reset");
        }

        geniex_LlmGenerateInput  gin;
        geniex_LlmGenerateOutput gout;
        memset(&gin, 0, sizeof(gin));
        memset(&gout, 0, sizeof(gout));
        if (o->prompt_buf) {
            gin.prompt_utf8 = o->prompt_buf;
        } else {
            gin.input_ids       = tokens;
            gin.input_ids_count = o->n_prompt;
        }
        gin.config   = &gconfig;
        gin.on_token = on_token;

        int32_t rc = geniex_llm_generate(llm, &gin, &gout);
        if (rc != GENIEX_SUCCESS) {
            const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
            fprintf(stderr, "ERROR: geniex_llm_generate run %d failed: %s (%d)\n", run_idx, msg ? msg : "?", rc);
            free(tokens);
            geniex_llm_destroy(llm);
            exit(1);
        }

        if (!is_warmup) {
            run_result_t* r = &out[run_idx];
            memset(r, 0, sizeof(*r));
            r->run_idx        = run_idx;
            r->ttft_us        = gout.profile_data.ttft;
            r->prompt_time_us = gout.profile_data.prompt_time;
            r->decode_time_us = gout.profile_data.decode_time;
            r->prompt_tokens  = gout.profile_data.prompt_tokens;
            r->gen_tokens     = gout.profile_data.generated_tokens;
            r->prefill_tps    = gout.profile_data.prefill_speed;
            r->decode_tps     = gout.profile_data.decoding_speed;
            r->stop_reason    = gout.profile_data.stop_reason;
            r->status         = 0;
        }

        if (!is_warmup && o->accuracy && gout.full_text) {
            print_gen_text(gout.full_text);
        }
        if (gout.full_text) {
            geniex_free(gout.full_text);
        }
    }

    free(tokens);
    check(geniex_llm_destroy(llm), "geniex_llm_destroy");
}

/* ----------------------------- VLM run loop ----------------------------- */

static void run_vlm(const options_t* o, const char* device_id, int32_t ngl, run_result_t* out) {
    geniex_VlmCreateInput cin;
    memset(&cin, 0, sizeof(cin));
    cin.model_name     = "benchmark";
    cin.model_path     = o->model_path;
    cin.mmproj_path    = o->mmproj_path;
    cin.tokenizer_path = o->tokenizer_path;
    cin.plugin_id      = o->plugin;
    cin.device_id      = device_id;
    fill_model_config(&cin.config, o, ngl);

    geniex_VLM* vlm = NULL;
    check(geniex_vlm_create(&cin, &vlm), "geniex_vlm_create");

    geniex_SamplerConfig    sampler;
    geniex_GenerationConfig gconfig;
    fill_sampler(&sampler, o);
    fill_gen_config(&gconfig, &sampler, o, /*with_media=*/true);

    int32_t total = o->warmup + o->repeat;
    for (int32_t i = 0; i < total; ++i) {
        bool    is_warmup = (i < o->warmup);
        int32_t run_idx   = is_warmup ? i : (i - o->warmup);
        /* VLM generate() takes a fully-templated prompt; run a fixed base
         * text + media through the bundle's chat template so the image
         * tokens land right. The LLM path uses random-ids and skips the
         * tokenizer; the VLM path can't because the chat template needs
         * real text plus typed content parts. */
        char* prompt = build_vlm_prompt(vlm, o, VLM_DEFAULT_PROMPT);
        if (!prompt) {
            geniex_vlm_destroy(vlm);
            exit(1);
        }

        if (o->reset_between_runs) {
            check(geniex_vlm_reset(vlm), "geniex_vlm_reset");
        }

        geniex_VlmGenerateInput  gin;
        geniex_VlmGenerateOutput gout;
        memset(&gin, 0, sizeof(gin));
        memset(&gout, 0, sizeof(gout));
        gin.prompt_utf8 = prompt;
        gin.config      = &gconfig;
        gin.on_token    = on_token;

        int32_t rc = geniex_vlm_generate(vlm, &gin, &gout);
        if (rc != GENIEX_SUCCESS) {
            const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
            fprintf(stderr, "ERROR: geniex_vlm_generate run %d failed: %s (%d)\n", run_idx, msg ? msg : "?", rc);
            geniex_free(prompt);
            geniex_vlm_destroy(vlm);
            exit(1);
        }

        if (!is_warmup) {
            run_result_t* r = &out[run_idx];
            memset(r, 0, sizeof(*r));
            r->run_idx        = run_idx;
            r->ttft_us        = gout.profile_data.ttft;
            r->prompt_time_us = gout.profile_data.prompt_time;
            r->decode_time_us = gout.profile_data.decode_time;
            r->prompt_tokens  = gout.profile_data.prompt_tokens;
            r->gen_tokens     = gout.profile_data.generated_tokens;
            r->prefill_tps    = gout.profile_data.prefill_speed;
            r->decode_tps     = gout.profile_data.decoding_speed;
            r->stop_reason    = gout.profile_data.stop_reason;
            r->status         = 0;
        }

        if (!is_warmup && o->accuracy && gout.full_text) {
            print_gen_text(gout.full_text);
        }
        if (gout.full_text) {
            geniex_free(gout.full_text);
        }
        geniex_free(prompt);
        /* Unlike the LLM loop, VLM must reset between runs: the image is
         * attached to the first message and consumed into the KV cache, so a
         * second run with the same history re-sends an already-processed
         * prompt and generates nothing (prompt_tokens=0, immediate eos). */
        if (i + 1 < total) {
            check(geniex_vlm_reset(vlm), "geniex_vlm_reset");
        }
    }

    check(geniex_vlm_destroy(vlm), "geniex_vlm_destroy");
}

/* ----------------------------- Reporting ----------------------------- */

typedef struct {
    double ttft_ms_med, ttft_ms_lo, ttft_ms_hi, ttft_ms_mean, ttft_ms_sd;
    double prefill_med, prefill_lo, prefill_hi, prefill_mean, prefill_sd;
    double decode_med, decode_lo, decode_hi, decode_mean, decode_sd;
    double gen_tokens_med;
    double prompt_tokens_med;
} agg_t;

static void aggregate(const run_result_t* runs, int n, agg_t* a) {
    double* tmp = (double*)malloc(sizeof(double) * (size_t)n);
    if (!tmp) {
        fprintf(stderr, "ERROR: oom\n");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        tmp[i] = (double)runs[i].ttft_us / 1000.0;
    }
    summarize_full(tmp, n, &a->ttft_ms_med, &a->ttft_ms_lo, &a->ttft_ms_hi, &a->ttft_ms_mean, &a->ttft_ms_sd);
    for (int i = 0; i < n; ++i) tmp[i] = runs[i].prefill_tps;
    summarize_full(tmp, n, &a->prefill_med, &a->prefill_lo, &a->prefill_hi, &a->prefill_mean, &a->prefill_sd);
    for (int i = 0; i < n; ++i) tmp[i] = runs[i].decode_tps;
    summarize_full(tmp, n, &a->decode_med, &a->decode_lo, &a->decode_hi, &a->decode_mean, &a->decode_sd);
    for (int i = 0; i < n; ++i) tmp[i] = (double)runs[i].gen_tokens;
    double med, lo, hi;
    summarize(tmp, n, &med, &lo, &hi);
    a->gen_tokens_med = med;
    for (int i = 0; i < n; ++i) tmp[i] = (double)runs[i].prompt_tokens;
    summarize(tmp, n, &med, &lo, &hi);
    a->prompt_tokens_med = med;
    free(tmp);
}

static void print_summary(const options_t* o, const char* device_id, int32_t ngl, const agg_t* a) {
    fprintf(stdout,
        "[ok  ] %s  plugin=%s device=%s%s%s%s ngl=%d "
        "ttft=%.1fms prefill=%.1ftps decode=%.1ftps gen=%.0f tok\n",
        o->cell_id ? o->cell_id : "cell",
        o->plugin,
        o->device,
        device_id ? "(id=" : "",
        device_id ? device_id : "",
        device_id ? ")" : "",
        ngl,
        a->ttft_ms_med,
        a->prefill_med,
        a->decode_med,
        a->gen_tokens_med);
}

/* Write a JSON string literal, escaping the characters JSON forbids raw.
 * Needed for paths: Windows model paths carry backslashes that would
 * otherwise emit invalid escape sequences (e.g. "C:\Users"). */
static void json_write_quoted(FILE* f, const char* v) {
    fputc('"', f);
    for (const unsigned char* p = (const unsigned char*)v; *p; ++p) {
        switch (*p) {
            case '"':
                fputs("\\\"", f);
                break;
            case '\\':
                fputs("\\\\", f);
                break;
            case '\n':
                fputs("\\n", f);
                break;
            case '\r':
                fputs("\\r", f);
                break;
            case '\t':
                fputs("\\t", f);
                break;
            default:
                if (*p < 0x20)
                    fprintf(f, "\\u%04x", *p);
                else
                    fputc((int)*p, f);
        }
    }
    fputc('"', f);
}

/* JSON helpers: tiny. String values are escaped via json_write_quoted. */
static void json_field_str(FILE* f, const char* k, const char* v, bool last) {
    fprintf(f, "    \"%s\": ", k);
    if (v)
        json_write_quoted(f, v);
    else
        fprintf(f, "null");
    fprintf(f, last ? "\n" : ",\n");
}
static void json_field_dbl(FILE* f, const char* k, double v, bool last) {
    fprintf(f, "    \"%s\": %.6f%s", k, v, last ? "\n" : ",\n");
}
static void json_field_i64(FILE* f, const char* k, int64_t v, bool last) {
    fprintf(f, "    \"%s\": %lld%s", k, (long long)v, last ? "\n" : ",\n");
}

static void write_json(const options_t* o, const char* device_id, int32_t ngl, int64_t model_size_bytes,
    const run_result_t* runs, const agg_t* a) {
    FILE* f = fopen(o->output_json, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for write\n", o->output_json);
        exit(1);
    }
    fprintf(f, "{\n");
    json_field_str(f, "schema_version", "3", false);
    json_field_str(f, "cell_id", o->cell_id ? o->cell_id : "cell", false);
    json_field_str(f, "plugin", o->plugin, false);
    json_field_str(f, "device", o->device, false);
    json_field_str(f, "device_id", device_id, false);
    json_field_str(f, "model_path", o->model_path, false);
    json_field_i64(f, "model_size_bytes", model_size_bytes, false);
    json_field_str(f, "qairt_version", geniex_get_plugin_version("qairt"), false);
    json_field_str(f, "llama_cpp_version", geniex_get_plugin_version("llama_cpp"), false);
    fprintf(f, "    \"params\": {\n");
    fprintf(f,
        "      \"warmup\": %d, \"repetitions\": %d, \"n_prompt\": %d, \"n_gen\": %d,\n"
        "      \"temperature\": %.6f, \"seed\": %d, \"n_ctx\": %d, \"n_threads\": %d, \"n_gpu_layers\": %d\n",
        o->warmup,
        o->repeat,
        o->n_prompt,
        o->max_new_tokens,
        (double)o->temperature,
        o->seed,
        o->n_ctx,
        o->n_threads,
        ngl);
    fprintf(f, "    },\n");
    fprintf(f, "    \"runs\": [\n");
    for (int i = 0; i < o->repeat; ++i) {
        const run_result_t* r = &runs[i];
        fprintf(f,
            "      {\"run_idx\": %d, \"ttft_us\": %lld, \"prompt_tokens\": %lld, "
            "\"gen_tokens\": %lld, \"prefill_tps\": %.6f, \"decode_tps\": %.6f, "
            "\"prompt_time_us\": %lld, \"decode_time_us\": %lld, \"stop_reason\": %s%s%s}%s\n",
            r->run_idx,
            (long long)r->ttft_us,
            (long long)r->prompt_tokens,
            (long long)r->gen_tokens,
            r->prefill_tps,
            r->decode_tps,
            (long long)r->prompt_time_us,
            (long long)r->decode_time_us,
            r->stop_reason ? "\"" : "null",
            r->stop_reason ? r->stop_reason : "",
            r->stop_reason ? "\"" : "",
            (i + 1 < o->repeat) ? "," : "");
    }
    fprintf(f, "    ],\n");
    fprintf(f, "    \"agg\": {\n");
    fprintf(f,
        "      \"ttft_ms\":     {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f, \"mean\": %.6f, \"stdev\": %.6f},\n",
        a->ttft_ms_med,
        a->ttft_ms_lo,
        a->ttft_ms_hi,
        a->ttft_ms_mean,
        a->ttft_ms_sd);
    fprintf(f,
        "      \"prefill_tps\": {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f, \"mean\": %.6f, \"stdev\": %.6f},\n",
        a->prefill_med,
        a->prefill_lo,
        a->prefill_hi,
        a->prefill_mean,
        a->prefill_sd);
    fprintf(f,
        "      \"decode_tps\":  {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f, \"mean\": %.6f, \"stdev\": %.6f},\n",
        a->decode_med,
        a->decode_lo,
        a->decode_hi,
        a->decode_mean,
        a->decode_sd);
    fprintf(f, "      \"gen_tokens\":  {\"median\": %.6f},\n", a->gen_tokens_med);
    fprintf(f, "      \"prompt_tokens\":{\"median\": %.6f}\n", a->prompt_tokens_med);
    fprintf(f, "    }\n");
    fprintf(f, "}\n");
    fclose(f);
    /* keep static-analysis happy */
    (void)json_field_dbl;
}

/* Trim the `-{plugin}-{device}[-c{N}]` tail from a cell_id; falls back to the
 * raw id when the suffix isn't present. Caller frees. */
static char* model_label(const char* cell_id, const char* plugin, const char* device) {
    if (!cell_id) return NULL;
    size_t cl = strlen(cell_id);
    /* Strip a trailing "-c<digits>" if present (bench ctx sweep). */
    size_t end = cl;
    while (end > 0 && cell_id[end - 1] >= '0' && cell_id[end - 1] <= '9') end--;
    if (end >= 2 && end < cl && cell_id[end - 1] == 'c' && cell_id[end - 2] == '-') {
        cl = end - 2;
    }
    size_t pl  = plugin ? strlen(plugin) : 0;
    size_t dl  = device ? strlen(device) : 0;
    size_t suf = 1 + pl + 1 + dl;
    if (pl && dl && cl > suf && cell_id[cl - suf] == '-' && cell_id[cl - suf + 1 + pl] == '-' &&
        strncmp(cell_id + cl - suf + 1, plugin, pl) == 0 && strncmp(cell_id + cl - dl, device, dl) == 0) {
        char* out = (char*)malloc(cl - suf + 1);
        if (!out) return NULL;
        memcpy(out, cell_id, cl - suf);
        out[cl - suf] = '\0';
        return out;
    }
    char* out = (char*)malloc(cl + 1);
    if (!out) return NULL;
    memcpy(out, cell_id, cl);
    out[cl] = '\0';
    return out;
}

static void format_size(int64_t bytes, char* buf, size_t bufsz) {
    if (bytes <= 0) {
        snprintf(buf, bufsz, "-");
        return;
    }
    double b = (double)bytes;
    if (b < 1024.0) {
        snprintf(buf, bufsz, "%lld B", (long long)bytes);
    } else if (b < 1024.0 * 1024.0) {
        snprintf(buf, bufsz, "%.1f KiB", b / 1024.0);
    } else if (b < 1024.0 * 1024.0 * 1024.0) {
        snprintf(buf, bufsz, "%.1f MiB", b / (1024.0 * 1024.0));
    } else {
        snprintf(buf, bufsz, "%.2f GiB", b / (1024.0 * 1024.0 * 1024.0));
    }
}

static void write_md_row(const options_t* o, int32_t ngl, int64_t model_size_bytes, const agg_t* a) {
    /* Write llama-bench-style row. First call writes the header + separator;
     * subsequent calls append a single row. Detect "first call" by checking
     * whether the file currently exists / is non-empty. */
    struct stat st;
    bool        first = (stat(o->output_md, &st) != 0) || st.st_size == 0;

    FILE* f = fopen(o->output_md, "a");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for append\n", o->output_md);
        exit(1);
    }
    if (first) {
        fprintf(f,
            "| Model | Size | Backend | Device | ngl | Test | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |\n"
            "|-------|-----:|---------|--------|----:|------|----------:|----------------:|---------------:|\n");
    }

    char  size_buf[32];
    char  ngl_buf[16];
    char  test_buf[32];
    char* model = model_label(o->cell_id, o->plugin, o->device);
    format_size(model_size_bytes, size_buf, sizeof(size_buf));
    if (strcmp(o->plugin, "qairt") == 0 || ngl <= 0) {
        snprintf(ngl_buf, sizeof(ngl_buf), "-");
    } else {
        snprintf(ngl_buf, sizeof(ngl_buf), "%d", ngl);
    }
    snprintf(
        test_buf, sizeof(test_buf), "pp%lld+tg%lld", (long long)a->prompt_tokens_med, (long long)a->gen_tokens_med);

    fprintf(f,
        "| %s | %s | %s | %s | %s | %s | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f |\n",
        model ? model : (o->cell_id ? o->cell_id : "cell"),
        size_buf,
        o->plugin,
        o->device,
        ngl_buf,
        test_buf,
        a->ttft_ms_med,
        a->ttft_ms_sd,
        a->prefill_med,
        a->prefill_sd,
        a->decode_med,
        a->decode_sd);
    free(model);
    fclose(f);
}

/* ----------------------------- main ----------------------------- */

/* Run one (plugin, device, model) cell using the already-`geniex_init`'d
 * runtime. Returns 0 on success, non-zero on failure. The caller owns
 * `geniex_init` / `geniex_deinit` so multiple cells in matrix mode share
 * one plugin-init pass. */
static int run_one_cell(options_t* o) {
    /* Model-manager branch: column 4 (matrix) or `-m` (single-cell) is a
     * model id. Resolve to local paths (pulling on first use); subsequent
     * cells with the same id hit the cache. mmproj/tokenizer columns from
     * the matrix file are ignored in this branch — the manager owns the
     * full path tuple. */
    if (!looks_like_path(o->model_path)) {
        if (resolve_via_mm(o, o->model_path) != 0) {
            return 1;
        }
    } else if (o->plugin && strcmp(o->plugin, "qairt") == 0 && !o->force_vlm && local_bundle_is_vlm(o->model_path)) {
        /* Local QAIRT VLM bundle (resolve_via_mm skipped for path inputs):
         * force VLM so it doesn't hit the dispatcher's "no LLM factory". */
        fprintf(stderr, "[info] %s is a VLM bundle (metadata.json); forcing VLM run loop\n", o->model_path);
        o->force_vlm = true;
    }

    char* anchored = resolve_local_anchor(o->model_path);
    if (anchored) {
        fprintf(stderr, "[info] resolved model dir to anchor: %s\n", anchored);
        o->model_path = anchored;
    }

    /* Device-alias resolution. ngl_default=-1 is the sentinel `auto.py` uses
     * to distinguish "SDK forced a value" (cpu→0, hybrid→999) from "alias
     * passed through". Treat -1 as "leave n_gpu_layers at its plugin default
     * (0)". */
    geniex_ResolveDeviceInput rin;
    memset(&rin, 0, sizeof(rin));
    rin.plugin_id   = o->plugin;
    rin.mode        = o->device;
    rin.ngl_default = -1;
    geniex_ResolveDeviceOutput rout;
    memset(&rout, 0, sizeof(rout));
    int32_t rc = geniex_resolve_device(&rin, &rout);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr, "ERROR: geniex_resolve_device: %s (%d)\n", geniex_get_error_message((geniex_ErrorCode)rc), rc);
        if (anchored) free(anchored);
        return 1;
    }
    if (rout.warning) {
        fprintf(stderr, "[warn] %s\n", rout.warning);
    }
    const char* device_id = o->device_id ? o->device_id : rout.device_id;
    int32_t     ngl       = (rout.ngl == -1) ? 0 : rout.ngl;
    /* --n-gpu-layers overrides the alias default. The gpu alias resolves
     * device_id but no ngl, so a high --n-gpu-layers is what actually
     * offloads layers to the GPU. */
    if (o->ngl_override >= 0) {
        ngl = o->ngl_override;
    }

    /* The qairt plugin doesn't consume n_gpu_layers or n_ctx; force both to 0
     * to match `_build_model_config()` in bindings/python/geniex/auto.py:179. */
    if (strcmp(o->plugin, "qairt") == 0) {
        ngl      = 0;
        o->n_ctx = 0;
    }

    run_result_t* runs = (run_result_t*)calloc((size_t)o->repeat, sizeof(run_result_t));
    if (!runs) {
        fprintf(stderr, "ERROR: oom\n");
        if (anchored) free(anchored);
        if (rout.device_id) geniex_free(rout.device_id);
        if (rout.warning) geniex_free(rout.warning);
        return 1;
    }

    bool is_vlm = (o->mmproj_path != NULL) || o->force_vlm;
    if (is_vlm) {
        run_vlm(o, device_id, ngl, runs);
    } else {
        run_llm(o, device_id, ngl, runs);
    }

    agg_t a;
    aggregate(runs, o->repeat, &a);
    print_summary(o, device_id, ngl, &a);

    int64_t model_size_bytes = compute_model_size(o->model_path);

    if (o->output_json) write_json(o, device_id, ngl, model_size_bytes, runs, &a);
    if (o->output_md) write_md_row(o, ngl, model_size_bytes, &a);

    free(runs);
    if (anchored) free(anchored);
    if (rout.device_id) geniex_free(rout.device_id);
    if (rout.warning) geniex_free(rout.warning);
    if (o->mm_model_path) {
        geniex_free(o->mm_model_path);
        o->mm_model_path = NULL;
    }
    if (o->mm_mmproj) {
        geniex_free(o->mm_mmproj);
        o->mm_mmproj = NULL;
    }
    if (o->mm_tokenizer) {
        geniex_free(o->mm_tokenizer);
        o->mm_tokenizer = NULL;
    }
    return 0;
}

/* Strip trailing CR/LF/whitespace in place and return the input pointer. */
static char* rstrip(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
    return s;
}

/* Run every cell listed in `o->matrix_file` inside one geniex_init / deinit.
 * Per-cell JSON goes to `<o->output_json_dir>/<cell_id>.json` when set.
 * Returns the number of cells that errored (0 = all ok). */
static int run_matrix(options_t* base) {
    FILE* f = fopen(base->matrix_file, "r");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open matrix file %s\n", base->matrix_file);
        return 1;
    }

    int  errors  = 0;
    int  line_no = 0;
    char line[2048];
    char json_path[1024];

    while (fgets(line, sizeof(line), f) != NULL) {
        line_no++;
        rstrip(line);
        if (line[0] == '\0' || line[0] == '#') continue;

        /* Tab-separated: cell_id <TAB> plugin <TAB> device <TAB> model_path
         *                [<TAB> tokenizer_path] [<TAB> mmproj_path]
         *                [<TAB> image_paths (comma-separated)] [<TAB> vlm] */
        char* fields[8] = {NULL};
        int   nf        = 0;
        char* p         = line;
        fields[nf++]    = p;
        while (*p && nf < 8) {
            if (*p == '\t') {
                *p           = '\0';
                fields[nf++] = p + 1;
            }
            p++;
        }
        if (nf < 4) {
            fprintf(stderr, "ERROR: matrix line %d: need at least 4 tab-separated fields, got %d\n", line_no, nf);
            errors++;
            continue;
        }

        /* Build a per-cell options copy from `base`. */
        options_t cell      = *base;
        cell.cell_id        = fields[0];
        cell.plugin         = fields[1];
        cell.device         = fields[2];
        cell.model_path     = fields[3];
        cell.tokenizer_path = (nf >= 5 && fields[4][0] != '\0') ? fields[4] : NULL;
        cell.mmproj_path    = (nf >= 6 && fields[5][0] != '\0') ? fields[5] : NULL;
        cell.output_md      = NULL;
        /* image_paths and force_vlm come per-row from fields[6]/[7], overwriting
         * the values copied from `base` so a global --image / --vlm can't leak
         * into every cell. No audio column: keep it explicitly zeroed. */
        cell.image_count = 0;
        cell.audio_count = 0;
        if (nf >= 7 && fields[6][0] != '\0') {
            char* tok = fields[6];
            while (tok && cell.image_count < MAX_PATHS) {
                char* comma = strchr(tok, ',');
                if (comma) *comma = '\0';
                cell.image_paths[cell.image_count++] = tok;
                tok                                  = comma ? comma + 1 : NULL;
            }
        }
        cell.force_vlm = (nf >= 8 && fields[7][0] != '\0');

        if (base->output_json_dir) {
            snprintf(json_path, sizeof(json_path), "%s/%s.json", base->output_json_dir, cell.cell_id);
            cell.output_json = json_path;
        } else {
            cell.output_json = NULL;
        }

        fprintf(stdout, "[run ] %s\n", cell.cell_id);
        fflush(stdout);
        if (run_one_cell(&cell) != 0) {
            errors++;
        }
    }
    fclose(f);
    return errors;
}

int main(int argc, char** argv) {
    options_t o;
    parse_args(argc, argv, &o);

    check(geniex_init(), "geniex_init");

    int rc;
    if (o.matrix_file) {
        rc = run_matrix(&o);
    } else {
        rc = run_one_cell(&o);
    }

    if (o.prompt_buf) free(o.prompt_buf);
    if (g_mm_inited) {
        /* Best-effort: release the model-manager runtime before geniex_deinit.
         * Failure here is non-fatal — we already produced the JSON the
         * caller cares about. */
        geniex_model_deinit();
        g_mm_inited = false;
    }
    check(geniex_deinit(), "geniex_deinit");
    return rc == 0 ? 0 : 1;
}
