/*
 * Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * geniex_benchmark — single-cell C inference benchmark, public-API only.
 *
 * Mirrors the Python harness in tests/benchmark/_runner.py:
 *   - same default prompt, max_new_tokens=128, temperature=0.0, seed=42
 *   - 1 warmup + 3 measured runs (configurable)
 *   - llama_cpp prompts get a [warmup=i] / [run=i] suffix to bust KV cache
 *     between runs, matching _runner.py:82-83
 *   - per-cell aggregation: median / min / max for ttft_ms, prefill_tps,
 *     decode_tps; median-only for token counts
 *
 * The binary takes raw filesystem paths to a model + tokenizer + (optional)
 * mmproj. Resolving HuggingFace / AI Hub aliases is done by the Python CLI
 * (`geniex-py pull ...`) before this binary is invoked; the C side does
 * not depend on the model-manager surface.
 */

#include <geniex.h>
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

/* Default prompt mirrors tests/benchmark/_runner.py:28-31 verbatim. */
static const char* const DEFAULT_PROMPT =
    "Explain in three short sentences why the speed of light is the same "
    "in every inertial reference frame, and what that implies for time "
    "dilation. Keep it accessible to a curious teenager.";

#define MAX_PATHS 16

typedef struct {
    const char* plugin;
    const char* device;
    const char* device_id;
    const char* model_path;
    const char* tokenizer_path;
    const char* mmproj_path;
    bool        force_vlm; /* run VLM path even without an mmproj (QAIRT bundles) */
    const char* image_paths[MAX_PATHS];
    int32_t     image_count;
    const char* audio_paths[MAX_PATHS];
    int32_t     audio_count;

    int32_t     max_new_tokens;
    float       temperature;
    int32_t     seed;
    int32_t     warmup;
    int32_t     repeat;
    const char* prompt;
    char*       prompt_buf; /* allocated when --prompt-file is used */
    int32_t     n_ctx;
    int32_t     n_threads;
    int32_t     ngl_override; /* -1 = use resolved alias default; >=0 overrides */

    const char* output_json;
    const char* output_md;
    const char* cell_id;

    /* Matrix mode: one process, one geniex_init, many cells */
    const char* matrix_file;
    const char* output_json_dir;
} options_t;

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
        "                              --model-path <path> [options]\n"
        "\n"
        "  Matrix (one process, shared geniex_init/deinit):\n"
        "    %s --matrix-file <path> [--output-json-dir <dir>] [shared options]\n"
        "\n"
        "Required (single-cell mode):\n"
        "  --plugin            llama_cpp | qairt\n"
        "  --device            cpu | gpu | npu | hybrid | auto (default auto)\n"
        "  --model-path        path to .gguf or qairt bundle dir\n"
        "\n"
        "Required (matrix mode):\n"
        "  --matrix-file PATH  one cell per line, tab-separated:\n"
        "                      cell_id<TAB>plugin<TAB>device<TAB>model_path"
        "[<TAB>tokenizer_path][<TAB>mmproj_path]\n"
        "                      lines starting with '#' are ignored\n"
        "\n"
        "Optional:\n"
        "  --tokenizer-path    explicit tokenizer file\n"
        "  --mmproj-path       multimodal projector — switches to VLM mode\n"
        "  --vlm               force VLM mode without an mmproj (QAIRT bundles)\n"
        "  --image PATH        image input (VLM); may be passed multiple times\n"
        "  --audio PATH        audio input (VLM); may be passed multiple times\n"
        "  --device-id ID      override resolved device id (e.g. HTP0, GPUOpenCL)\n"
        "  --max-new-tokens N  default 128\n"
        "  --temperature F     default 0.0\n"
        "  --seed N            default 42\n"
        "  --warmup N          default 1\n"
        "  --repeat N          default 3 (measured runs)\n"
        "  --prompt TEXT       inline prompt; default mirrors Python BENCH_PROMPT\n"
        "  --prompt-file PATH  read prompt from file\n"
        "  --n-ctx N           model n_ctx (0 = from model, default 0)\n"
        "  --n-threads N       generation threads (0 = SDK default)\n"
        "  --ngl N             llama_cpp layers to offload; overrides the device\n"
        "                      alias default (needed for a real gpu run)\n"
        "  --output-json PATH  (single-cell) write per-cell JSON report\n"
        "  --output-md   PATH  (single-cell) write per-cell Markdown row\n"
        "  --output-json-dir DIR  (matrix) write <DIR>/<cell_id>.json per cell\n"
        "  --cell-id ID        (single-cell) cell label used in reports\n"
        "  --help / -h\n",
        argv0,
        argv0);
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

/* Load whole file into a heap buffer (caller frees). */
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
    o->plugin          = NULL;
    o->device          = "auto";
    o->device_id       = NULL;
    o->model_path      = NULL;
    o->tokenizer_path  = NULL;
    o->mmproj_path     = NULL;
    o->force_vlm       = false;
    o->image_count     = 0;
    o->audio_count     = 0;
    o->max_new_tokens  = 128;
    o->temperature     = 0.0f;
    o->seed            = 42;
    o->warmup          = 1;
    o->repeat          = 3;
    o->prompt          = DEFAULT_PROMPT;
    o->prompt_buf      = NULL;
    o->n_ctx           = 0;
    o->n_threads       = 0;
    o->ngl_override    = -1;
    o->output_json     = NULL;
    o->output_md       = NULL;
    o->cell_id         = NULL;
    o->matrix_file     = NULL;
    o->output_json_dir = NULL;

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
        } else if (strcmp(a, "--model-path") == 0) {
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
        } else if (strcmp(a, "--max-new-tokens") == 0) {
            o->max_new_tokens = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--temperature") == 0) {
            o->temperature = (float)atof(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--seed") == 0) {
            o->seed = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--warmup") == 0) {
            o->warmup = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--repeat") == 0) {
            o->repeat = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--prompt") == 0) {
            o->prompt = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--prompt-file") == 0) {
            o->prompt_buf = slurp(arg_value(argc, argv, &i, a));
            o->prompt     = o->prompt_buf;
        } else if (strcmp(a, "--n-ctx") == 0) {
            o->n_ctx = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--n-threads") == 0) {
            o->n_threads = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--ngl") == 0) {
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
        } else if (strcmp(a, "--token-callback-delay-us") == 0) {
            g_token_callback_delay_us = atoi(arg_value(argc, argv, &i, a));
        } else {
            fprintf(stderr, "ERROR: unknown arg %s\n", a);
            usage(argv[0]);
            exit(2);
        }
    }

    if (o->matrix_file) {
        /* In matrix mode, --plugin/--device/--model-path come from each line of
         * the matrix file, not from argv. */
        if (o->repeat < 1) {
            fprintf(stderr, "ERROR: --repeat must be >=1\n");
            exit(2);
        }
        return;
    }

    if (!o->plugin) {
        fprintf(stderr, "ERROR: --plugin is required\n");
        exit(2);
    }
    if (!o->model_path) {
        fprintf(stderr, "ERROR: --model-path is required\n");
        exit(2);
    }
    if (o->repeat < 1) {
        fprintf(stderr, "ERROR: --repeat must be >=1\n");
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

/* Build the per-run prompt the same way Python does (busts KV cache). */
static char* build_run_prompt(const char* base_prompt, int idx, bool warmup) {
    size_t cap = strlen(base_prompt) + 32;
    char*  out = (char*)malloc(cap);
    if (!out) {
        fprintf(stderr, "ERROR: oom\n");
        exit(1);
    }
    snprintf(out, cap, "%s\n[%s=%d]", base_prompt, warmup ? "warmup" : "run", idx);
    return out;
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

    geniex_SamplerConfig    sampler;
    geniex_GenerationConfig gconfig;
    fill_sampler(&sampler, o);
    fill_gen_config(&gconfig, &sampler, o, /*with_media=*/false);

    int32_t total = o->warmup + o->repeat;
    for (int32_t i = 0; i < total; ++i) {
        bool    is_warmup = (i < o->warmup);
        int32_t run_idx   = is_warmup ? i : (i - o->warmup);
        char*   prompt    = build_run_prompt(o->prompt, run_idx, is_warmup);

        geniex_LlmGenerateInput  gin;
        geniex_LlmGenerateOutput gout;
        memset(&gin, 0, sizeof(gin));
        memset(&gout, 0, sizeof(gout));
        gin.prompt_utf8 = prompt;
        gin.config      = &gconfig;
        gin.on_token    = on_token;

        int32_t rc = geniex_llm_generate(llm, &gin, &gout);
        if (rc != GENIEX_SUCCESS) {
            const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
            fprintf(stderr, "ERROR: geniex_llm_generate run %d failed: %s (%d)\n", run_idx, msg ? msg : "?", rc);
            free(prompt);
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

        if (gout.full_text) {
            geniex_free(gout.full_text);
        }
        free(prompt);
        /* Intentionally NOT calling geniex_llm_reset() between runs — matching
         * tests/benchmark/_runner.py:87-103 which leaves the KV cache intact.
         * The `[warmup=i]` / `[run=i]` suffix on the prompt is what differs
         * between runs and forces llama.cpp to actually run prefill instead
         * of short-circuiting on identical input (which would make
         * prompt_time = 0 and prefill_speed = +inf). */
    }

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
        char*   base      = build_run_prompt(o->prompt, run_idx, is_warmup);
        /* VLM generate() takes a fully-templated prompt; run base text + media
         * through the bundle's chat template so the image tokens land right. */
        char* prompt = build_vlm_prompt(vlm, o, base);
        free(base);
        if (!prompt) {
            geniex_vlm_destroy(vlm);
            exit(1);
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
    double ttft_ms_med, ttft_ms_lo, ttft_ms_hi;
    double prefill_med, prefill_lo, prefill_hi;
    double decode_med, decode_lo, decode_hi;
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
    summarize(tmp, n, &a->ttft_ms_med, &a->ttft_ms_lo, &a->ttft_ms_hi);
    for (int i = 0; i < n; ++i) tmp[i] = runs[i].prefill_tps;
    summarize(tmp, n, &a->prefill_med, &a->prefill_lo, &a->prefill_hi);
    for (int i = 0; i < n; ++i) tmp[i] = runs[i].decode_tps;
    summarize(tmp, n, &a->decode_med, &a->decode_lo, &a->decode_hi);
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

static void write_json(
    const options_t* o, const char* device_id, int32_t ngl, const run_result_t* runs, const agg_t* a) {
    FILE* f = fopen(o->output_json, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for write\n", o->output_json);
        exit(1);
    }
    fprintf(f, "{\n");
    json_field_str(f, "schema_version", "1", false);
    json_field_str(f, "cell_id", o->cell_id ? o->cell_id : "cell", false);
    json_field_str(f, "plugin", o->plugin, false);
    json_field_str(f, "device", o->device, false);
    json_field_str(f, "device_id", device_id, false);
    json_field_str(f, "model_path", o->model_path, false);
    fprintf(f, "    \"params\": {\n");
    fprintf(f,
        "      \"warmup\": %d, \"repeat\": %d, \"max_new_tokens\": %d,\n"
        "      \"temperature\": %.6f, \"seed\": %d, \"n_ctx\": %d, \"n_threads\": %d, \"n_gpu_layers\": %d\n",
        o->warmup,
        o->repeat,
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
        "      \"ttft_ms\":     {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f},\n",
        a->ttft_ms_med,
        a->ttft_ms_lo,
        a->ttft_ms_hi);
    fprintf(f,
        "      \"prefill_tps\": {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f},\n",
        a->prefill_med,
        a->prefill_lo,
        a->prefill_hi);
    fprintf(f,
        "      \"decode_tps\":  {\"median\": %.6f, \"min\": %.6f, \"max\": %.6f},\n",
        a->decode_med,
        a->decode_lo,
        a->decode_hi);
    fprintf(f, "      \"gen_tokens\":  {\"median\": %.6f},\n", a->gen_tokens_med);
    fprintf(f, "      \"prompt_tokens\":{\"median\": %.6f}\n", a->prompt_tokens_med);
    fprintf(f, "    }\n");
    fprintf(f, "}\n");
    fclose(f);
    /* keep static-analysis happy */
    (void)json_field_dbl;
    (void)json_field_i64;
}

static void write_md_row(const options_t* o, const agg_t* a) {
    /* Append a single Markdown row to the file. */
    FILE* f = fopen(o->output_md, "a");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for append\n", o->output_md);
        exit(1);
    }
    fprintf(f,
        "| %s | %s | %s | ok | %.1f | %.1f | %.1f | %.0f |\n",
        o->cell_id ? o->cell_id : "cell",
        o->plugin,
        o->device,
        a->ttft_ms_med,
        a->prefill_med,
        a->decode_med,
        a->gen_tokens_med);
    fclose(f);
}

/* ----------------------------- main ----------------------------- */

/* Run one (plugin, device, model) cell using the already-`geniex_init`'d
 * runtime. Returns 0 on success, non-zero on failure. The caller owns
 * `geniex_init` / `geniex_deinit` so multiple cells in matrix mode share
 * one plugin-init pass. */
static int run_one_cell(options_t* o) {
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
    /* --ngl overrides the alias default. The gpu alias resolves device_id but
     * no ngl, so a high --ngl is what actually offloads layers to the GPU. */
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

    if (o->output_json) write_json(o, device_id, ngl, runs, &a);
    if (o->output_md) write_md_row(o, &a);

    free(runs);
    if (anchored) free(anchored);
    if (rout.device_id) geniex_free(rout.device_id);
    if (rout.warning) geniex_free(rout.warning);
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
         *                [<TAB> tokenizer_path] [<TAB> mmproj_path] */
        char* fields[6] = {NULL};
        int   nf        = 0;
        char* p         = line;
        fields[nf++]    = p;
        while (*p && nf < 6) {
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
    check(geniex_deinit(), "geniex_deinit");
    return rc == 0 ? 0 : 1;
}
