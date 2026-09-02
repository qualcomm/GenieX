// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/*
 * Shared types and cross-module declarations for geniex-bench.
 *
 * Module layout:
 *   benchmark.c — main(), single-cell driver, matrix-file driver
 *   options.c   — usage text + argv parsing
 *   resolve.c   — model-manager ids, local VLM bundle probing, path anchoring
 *   run.c       — LLM / VLM / raw-logits run loops
 *   stats.c     — median / min / max / mean / stdev aggregation
 *   report.c    — stdout summary line, JSON + Markdown reports
 *   util.c      — error exits, file + string helpers, on-disk model size
 */

#ifndef GENIEX_BENCH_H
#define GENIEX_BENCH_H

#include <geniex.h>
#include <geniex_model.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

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

#define MAX_PATHS 16

typedef struct {
    const char* plugin;
    const char* device;
    const char* device_id;
    /* A filesystem path (model dir or .gguf) or a model-manager id
     * `org/repo[:quant]`; resolve_model_id() handles the latter.
     *
     * Shadowing rule: mm.model_path and mm.tokenizer always win over these,
     * but mm.mmproj wins only when VLM was explicitly requested (--vlm or
     * matrix col 8). Otherwise a passively-present mmproj in a bundle would
     * redirect a `-p N` LLM bench into the VLM run loop (#1090). */
    const char* model_path;
    const char* tokenizer_path;
    const char* mmproj_path;
    bool        force_vlm; /* run VLM path even without an mmproj (QAIRT bundles) */
    /* Written by resolve.c, never by argv: what the model manager resolved
     * this cell's id to. The four paths are heap-owned; free_mm_paths()
     * releases them at the end of run_one_cell. */
    struct {
        char* model_path;
        char* mmproj;
        char* tokenizer;
        char* draft_model;
        bool  is_vlm; /* manager classified it as VLM (geniex_ModelType) */
    } mm;
    const char* image_paths[MAX_PATHS];
    int32_t     image_count;
    const char* audio_paths[MAX_PATHS];
    int32_t     audio_count;

    int32_t n_prompt;   /* LLM random-ids prefill length (llama-bench -p), used when prompt_buf is NULL */
    char*   prompt_buf; /* heap-owned raw --prompt-file text; NULL = random-ids */
    /* prompt_buf split once at parse time on lines that are exactly "---";
     * entries point into prompt_buf, which is NUL-terminated in place. A
     * single NULL entry means random-ids. Splitting here rather than per cell
     * keeps it out of the run loops, so every cell of a matrix sees the same
     * list instead of the leftovers of the previous cell's tokenisation.
     * Multiple segments are accepted only under --accuracy. */
    const char** prompts;
    int32_t      prompt_count;
    int32_t      max_new_tokens;
    float        temperature;
    int32_t      seed;
    int32_t      warmup;
    int32_t      repeat;
    bool         reset_between_runs; /* true => geniex_llm_reset() before each run, freeing KV */
    bool         accuracy;           /* true => single run (warmup=0, repeat=1), print generated text */
    /* --accuracy --prompt-file only: run each prompt through the bundle's own
     * chat template (geniex_llm_apply_chat_template) before generation, so the
     * benchmark exercises the same templating production inference uses
     * instead of feeding the file verbatim. */
    const char* system_prompt;   /* --system-prompt; NULL = no system message */
    bool        enable_thinking; /* --think / --no-think; default true, matches `geniex infer` */

    /* Prefill-only raw-logits mode (--logits): one forward pass over the prompt,
     * no decode loop. Bypasses the timing/warmup/repeat machinery entirely. */
    bool    logits_mode;
    bool    logits_last_only;        /* default: every position; set to emit only the last token's row */
    int32_t logits_top_n;            /* per row, emit only the top-N (token_id, logit) pairs */
    int32_t token_callback_delay_us; /* per-token busy-wait in on_token; 0 = no-op */
    int32_t n_ctx;
    int32_t n_threads;
    int32_t ngl_override; /* -1 = use resolved alias default; >=0 overrides */

    const char* spec_type;    /* speculative type(s), comma-separated (llama_cpp); NULL = disabled */
    const char* draft_model;  /* draft/MTP GGUF for draft-* spec types (NULL for ngram-*) */
    int32_t     draft_tokens; /* max draft tokens per step (0 = plugin default) */
    int32_t     draft_min;    /* min draft tokens per step (0 = llama.cpp default) */
    float       draft_p_min;  /* min greedy draft probability (0 = llama.cpp default) */

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

/* One measured run. Every failure path exits the process, so there is no
 * per-run error state to carry. */
typedef struct {
    int32_t     run_idx;
    int64_t     ttft_us;
    int64_t     media_us; /* image/audio encoder time; 0 for text-only */
    int64_t     prompt_time_us;
    int64_t     decode_time_us;
    int64_t     prompt_tokens; /* text + media tokens */
    int64_t     gen_tokens;
    double      prefill_tps;
    double      decode_tps;
    const char* stop_reason; /* not freed; lifetime tied to SDK output */
} run_result_t;

/* One metric reduced over the measured runs. n=1 yields sd=0 (sample stdev
 * with one observation is undefined; 0 matches llama-bench's `123.4 ± 0.0`). */
typedef struct {
    double med, lo, hi, mean, sd;
} stat_t;

typedef struct {
    stat_t ttft_ms;
    stat_t prefill_tps;
    stat_t decode_tps;
    /* Token counts and media time are reported as a median only. */
    double gen_tokens_med;
    double prompt_tokens_med;
    double media_ms_med;
} agg_t;

/* What geniex_resolve_device settled on for this cell, after the --device-id
 * and --n-gpu-layers overrides. Produced once in run_one_cell and consumed
 * together by every run loop and report writer. */
typedef struct {
    const char* id; /* NULL lets the plugin pick */
    int32_t     ngl;
} device_t;

/* Report label for a cell: the --cell-id / matrix col 1 value, or "cell". */
static inline const char* cell_name(const options_t* o) { return o->cell_id ? o->cell_id : "cell"; }

/* ------------------------------- util.c ------------------------------- */

/* Print `what` plus the SDK message for `code` to stderr and exit(1). */
void die(int32_t code, const char* what);
/* die() unless `code` is GENIEX_SUCCESS. */
void check(int32_t code, const char* what);
/* Strips trailing CR/LF/space/tab in place; returns `s`. */
char* rstrip(char* s);
/* Load a whole file into a heap buffer (caller frees); exits on any failure. */
char* slurp(const char* path);
/* st_size for a .gguf, recursive sum of regular children for a bundle dir,
 * 0 on stat failure. */
int64_t model_disk_bytes(const char* path);

/* ------------------------------ options.c ----------------------------- */

/* Parse argv into `o`, applying defaults and mode-specific validation.
 * Exits with a usage message on bad input. */
void parse_args(int argc, char** argv, options_t* o);

/* ------------------------------ resolve.c ----------------------------- */

/* Distinguishes a filesystem path from a model-manager id. */
bool looks_like_path(const char* s);
/* Reads metadata.json's genie.supports_vision; false on any read/parse miss. */
bool local_bundle_is_vlm(const char* model_path);
/* If `path` is a directory, return a heap path to a regular file inside it
 * (the SDK derives the model dir via parent_path()). NULL for a plain file. */
char* resolve_local_anchor(const char* path);
/* Fills o->mm and repoints o->model_path / mmproj_path / tokenizer_path at
 * it, pulling on cache miss. 0 on success. */
int resolve_model_id(options_t* o, const char* id_in);
/* Same for o->draft_model when it is an id rather than a path. */
int resolve_draft_id(options_t* o);
/* Frees o->mm's four paths and NULLs them, so calling it twice is safe. */
void free_mm_paths(options_t* o);
/* Best-effort geniex_model_deinit() when the manager was initialised. */
void mm_shutdown(void);

/* -------------------------------- run.c ------------------------------- */

/* warmup + repeat generation runs per prompt; both write exactly
 * `o->repeat * o->prompt_count` measured results into `out`. */
void run_llm(const options_t* o, const device_t* dev, run_result_t* out);
void run_vlm(const options_t* o, const device_t* dev, run_result_t* out);
/* --logits: one prefill-only forward pass, writes its own JSON report.
 * 0 on success, 1 when that report could not be written. */
int run_logits(const options_t* o, const device_t* dev);

/* ------------------------------- stats.c ------------------------------ */

void aggregate_runs(const run_result_t* runs, int32_t n, agg_t* a);

/* ------------------------------ report.c ----------------------------- */

void print_summary(const options_t* o, const device_t* dev, const agg_t* a);
/* The three writers return 0 on success and 1 when the destination could not
 * be opened. An unwritable report is a cell-level failure: matrix mode counts
 * it and moves to the next cell rather than losing the rest of the sweep. */
int write_cell_json(const options_t* o, const device_t* dev, int64_t model_size_bytes, const run_result_t* runs,
    int32_t n_runs, const agg_t* a);
int write_md_row(const options_t* o, const device_t* dev, int64_t model_size_bytes, const agg_t* a);
int write_logits_json(const options_t* o, const device_t* dev, const geniex_LlmForwardLogitsInput* fin,
    const geniex_LlmForwardLogitsOutput* fout);

#endif /* GENIEX_BENCH_H */
