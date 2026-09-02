// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/* Usage text and argv parsing. */

#include <stdlib.h>
#include <string.h>

#include "bench.h"

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
        "                         Requires the plugin to implement\n"
        "                         geniex_llm_get_model_info; the tool fails with a\n"
        "                         clear error otherwise.\n"
        "  -n, --n-gen N          tokens to generate per run; default 128\n"
        "  -c, --ctx-size N       model n_ctx (0 = from model, default 0)\n"
        "  -t, --threads N        generation threads (0 = SDK default)\n"
        "  -ngl, --n-gpu-layers N llama_cpp layers to offload; overrides the\n"
        "                         device alias default (-1 = all layers)\n"
        "  --spec-type TYPES      speculative type(s), comma-separated: draft-mtp,\n"
        "                         draft-eagle3,draft-simple,ngram-simple,ngram-map-k,\n"
        "                         ngram-map-k4v,ngram-mod,ngram-cache (llama_cpp)\n"
        "  --draft-model PATH     draft/MTP GGUF for draft-* spec types (llama_cpp)\n"
        "  --draft-tokens N       max draft tokens per step (0 = plugin default)\n"
        "  --draft-min N          min draft tokens per step (0 = llama.cpp default)\n"
        "  --draft-p-min F        min greedy draft probability (0 = llama.cpp default)\n"
        "  --warmup N             default 1\n"
        "  --no-warmup            equivalent to --warmup 0\n"
        "  --temperature F        default 0.0\n"
        "  --seed N               default 42; also seeds rand() for prompt ids\n"
        "  --prompt-file PATH     opt out of random-ids prefill: read a UTF-8 prompt\n"
        "                         from PATH and feed it via prompt_utf8 instead. The\n"
        "                         only way to bench plugins that don't implement\n"
        "                         geniex_llm_get_model_info. With this flag, reported\n"
        "                         `pp` is the tokenizer's count, NOT --n-prompt.\n"
        "                         For qairt, `pp` and prefill tok/s are reported over\n"
        "                         the padded length ceil(pp/128)*128, matching the\n"
        "                         engine's 128-token prefill chunking (#1194).\n"
        "                         Under --accuracy only, a line that is exactly `---`\n"
        "                         splits the file into several prompts; each runs on\n"
        "                         its own KV cache, is marked `[sep ] prompt i/n` on\n"
        "                         stdout and gets its own entry in the JSON `runs`\n"
        "                         array. Every other mode feeds the file verbatim as\n"
        "                         one prompt, `---` lines included: the segments differ\n"
        "                         in length, so a tok/s median across them would mix\n"
        "                         populations.\n"
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
        "                         Each --prompt-file segment is run through the\n"
        "                         bundle's own chat template before generation --\n"
        "                         same templating `geniex infer` uses -- so pass the\n"
        "                         raw user turn, not pre-templated text.\n"
        "  --system-prompt TEXT   with --accuracy --prompt-file: system message added\n"
        "                         ahead of the prompt in the chat template\n"
        "  --think / --no-think  with --accuracy --prompt-file: enable_thinking for\n"
        "                         the chat template (default: think)\n"
        "  --logits               prefill-only raw-logits mode: run one forward pass\n"
        "                         (geniex_llm_forward_logits, no decode loop) over N\n"
        "                         random token ids (-p N, like the timing default) and\n"
        "                         write every position's logits row ([n_tokens, vocab])\n"
        "                         to the JSON report, for on-target accuracy metrics\n"
        "                         (perplexity/MMLU/MMMU). Bypasses timing; --warmup/-r/-n\n"
        "                         are ignored. Input is random ids only (the\n"
        "                         forward-logits API takes input_ids, and the bench tool\n"
        "                         has no tokenizer): --prompt-file is rejected with\n"
        "                         --logits.\n"
        "  --logits-last-only     with --logits, emit only the last token's logits row\n"
        "                         instead of every position (next-token / MMLU scoring).\n"
        "  --logits-top-n N       with --logits, emit only the top-N (token_id, logit)\n"
        "                         pairs per row (default 20) to keep the JSON small;\n"
        "                         the report records this truncation.\n"
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

static const char* arg_value(int argc, char** argv, int* i, const char* flag) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ERROR: %s requires a value\n", flag);
        exit(2);
    }
    *i += 1;
    return argv[*i];
}

static void require_min(int32_t value, int32_t min, const char* flag) {
    if (value < min) {
        fprintf(stderr, "ERROR: %s must be >=%d\n", flag, min);
        exit(2);
    }
}

/* Is this line exactly "---", ignoring surrounding whitespace? Read-only on
 * purpose: an rstrip()-based test leaves NULs in prompt_buf and the prompt,
 * read as a C string, ends at the first one. */
static bool is_separator(const char* line) {
    while (*line == ' ' || *line == '\t' || *line == '\r') line++;
    if (line[0] != '-' || line[1] != '-' || line[2] != '-') return false;
    line += 3;
    while (*line == ' ' || *line == '\t' || *line == '\r') line++;
    return *line == '\n' || *line == '\0';
}

/* A one-entry list holding `p` verbatim; `p` may be NULL for random-ids. */
static const char** one_prompt(const char* p, int32_t* n_out) {
    const char** prompts = (const char**)malloc(sizeof(char*));
    if (!prompts) {
        fprintf(stderr, "ERROR: oom\n");
        exit(1);
    }
    prompts[0] = p;
    *n_out     = 1;
    return prompts;
}

/* Append `seg` to the prompt list unless it is NULL or all whitespace, so
 * stray/leading/trailing "---" separators don't produce empty prompts. */
static void append_prompt_if_nonempty(const char** prompts, int32_t* n, char* seg) {
    if (!seg) return;
    const char* s = seg;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    if (*s) prompts[(*n)++] = seg;
}

/* Split --prompt-file's text into the list the run loops iterate over.
 *
 * `prompt_buf` NULL (no --prompt-file) yields a single NULL entry, which the
 * LLM loop reads as "random-ids prefill" and the VLM loop as "template
 * VLM_DEFAULT_PROMPT". Otherwise the buffer is split on lines that are exactly
 * "---", ignoring leading/trailing whitespace on that line: each separator is
 * overwritten with a NUL so the preceding segment ends there.
 *
 * Entries point into `prompt_buf`. Returns NULL, after printing the error,
 * when every segment is empty. */
static const char** build_prompt_list(char* prompt_buf, int32_t* n_out) {
    if (!prompt_buf) return one_prompt(NULL, n_out);

    /* Upper bound: one more segment than newlines. */
    int32_t cap = 1;
    for (char* p = prompt_buf; *p; ++p)
        if (*p == '\n') cap++;
    const char** prompts = (const char**)malloc((size_t)cap * sizeof(char*));
    if (!prompts) {
        fprintf(stderr, "ERROR: oom\n");
        exit(1);
    }
    int32_t n_prompts = 0;

    /* Only separator lines are overwritten, so every segment keeps its text
     * byte for byte. */
    char* seg = prompt_buf; /* start of the current segment, NULL at EOF */
    for (char* line = prompt_buf; line;) {
        char* nl = strchr(line, '\n');
        if (is_separator(line)) {
            /* End the preceding segment. When it spans real lines (seg <
             * line) terminate at the newline before this separator so the
             * trailing "\n" is dropped; an empty segment (seg == line, e.g.
             * a leading or back-to-back separator) collapses to "". */
            if (line > seg)
                line[-1] = '\0';
            else if (seg)
                *seg = '\0';
            append_prompt_if_nonempty(prompts, &n_prompts, seg);
            seg = nl ? nl + 1 : NULL;
        }
        line = nl ? nl + 1 : NULL;
    }
    /* Trailing segment (the whole buffer when there is no "---"). */
    append_prompt_if_nonempty(prompts, &n_prompts, seg);
    if (n_prompts == 0) {
        fprintf(stderr, "ERROR: --prompt-file has no non-empty prompts\n");
        free(prompts);
        return NULL;
    }
    *n_out = n_prompts;
    return prompts;
}

void parse_args(int argc, char** argv, options_t* o) {
    o->plugin                  = NULL;
    o->device                  = "auto";
    o->device_id               = NULL;
    o->model_path              = NULL;
    o->tokenizer_path          = NULL;
    o->mmproj_path             = NULL;
    o->mm.model_path           = NULL;
    o->mm.mmproj               = NULL;
    o->mm.tokenizer            = NULL;
    o->mm.draft_model          = NULL;
    o->force_vlm               = false;
    o->mm.is_vlm               = false;
    o->image_count             = 0;
    o->audio_count             = 0;
    o->n_prompt                = 512;
    o->prompt_buf              = NULL;
    o->max_new_tokens          = 128;
    o->temperature             = 0.0f;
    o->seed                    = 42;
    o->warmup                  = 1;
    o->repeat                  = 5;
    o->reset_between_runs      = true;
    o->accuracy                = false;
    o->system_prompt           = NULL;
    o->enable_thinking         = true;
    o->logits_mode             = false;
    o->logits_last_only        = false;
    o->logits_top_n            = 20;
    o->token_callback_delay_us = 0;
    o->n_ctx                   = 0;
    o->n_threads               = 0;
    o->ngl_override            = -1;
    o->spec_type               = NULL;
    o->draft_model             = NULL;
    o->draft_tokens            = 0;
    o->draft_min               = 0;
    o->draft_p_min             = 0.0f;
    o->output_json             = NULL;
    o->output_md               = NULL;
    o->cell_id                 = NULL;
    o->matrix_file             = NULL;
    o->output_json_dir         = NULL;
    o->mm_data_dir             = NULL;
    o->mm_chipset              = NULL;
    o->mm_hub                  = NULL;

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
        } else if (strcmp(a, "--system-prompt") == 0) {
            o->system_prompt = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--think") == 0) {
            o->enable_thinking = true;
        } else if (strcmp(a, "--no-think") == 0) {
            o->enable_thinking = false;
        } else if (strcmp(a, "--logits") == 0) {
            o->logits_mode = true;
        } else if (strcmp(a, "--logits-last-only") == 0) {
            o->logits_mode      = true;
            o->logits_last_only = true;
        } else if (strcmp(a, "--logits-top-n") == 0) {
            o->logits_top_n = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-c") == 0 || strcmp(a, "--ctx-size") == 0) {
            o->n_ctx = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-t") == 0 || strcmp(a, "--threads") == 0) {
            o->n_threads = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "-ngl") == 0 || strcmp(a, "--n-gpu-layers") == 0) {
            o->ngl_override = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--spec-type") == 0) {
            o->spec_type = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--draft-model") == 0) {
            o->draft_model = arg_value(argc, argv, &i, a);
        } else if (strcmp(a, "--draft-tokens") == 0) {
            o->draft_tokens = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--draft-min") == 0) {
            o->draft_min = atoi(arg_value(argc, argv, &i, a));
        } else if (strcmp(a, "--draft-p-min") == 0) {
            o->draft_p_min = (float)atof(arg_value(argc, argv, &i, a));
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
            o->token_callback_delay_us = atoi(arg_value(argc, argv, &i, a));
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
    } else if (o->system_prompt) {
        fprintf(stderr, "ERROR: --system-prompt requires --accuracy --prompt-file\n");
        exit(2);
    }

    if (o->logits_mode) {
        if (o->prompt_buf) {
            fprintf(stderr, "ERROR: --logits uses random token ids; --prompt-file is not supported with it\n");
            exit(2);
        }
        require_min(o->logits_top_n, 1, "--logits-top-n");
    }

    /* Resolve the prompt list once, here, rather than in each run loop.
     *
     * Splitting on `---` is an --accuracy feature: it exists to eyeball
     * several prompts' output in one invocation. Every other mode feeds the
     * file verbatim as a single prompt — the segments have different lengths,
     * so a median of prefill_tps across them would mix populations. */
    o->prompts =
        o->accuracy ? build_prompt_list(o->prompt_buf, &o->prompt_count) : one_prompt(o->prompt_buf, &o->prompt_count);
    if (!o->prompts) {
        exit(2);
    }

    /* In matrix mode --plugin/--device/--model come from each line of the
     * matrix file, not from argv. Checked before the run-count sanity below so
     * a single-cell invocation still reports the missing required flag first. */
    if (!o->matrix_file) {
        if (!o->plugin) {
            fprintf(stderr, "ERROR: --plugin is required\n");
            exit(2);
        }
        if (!o->model_path) {
            fprintf(stderr, "ERROR: --model is required\n");
            exit(2);
        }
    }

    /* Run-count sanity, checked for both modes. */
    require_min(o->repeat, 1, "--repetitions");
    require_min(o->n_prompt, 1, "--n-prompt");
}
