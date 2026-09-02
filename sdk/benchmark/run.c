// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/* The three run loops: LLM timing, VLM timing, and the prefill-only
 * --logits forward pass. */

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bench.h"

/* QAIRT prefill chunk size: the engine pads input_ids up to the next multiple
 * of this before running prefill, so the real prefill work (and thus the honest
 * tokens/sec) is over the padded length, not the raw prompt length. llama_cpp
 * has no such padding and is left untouched. See qcom-ai-hub/geniex#1194. */
#define QAIRT_PREFILL_CHUNK 128

/* Throughput benchmarking only needs a representative fixed text. Used when
 * --prompt-file is absent; build_vlm_prompt() runs it through the chat
 * template so the image tokens land in the right place. */
static const char* const VLM_DEFAULT_PROMPT = "Describe the image.";

/* ------------------------- token callback ------------------------- */

static void busy_wait_us(int32_t us) {
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

/* A no-op unless --token-callback-delay-us inflates each call, which
 * simulates the per-token cost a Python binding pays for its ctypes wrapper
 * plus GIL acquire/release. */
static bool on_token(const char* token, void* user_data) {
    (void)token;
    busy_wait_us(((const options_t*)user_data)->token_callback_delay_us);
    return true;
}

/* --------------------------- SDK config fill --------------------------- */

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
    c->n_ctx            = o->n_ctx;
    c->n_threads        = o->n_threads;
    c->n_gpu_layers     = ngl;
    c->spec_type        = o->spec_type;   /* may be NULL */
    c->spec_draft_model = o->draft_model; /* may be NULL */
    c->spec_n_max       = o->draft_tokens;
    c->spec_n_min       = o->draft_min;
    c->spec_p_min       = o->draft_p_min;
}

/* Random-ids prefill (mirrors llama-bench test_prompt): query vocab + BOS via
 * geniex_llm_get_model_info, fill `o->n_prompt` positions with
 * rand() % vocab_size, then overwrite pos 0 with BOS when the model wants one.
 * `pp` is therefore exactly n_prompt.
 *
 * Destroys `llm` and exits when the plugin has no model info; `hint` names the
 * way out for the calling mode. Caller frees the returned array. */
static int32_t* make_random_tokens(geniex_LLM* llm, const options_t* o, const char* hint) {
    geniex_LlmModelInfo info;
    int32_t             rc_info = geniex_llm_get_model_info(llm, &info);
    if (rc_info != GENIEX_SUCCESS || info.vocab_size <= 0) {
        const char* msg = geniex_get_error_message((geniex_ErrorCode)rc_info);
        fprintf(stderr,
            "ERROR: %s plugin does not support random-ids prefill "
            "(geniex_llm_get_model_info: %s, code=%d). %s\n",
            o->plugin,
            msg ? msg : "?",
            rc_info,
            hint);
        geniex_llm_destroy(llm);
        exit(1);
    }

    int32_t* tokens = (int32_t*)malloc((size_t)o->n_prompt * sizeof(int32_t));
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
    return tokens;
}

/* --------------------------- result recording -------------------------- */

/* Adjust the reported prefill metrics for the engine's real prefill work.
 * QAIRT pads the prompt to a QAIRT_PREFILL_CHUNK multiple before prefill, so
 * prompt_tokens/prefill_tps should reflect that padded length (#1194). Padding
 * the full count is correct: prompt_tokens already includes the media tokens,
 * which are prefilled through the same chunked path. llama_cpp does no such
 * padding, so its metrics are left as the SDK reported them. */
static void normalize_prefill_metrics(run_result_t* r, const char* plugin) {
    if (!plugin || strcmp(plugin, "qairt") != 0 || r->prompt_tokens <= 0) return;
    int64_t padded   = ((r->prompt_tokens + QAIRT_PREFILL_CHUNK - 1) / QAIRT_PREFILL_CHUNK) * QAIRT_PREFILL_CHUNK;
    r->prompt_tokens = padded;
    r->prefill_tps   = r->prompt_time_us > 0 ? (double)padded / ((double)r->prompt_time_us / 1e6) : 0.0;
}

/* Copy one generate() call's profile data into the results array. */
static void record_run(run_result_t* r, int32_t run_idx, const geniex_ProfileData* p, const char* plugin) {
    memset(r, 0, sizeof(*r));
    r->run_idx        = run_idx;
    r->ttft_us        = p->ttft;
    r->media_us       = p->media_time;
    r->prompt_time_us = p->prompt_time;
    r->decode_time_us = p->decode_time;
    r->prompt_tokens  = p->prompt_tokens;
    r->gen_tokens     = p->generated_tokens;
    r->prefill_tps    = p->prefill_speed;
    r->decode_tps     = p->decoding_speed;
    r->stop_reason    = p->stop_reason;
    normalize_prefill_metrics(r, plugin);
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

/* --accuracy --prompt-file: run user_prompt (optionally preceded by
 * --system-prompt) through the bundle's own chat template before
 * generation, so the benchmark exercises the same templating `geniex infer`
 * uses instead of feeding the file verbatim. Returns heap text the caller
 * frees with geniex_free, or NULL on failure. */
static char* build_llm_accuracy_prompt(geniex_LLM* llm, const options_t* o, const char* user_prompt) {
    /* Zero first: the plugins dereference the optional tool-calling fields
     * whenever they're non-NULL, so stack garbage there is an access violation. */
    geniex_LlmChatMessage messages[2];
    memset(messages, 0, sizeof(messages));
    int32_t nm = 0;
    if (o->system_prompt) {
        messages[nm].role    = "system";
        messages[nm].content = o->system_prompt;
        nm++;
    }
    messages[nm].role    = "user";
    messages[nm].content = user_prompt;
    nm++;

    geniex_LlmApplyChatTemplateInput  tin;
    geniex_LlmApplyChatTemplateOutput tout;
    memset(&tin, 0, sizeof(tin));
    memset(&tout, 0, sizeof(tout));
    tin.messages              = messages;
    tin.message_count         = nm;
    tin.enable_thinking       = o->enable_thinking;
    tin.add_generation_prompt = true;

    int32_t rc = geniex_llm_apply_chat_template(llm, &tin, &tout);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr,
            "ERROR: geniex_llm_apply_chat_template: %s (%d)\n",
            geniex_get_error_message((geniex_ErrorCode)rc),
            rc);
        return NULL;
    }
    return tout.formatted_text;
}

void run_llm(const options_t* o, const device_t* dev, run_result_t* out) {
    geniex_LlmCreateInput cin;
    memset(&cin, 0, sizeof(cin));
    cin.model_path     = o->model_path;
    cin.tokenizer_path = o->tokenizer_path; /* may be NULL */
    cin.plugin_id      = o->plugin;
    cin.device_id      = dev->id; /* may be NULL */
    fill_model_config(&cin.config, o, dev->ngl);

    geniex_LLM* llm = NULL;
    check(geniex_llm_create(&cin, &llm), "geniex_llm_create");

    /* --prompt-file feeds prompt_utf8 verbatim, so `pp` is the tokenizer's
     * count rather than n_prompt — the only mode plugins that reject
     * input_ids (today: qairt) can run. Without it, random ids. */
    int32_t* tokens = NULL;
    if (!o->prompt_buf) {
        tokens = make_random_tokens(llm,
            o,
            "Pass --prompt-file PATH to use text-prompt mode instead, "
            "or see https://github.com/qualcomm/GenieX/issues/1008.");
    }

    geniex_SamplerConfig    sampler;
    geniex_GenerationConfig gconfig;
    fill_sampler(&sampler, o);
    fill_gen_config(&gconfig, &sampler, o, /*with_media=*/false);

    int32_t total   = o->warmup + o->repeat;
    int32_t out_idx = 0; /* results are written across all prompts, not per prompt */
    for (int32_t pi = 0; pi < o->prompt_count; ++pi) {
        const char* cur_prompt = o->prompts[pi];
        if (o->prompt_count > 1) {
            fprintf(stdout, "[sep ] prompt %d/%d\n", pi + 1, o->prompt_count);
        }
        for (int32_t i = 0; i < total; ++i) {
            bool    is_warmup = (i < o->warmup);
            int32_t run_idx   = is_warmup ? i : (i - o->warmup);

            /* Reset before each run (llama-bench semantics) OR always at the
             * start of a new prompt segment, so batched prompts never inherit
             * the previous segment's KV cache even under
             * --no-reset-between-runs. */
            if (o->reset_between_runs || (o->prompt_count > 1 && i == 0)) {
                check(geniex_llm_reset(llm), "geniex_llm_reset");
            }

            geniex_LlmGenerateInput  gin;
            geniex_LlmGenerateOutput gout;
            memset(&gin, 0, sizeof(gin));
            memset(&gout, 0, sizeof(gout));
            char* templated_prompt = NULL;
            if (cur_prompt) {
                if (o->accuracy) {
                    templated_prompt = build_llm_accuracy_prompt(llm, o, cur_prompt);
                    if (!templated_prompt) {
                        free(tokens);
                        geniex_llm_destroy(llm);
                        exit(1);
                    }
                    gin.prompt_utf8 = templated_prompt;
                } else {
                    gin.prompt_utf8 = cur_prompt;
                }
            } else {
                gin.input_ids       = tokens;
                gin.input_ids_count = o->n_prompt;
            }
            gin.config    = &gconfig;
            gin.on_token  = on_token;
            gin.user_data = (void*)o;

            int32_t rc = geniex_llm_generate(llm, &gin, &gout);
            if (rc != GENIEX_SUCCESS) {
                const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
                fprintf(stderr, "ERROR: geniex_llm_generate run %d failed: %s (%d)\n", run_idx, msg ? msg : "?", rc);
                if (templated_prompt) geniex_free(templated_prompt);
                free(tokens);
                geniex_llm_destroy(llm);
                exit(1);
            }

            if (!is_warmup) {
                record_run(&out[out_idx], out_idx, &gout.profile_data, o->plugin);
                out_idx++;
            }

            if (!is_warmup && o->accuracy && gout.full_text) {
                print_gen_text(gout.full_text);
            }
            if (!is_warmup && gout.profile_data.draft_n_total > 0) {
                fprintf(stderr,
                    "[spec] draft acceptance = %.5f (%lld accepted / %lld generated)\n",
                    (double)gout.profile_data.draft_n_accepted / (double)gout.profile_data.draft_n_total,
                    (long long)gout.profile_data.draft_n_accepted,
                    (long long)gout.profile_data.draft_n_total);
            }
            if (gout.full_text) {
                geniex_free(gout.full_text);
            }
            if (templated_prompt) {
                geniex_free(templated_prompt);
            }
        }
    }

    free(tokens);
    check(geniex_llm_destroy(llm), "geniex_llm_destroy");
}

/* ----------------------------- VLM run loop ----------------------------- */

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

void run_vlm(const options_t* o, const device_t* dev, run_result_t* out) {
    geniex_VlmCreateInput cin;
    memset(&cin, 0, sizeof(cin));
    cin.model_path     = o->model_path;
    cin.mmproj_path    = o->mmproj_path;
    cin.tokenizer_path = o->tokenizer_path;
    cin.plugin_id      = o->plugin;
    cin.device_id      = dev->id;
    fill_model_config(&cin.config, o, dev->ngl);

    geniex_VLM* vlm = NULL;
    check(geniex_vlm_create(&cin, &vlm), "geniex_vlm_create");

    geniex_SamplerConfig    sampler;
    geniex_GenerationConfig gconfig;
    fill_sampler(&sampler, o);
    fill_gen_config(&gconfig, &sampler, o, /*with_media=*/true);

    int32_t total   = o->warmup + o->repeat;
    int32_t out_idx = 0; /* results are written across all prompts, not per prompt */
    for (int32_t pi = 0; pi < o->prompt_count; ++pi) {
        const char* cur_prompt = o->prompts[pi];
        if (o->prompt_count > 1) {
            fprintf(stdout, "[sep ] prompt %d/%d\n", pi + 1, o->prompt_count);
        }
        for (int32_t i = 0; i < total; ++i) {
            bool    is_warmup = (i < o->warmup);
            int32_t run_idx   = is_warmup ? i : (i - o->warmup);

            /* Build the templated prompt once per run.  When --prompt-file
             * supplies a pre-templated string, use it directly; otherwise run
             * the fixed default text through the bundle's chat template so the
             * image tokens are placed correctly. */
            char*       built_prompt = NULL;
            const char* final_prompt;
            if (cur_prompt) {
                final_prompt = cur_prompt;
            } else {
                built_prompt = build_vlm_prompt(vlm, o, VLM_DEFAULT_PROMPT);
                if (!built_prompt) {
                    geniex_vlm_destroy(vlm);
                    exit(1);
                }
                final_prompt = built_prompt;
            }

            /* VLM must reset between runs: the image is consumed into the KV
             * cache on the first generate() call, so a second call re-sends an
             * already-processed prompt and generates nothing. */
            if (o->reset_between_runs) {
                check(geniex_vlm_reset(vlm), "geniex_vlm_reset");
            }

            geniex_VlmGenerateInput  gin;
            geniex_VlmGenerateOutput gout;
            memset(&gin, 0, sizeof(gin));
            memset(&gout, 0, sizeof(gout));
            gin.prompt_utf8 = final_prompt;
            gin.config      = &gconfig;
            gin.on_token    = on_token;
            gin.user_data   = (void*)o;

            int32_t rc = geniex_vlm_generate(vlm, &gin, &gout);
            if (rc != GENIEX_SUCCESS) {
                const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
                fprintf(stderr, "ERROR: geniex_vlm_generate run %d failed: %s (%d)\n", run_idx, msg ? msg : "?", rc);
                if (built_prompt) geniex_free(built_prompt);
                geniex_vlm_destroy(vlm);
                exit(1);
            }

            if (!is_warmup) {
                record_run(&out[out_idx], out_idx, &gout.profile_data, o->plugin);
                out_idx++;
            }

            if (!is_warmup && o->accuracy && gout.full_text) {
                print_gen_text(gout.full_text);
            }
            if (gout.full_text) {
                geniex_free(gout.full_text);
            }
            if (built_prompt) geniex_free(built_prompt);
        }
    }

    check(geniex_vlm_destroy(vlm), "geniex_vlm_destroy");
}

/* ---------------------------- logits run ---------------------------- */

/* --logits mode: prefill-only raw logits, no timing. Runs one forward pass over
 * `o->n_prompt` random token ids and writes the top-N (token_id, logit) pairs
 * per row to `o->output_json`. Bypasses run_llm's warmup/repeat/aggregate_runs path.
 * The forward-logits API takes input_ids only, so this is random-ids input
 * (bench has no tokenizer). */
int run_logits(const options_t* o, const device_t* dev) {
    geniex_LlmCreateInput cin;
    memset(&cin, 0, sizeof(cin));
    cin.model_path     = o->model_path;
    cin.tokenizer_path = o->tokenizer_path; /* may be NULL */
    cin.plugin_id      = o->plugin;
    cin.device_id      = dev->id; /* may be NULL */
    fill_model_config(&cin.config, o, dev->ngl);

    geniex_LLM* llm = NULL;
    check(geniex_llm_create(&cin, &llm), "geniex_llm_create");

    int32_t* tokens = make_random_tokens(llm, o, "--logits requires it; --prompt-file is not supported with --logits.");

    geniex_LlmForwardLogitsInput  fin;
    geniex_LlmForwardLogitsOutput fout;
    memset(&fin, 0, sizeof(fin));
    memset(&fout, 0, sizeof(fout));
    fin.input_ids       = tokens;
    fin.input_ids_count = o->n_prompt;
    fin.all_positions   = !o->logits_last_only;
    fin.top_n           = o->logits_top_n;

    int32_t rc = geniex_llm_forward_logits(llm, &fin, &fout);
    if (rc != GENIEX_SUCCESS) {
        const char* msg = geniex_get_error_message((geniex_ErrorCode)rc);
        fprintf(stderr, "ERROR: geniex_llm_forward_logits failed: %s (%d)\n", msg ? msg : "?", rc);
        free(tokens);
        geniex_llm_destroy(llm);
        exit(1);
    }

    fprintf(stdout,
        "[ok  ] %s  plugin=%s device=%s ngl=%d logits n_rows=%d vocab=%d top_n=%d%s\n",
        cell_name(o),
        o->plugin,
        o->device,
        dev->ngl,
        fout.n_rows,
        fout.vocab_size,
        fout.row_width,
        fout.row_width < fout.vocab_size ? " (rows truncated to top_n)" : "");

    int rc_report = 0;
    if (o->output_json) {
        rc_report = write_logits_json(o, dev, &fin, &fout);
    } else {
        fprintf(stderr, "[warn] --logits without --output-json: logits computed but not written\n");
    }

    geniex_free(fout.logits);
    geniex_free(fout.token_ids);
    free(tokens);
    geniex_llm_destroy(llm);
    return rc_report;
}
