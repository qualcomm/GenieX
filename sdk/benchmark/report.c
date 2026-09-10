// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/* Output side: the stdout summary line, the per-cell JSON report, the
 * llama-bench-style Markdown row, and the --logits JSON report. */

#include <stdlib.h>
#include <string.h>

#include "bench.h"

void print_summary(const options_t* o, const device_t* dev, const agg_t* a) {
    fprintf(stdout,
        "[ok  ] %s  plugin=%s device=%s%s%s%s ngl=%d "
        "ttft=%.1fms prefill=%.1ftps decode=%.1ftps gen=%.0f tok\n",
        cell_name(o),
        o->plugin,
        o->device,
        dev->id ? "(id=" : "",
        dev->id ? dev->id : "",
        dev->id ? ")" : "",
        dev->ngl,
        a->ttft_ms.med,
        a->prefill_tps.med,
        a->decode_tps.med,
        a->gen_tokens_med);
}

/* ------------------------------- JSON ------------------------------- */

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

static void json_field_str(FILE* f, const char* k, const char* v, bool last) {
    fprintf(f, "    \"%s\": ", k);
    if (v)
        json_write_quoted(f, v);
    else
        fprintf(f, "null");
    fprintf(f, last ? "\n" : ",\n");
}

static void json_field_i64(FILE* f, const char* k, int64_t v, bool last) {
    fprintf(f, "    \"%s\": %lld%s", k, (long long)v, last ? "\n" : ",\n");
}

/* One `"<key>": {"median": ..., "min": ..., ...}` line of the "agg" object.
 * The key is padded to a fixed width so the value objects line up in the
 * written file. */
static void json_agg_stat(FILE* f, const char* key, const stat_t* st) {
    char field[24];
    snprintf(field, sizeof(field), "\"%s\":", key);
    fprintf(f,
        "      %-15s{\"median\": %.6f, \"min\": %.6f, \"max\": %.6f, \"mean\": %.6f, \"stdev\": %.6f},\n",
        field,
        st->med,
        st->lo,
        st->hi,
        st->mean,
        st->sd);
}

int write_cell_json(const options_t* o, const device_t* dev, int64_t model_size_bytes, const run_result_t* runs,
    int32_t n_runs, const agg_t* a) {
    FILE* f = fopen(o->output_json, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for write\n", o->output_json);
        return 1;
    }
    fprintf(f, "{\n");
    json_field_str(f, "schema_version", "5", false);
    json_field_str(f, "cell_id", cell_name(o), false);
    json_field_str(f, "plugin", o->plugin, false);
    json_field_str(f, "device", o->device, false);
    json_field_str(f, "device_id", dev->id, false);
    json_field_str(f, "model_path", o->model_path, false);
    json_field_i64(f, "model_size_bytes", model_size_bytes, false);
    json_field_str(f, "qairt_version", geniex_get_plugin_version("qairt"), false);
    json_field_str(f, "llama_cpp_version", geniex_get_plugin_version("llama_cpp"), false);
    fprintf(f, "    \"params\": {\n");
    fprintf(f,
        "      \"warmup\": %d, \"repetitions\": %d, \"n_prompt\": %d, \"n_gen\": %d,\n"
        "      \"temperature\": %.6f, \"seed\": %d, \"n_ctx\": %d, \"n_threads\": %d, \"n_gpu_layers\": %d",
        o->warmup,
        o->repeat,
        o->n_prompt,
        o->max_new_tokens,
        (double)o->temperature,
        o->seed,
        o->n_ctx,
        o->n_threads,
        dev->ngl);
    if (o->spec_type) {
        fprintf(f, ",\n      \"spec_type\": ");
        json_write_quoted(f, o->spec_type);
        if (o->draft_model) {
            fprintf(f, ",\n      \"draft_model\": ");
            json_write_quoted(f, o->draft_model);
        }
        fprintf(f, ",\n      \"draft_tokens\": %d", o->draft_tokens);
    }
    if (o->power_mode) {
        fprintf(f, ",\n      \"power_mode\": ");
        json_write_quoted(f, o->power_mode);
    }
    /* Which QAIRT runtime produced these numbers -- the reason the override exists. */
    if (o->qairt_lib) {
        fprintf(f, ",\n      \"qairt_lib\": ");
        json_write_quoted(f, o->qairt_lib);
    }
    fprintf(f, "\n    },\n");
    fprintf(f, "    \"runs\": [\n");
    for (int32_t i = 0; i < n_runs; ++i) {
        const run_result_t* r = &runs[i];
        fprintf(f,
            "      {\"run_idx\": %d, \"ttft_us\": %lld, \"media_us\": %lld, "
            "\"prompt_time_us\": %lld, \"decode_time_us\": %lld, "
            "\"prompt_tokens\": %lld, \"gen_tokens\": %lld, "
            "\"prefill_tps\": %.6f, \"decode_tps\": %.6f, \"stop_reason\": %s%s%s}%s\n",
            r->run_idx,
            (long long)r->ttft_us,
            (long long)r->media_us,
            (long long)r->prompt_time_us,
            (long long)r->decode_time_us,
            (long long)r->prompt_tokens,
            (long long)r->gen_tokens,
            r->prefill_tps,
            r->decode_tps,
            r->stop_reason ? "\"" : "null",
            r->stop_reason ? r->stop_reason : "",
            r->stop_reason ? "\"" : "",
            (i + 1 < n_runs) ? "," : "");
    }
    fprintf(f, "    ],\n");
    fprintf(f, "    \"agg\": {\n");
    json_agg_stat(f, "ttft_ms", &a->ttft_ms);
    json_agg_stat(f, "prefill_tps", &a->prefill_tps);
    json_agg_stat(f, "decode_tps", &a->decode_tps);
    fprintf(f, "      \"gen_tokens\":  {\"median\": %.6f},\n", a->gen_tokens_med);
    fprintf(f, "      \"prompt_tokens\":{\"median\": %.6f},\n", a->prompt_tokens_med);
    fprintf(f, "      \"media_ms\":{\"median\": %.6f}\n", a->media_ms_med);
    fprintf(f, "    }\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/* Write one row of the SDK's top-N output as a JSON array of [token_id, logit]
 * pairs. The SDK already selected and sorted the top-N (descending logit), so
 * this just formats row_width pairs. */
static void write_top_n_row(FILE* f, const int32_t* ids, const float* logits, int32_t row_width) {
    fputc('[', f);
    for (int32_t k = 0; k < row_width; ++k) {
        fprintf(f, "%s[%d, %.6f]", k ? ", " : "", ids[k], logits[k]);
    }
    fputc(']', f);
}

/* Write the prefill-only logits report for --logits. Emits shape metadata plus,
 * per row, the top-N [token_id, logit] pairs. Records the top-N truncation
 * explicitly so a consumer never mistakes it for the full vocabulary. */
int write_logits_json(const options_t* o, const device_t* dev, const geniex_LlmForwardLogitsInput* fin,
    const geniex_LlmForwardLogitsOutput* fout) {
    FILE* f = fopen(o->output_json, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for write\n", o->output_json);
        return 1;
    }
    const bool truncated = fout->row_width < fout->vocab_size;

    fprintf(f, "{\n");
    json_field_str(f, "schema_version", "logits-1", false);
    json_field_str(f, "cell_id", cell_name(o), false);
    json_field_str(f, "plugin", o->plugin, false);
    json_field_str(f, "device", o->device, false);
    json_field_str(f, "device_id", dev->id, false);
    json_field_str(f, "model_path", o->model_path, false);
    json_field_i64(f, "n_gpu_layers", dev->ngl, false);
    json_field_i64(f, "n_prompt", fin->input_ids_count, false);
    fprintf(f, "    \"all_positions\": %s,\n", fin->all_positions ? "true" : "false");
    json_field_i64(f, "n_rows", fout->n_rows, false);
    json_field_i64(f, "vocab_size", fout->vocab_size, false);
    json_field_i64(f, "top_n", fout->row_width, false);
    fprintf(f, "    \"truncated_to_top_n\": %s,\n", truncated ? "true" : "false");

    fprintf(f, "    \"rows\": [\n");
    for (int32_t r = 0; r < fout->n_rows; ++r) {
        const size_t off = (size_t)r * fout->row_width;
        fprintf(f, "      ");
        write_top_n_row(f, fout->token_ids + off, fout->logits + off, fout->row_width);
        fprintf(f, r + 1 < fout->n_rows ? ",\n" : "\n");
    }
    fprintf(f, "    ]\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/* ----------------------------- Markdown ----------------------------- */

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

static void format_bytes(int64_t bytes, char* buf, size_t bufsz) {
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

int write_md_row(const options_t* o, const device_t* dev, int64_t model_size_bytes, const agg_t* a) {
    /* Write llama-bench-style row. First call writes the header + separator;
     * subsequent calls append a single row. Detect "first call" by checking
     * whether the file currently exists / is non-empty. */
    struct stat st;
    bool        first = (stat(o->output_md, &st) != 0) || st.st_size == 0;

    FILE* f = fopen(o->output_md, "a");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for append\n", o->output_md);
        return 1;
    }
    if (first) {
        fprintf(f,
            "| Model | Size | Backend | Device | ngl | Test | TTFT (ms) | Media enc (ms) | Prefill (tok/s) | "
            "Decode (tok/s) |\n"
            "|-------|-----:|---------|--------|----:|------|----------:|---------------:|----------------:|"
            "---------------:|\n");
    }

    char  size_buf[32];
    char  ngl_buf[16];
    char  test_buf[32];
    char* model = model_label(o->cell_id, o->plugin, o->device);
    format_bytes(model_size_bytes, size_buf, sizeof(size_buf));
    if (strcmp(o->plugin, "qairt") == 0 || dev->ngl <= 0) {
        snprintf(ngl_buf, sizeof(ngl_buf), "-");
    } else {
        snprintf(ngl_buf, sizeof(ngl_buf), "%d", dev->ngl);
    }
    snprintf(
        test_buf, sizeof(test_buf), "pp%lld+tg%lld", (long long)a->prompt_tokens_med, (long long)a->gen_tokens_med);

    char media_buf[24];
    if (a->media_ms_med > 0)
        snprintf(media_buf, sizeof(media_buf), "%.1f", a->media_ms_med);
    else
        snprintf(media_buf, sizeof(media_buf), "-");

    fprintf(f,
        "| %s | %s | %s | %s | %s | %s | %.1f ± %.1f | %s | %.1f ± %.1f | %.1f ± %.1f |\n",
        model ? model : cell_name(o),
        size_buf,
        o->plugin,
        o->device,
        ngl_buf,
        test_buf,
        a->ttft_ms.med,
        a->ttft_ms.sd,
        media_buf,
        a->prefill_tps.med,
        a->prefill_tps.sd,
        a->decode_tps.med,
        a->decode_tps.sd);
    free(model);
    fclose(f);
    return 0;
}
