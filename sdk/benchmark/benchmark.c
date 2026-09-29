// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/*
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
 * The model argument takes either a filesystem path or a model-manager id
 * (`org/repo[:quant]`, `qualcomm/<aihub_repo>`); the id form downloads and
 * resolves through geniex_model_*, replacing the curl/IWR shell loops the
 * QDC bench run used to carry per device.
 *
 * This file holds the drivers only; see bench.h for the module layout.
 */

#include <stdlib.h>
#include <string.h>

#include "bench.h"

/* Run one (plugin, device, model) cell using the already-`geniex_init`'d
 * runtime. Returns 0 on success, non-zero on failure. The caller owns
 * `geniex_init` / `geniex_deinit` so multiple cells in matrix mode share
 * one plugin-init pass. */
static int run_one_cell(options_t* o) {
    /* Single exit through `done`: the model-manager paths in o->mm are
     * heap-owned by this cell, so every failure path has to release them. */
    int                        result   = 0;
    char*                      anchored = NULL;
    run_result_t*              runs     = NULL;
    geniex_ResolveDeviceOutput rout;
    memset(&rout, 0, sizeof(rout));

    /* Model-manager branch: column 4 (matrix) or `-m` (single-cell) is a
     * model id. Resolve to local paths (pulling on first use); subsequent
     * cells with the same id hit the cache. mmproj/tokenizer columns from
     * the matrix file are ignored in this branch — the manager owns the
     * full path tuple. */
    if (!looks_like_path(o->model_path)) {
        if (resolve_model_id(o, o->model_path) != 0) {
            result = 1;
            goto done;
        }
    } else if (o->plugin && strcmp(o->plugin, "qairt") == 0 && !o->force_vlm && local_bundle_is_vlm(o->model_path)) {
        /* Local QAIRT VLM bundle (resolve_model_id skipped for path inputs):
         * force VLM so it doesn't hit the dispatcher's "no LLM factory". */
        fprintf(stderr, "[info] %s is a VLM bundle (metadata.json); forcing VLM run loop\n", o->model_path);
        o->force_vlm = true;
    }

    if (resolve_draft_id(o) != 0) {
        result = 1;
        goto done;
    }

    anchored = resolve_local_anchor(o->model_path);
    if (anchored) {
        fprintf(stderr, "[info] resolved model dir to anchor: %s\n", anchored);
        o->model_path = anchored;
    }

    /* Device-alias resolution. ngl_default=-1 means "all layers" to
     * llama.cpp (cpu and qairt are forced to 0 by the SDK). */
    geniex_ResolveDeviceInput rin;
    memset(&rin, 0, sizeof(rin));
    rin.plugin_id   = o->plugin;
    rin.mode        = o->device;
    rin.ngl_default = -1;
    int32_t rc      = geniex_resolve_device(&rin, &rout);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr, "ERROR: geniex_resolve_device: %s (%d)\n", geniex_get_error_message((geniex_ErrorCode)rc), rc);
        result = 1;
        goto done;
    }
    if (rout.warning) {
        fprintf(stderr, "[warn] %s\n", rout.warning);
    }
    device_t dev;
    dev.id  = o->device_id ? o->device_id : rout.device_id;
    dev.ngl = rout.ngl;
    /* --n-gpu-layers overrides the resolved value. */
    if (o->ngl_override >= 0) {
        dev.ngl = o->ngl_override;
    }

    /* The qairt plugin doesn't consume n_gpu_layers or n_ctx; force both to 0
     * to match `_build_model_config()` in bindings/python/geniex/auto.py:179. */
    if (strcmp(o->plugin, "qairt") == 0) {
        dev.ngl  = 0;
        o->n_ctx = 0;
    }

    apply_chipset_defaults(o, &dev);

    bool is_vlm = (o->mmproj_path != NULL) || o->force_vlm;

    /* --logits is a prefill-only forward pass, not a timing run: it skips the
     * warmup/repeat/aggregate_runs machinery and writes its own report. */
    if (o->logits_mode) {
        if (is_vlm) {
            fprintf(stderr, "ERROR: --logits is not supported for VLM models\n");
            result = 1;
        } else {
            result = run_logits(o, &dev);
        }
        goto done;
    }

    /* warmup + repeat runs per prompt; only the measured ones are recorded. */
    int32_t n_runs = o->repeat * o->prompt_count;
    runs           = (run_result_t*)calloc((size_t)n_runs, sizeof(run_result_t));
    if (!runs) {
        fprintf(stderr, "ERROR: oom\n");
        result = 1;
        goto done;
    }

    if (is_vlm) {
        run_vlm(o, &dev, runs);
    } else {
        run_llm(o, &dev, runs);
    }

    agg_t a;
    aggregate_runs(runs, n_runs, &a);
    print_summary(o, &dev, &a);

    int64_t model_size_bytes = model_disk_bytes(o->model_path);
    if (o->output_json && write_cell_json(o, &dev, model_size_bytes, runs, n_runs, &a) != 0) result = 1;
    if (o->output_md && write_md_row(o, &dev, model_size_bytes, &a) != 0) result = 1;

done:
    free(runs);
    free(anchored);
    if (rout.device_id) geniex_free(rout.device_id);
    if (rout.warning) geniex_free(rout.warning);
    free_mm_paths(o);
    return result;
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

    /* Before init, and run-wide rather than per cell: the QNN libraries load once per
     * process, so matrix mode cannot switch runtimes between cells. */
    if (o.qairt_lib) check(geniex_set_qairt_runtime_path(o.qairt_lib), "geniex_set_qairt_runtime_path");

    check(geniex_init(), "geniex_init");

    int rc;
    if (o.matrix_file) {
        rc = run_matrix(&o);
    } else {
        rc = run_one_cell(&o);
    }

    free(o.prompts);
    if (o.prompt_buf) free(o.prompt_buf);
    /* Release the model-manager runtime before geniex_deinit. */
    mm_shutdown();
    check(geniex_deinit(), "geniex_deinit");
    return rc == 0 ? 0 : 1;
}
