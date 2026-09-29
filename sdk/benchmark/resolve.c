// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

/* Turning the `-m` / matrix col 4 value into paths the SDK can open:
 * model-manager ids (download + cache), local QAIRT VLM bundle detection,
 * and directory -> anchor-file resolution. */

#include <stdlib.h>
#include <string.h>

#include "bench.h"

/* Process-wide guard: geniex_model_init is one-shot per process (a second
 * call returns INVALID_INPUT). matrix mode copies options_t per cell so
 * the flag must live outside the struct. */
static bool g_mm_inited = false;

void apply_chipset_defaults(options_t* o, const device_t* dev) {
    o->n_ubatch = 0;
    if (!o->plugin || strcmp(o->plugin, "llama_cpp") != 0 || !dev->id || strcmp(dev->id, "GPUOpenCL") != 0 ||
        dev->ngl == 0) {
        return;
    }

    char*   chipset = NULL;
    int32_t rc      = geniex_model_detect_chipset(1, &chipset);
    if (rc != GENIEX_SUCCESS) {
        fprintf(stderr, "[warn] host chipset detection failed (%d); using SDK batch defaults\n", rc);
    } else if (chipset && strcmp(chipset, "qualcomm-qcs6490") == 0) {
        o->n_ubatch = 256;
        fprintf(stderr, "[info] %s: setting n_ubatch=256 for %s on %s\n", cell_name(o), dev->id, chipset);
    }
    if (chipset) geniex_free(chipset);
}

/* A model-manager id is `org/repo[:quant]`: an embedded '/', no leading
 * '/' '.' or '\\', no drive prefix. Everything else is a path — including a
 * bare `model.gguf`, which has no '/' and so belongs in the cwd branch. */
bool looks_like_path(const char* s) {
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

/* Read metadata.json's genie.supports_vision, so a VLM bundle reaches the VLM
 * run loop even when --vlm / matrix col 8 is unset. `model_path` may be a
 * bundle directory or a file inside it. Any read/parse miss means "not VLM". */
bool local_bundle_is_vlm(const char* model_path) {
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

/* geniex_model_init is one-shot per process; call it lazily so matrix mode
 * (many cells, one process) can ask on every cell. */
static int mm_init_once(const char* data_dir) {
    if (g_mm_inited) return 0;
    int32_t rc = geniex_model_init(data_dir);
    if (rc != GENIEX_SUCCESS) {
        const char* m = geniex_model_last_error_message();
        fprintf(stderr, "ERROR: geniex_model_init: %s (%d)\n", m ? m : "?", rc);
        return 1;
    }
    g_mm_inited = true;
    return 0;
}

/* Resolve model-manager id `id` to `out`, downloading on cache miss. `kind`
 * labels the progress line ("" for the main model, "draft " for the draft).
 * Returns 0 on success; on success the caller owns every string in `out` and
 * releases the leftovers with geniex_model_paths_free. */
static int mm_resolve(const options_t* o, const char* id, const char* kind, geniex_ModelPaths* out) {
    if (mm_init_once(o->mm_data_dir) != 0) return 1;

    /* Copy the id into a writable buffer for in-place split_id(). */
    size_t n   = strlen(id);
    char*  buf = (char*)malloc(n + 1);
    if (!buf) return 1;
    memcpy(buf, id, n + 1);
    const char* name;
    const char* quant;
    split_id(buf, &name, &quant);

    memset(out, 0, sizeof(*out));
    /* Try `get_paths` first — already cached is the common path on the
     * second cell of a matrix. Fall through to pull on file-not-found. */
    int32_t rc = geniex_model_get_paths(id, out);
    if (rc != GENIEX_SUCCESS) {
        geniex_ModelPullInput in;
        memset(&in, 0, sizeof(in));
        in.struct_size = (uint32_t)sizeof(in);
        in.model_name  = name;
        in.quant       = quant;
        in.hub         = parse_hub(o->mm_hub);
        in.chipset     = o->mm_chipset;
        in.model_type  = GENIEX_MODEL_TYPE_AUTO;
        fprintf(stderr, "[mm  ] pulling %s%s%s%s ...\n", kind, name, quant ? ":" : "", quant ? quant : "");
        rc = geniex_model_pull(&in);
        if (rc != GENIEX_SUCCESS) {
            const char* m = geniex_model_last_error_message();
            fprintf(stderr, "ERROR: geniex_model_pull(%s): %s (%d)\n", id, m ? m : "?", rc);
            free(buf);
            return 1;
        }
        rc = geniex_model_get_paths(id, out);
        if (rc != GENIEX_SUCCESS) {
            const char* m = geniex_model_last_error_message();
            fprintf(stderr, "ERROR: geniex_model_get_paths(%s): %s (%d)\n", id, m ? m : "?", rc);
            free(buf);
            return 1;
        }
    }
    free(buf);
    return 0;
}

int resolve_model_id(options_t* o, const char* id_in) {
    geniex_ModelPaths paths;
    if (mm_resolve(o, id_in, "", &paths) != 0) return 1;

    /* Take ownership of the heap strings the SDK handed us. */
    o->mm.model_path = paths.model_path;
    o->mm.mmproj     = paths.mmproj_path;
    o->mm.tokenizer  = paths.tokenizer_path;
    /* The CLI's _is_vlm() signal (3). */
    o->mm.is_vlm     = (paths.model_type == GENIEX_MODEL_TYPE_VLM);
    paths.model_path = paths.mmproj_path = paths.tokenizer_path = NULL;
    /* model_dir / model_name / plugin_id aren't consumed here; free via the
     * paths_free() helper to keep the allocator pairing intact. */
    geniex_model_paths_free(&paths);

    o->model_path = o->mm.model_path;
    /* QAIRT VLM bundles carry no mmproj and the dispatcher has no LLM factory
     * for a VLM model_id, so such a bundle would hard-fail inside run_llm. */
    if (o->mm.is_vlm && o->plugin && strcmp(o->plugin, "qairt") == 0 && !o->force_vlm) {
        fprintf(stderr, "[mm  ] %s is a VLM bundle; forcing VLM run loop\n", id_in);
        o->force_vlm = true;
    }
    /* Adopt the manager's mmproj only under an explicit VLM request — see the
     * shadowing rule on options_t.model_path. An explicit --mmproj-path or
     * matrix col 6 was already assigned and still wins. */
    if (o->force_vlm && o->mmproj_path == NULL && o->mm.mmproj) {
        o->mmproj_path = o->mm.mmproj;
    }
    if (o->mm.tokenizer) o->tokenizer_path = o->mm.tokenizer;
    fprintf(stderr, "[mm  ] resolved %s -> %s\n", id_in, o->mm.model_path);
    return 0;
}

/* Without this the plugin gets a raw id like "org/repo:Q4_0", fails to
 * open it as a file, and silently falls back to non-speculative decode. */
int resolve_draft_id(options_t* o) {
    if (!o->draft_model || looks_like_path(o->draft_model)) return 0;

    /* Capture the id before o->draft_model is repointed at the resolved path,
     * so the log below reports id -> path rather than path -> path. */
    const char*       id = o->draft_model;
    geniex_ModelPaths paths;
    if (mm_resolve(o, id, "draft ", &paths) != 0) return 1;

    o->mm.draft_model = paths.model_path;
    paths.model_path  = NULL;
    geniex_model_paths_free(&paths);
    o->draft_model = o->mm.draft_model;
    fprintf(stderr, "[mm  ] resolved draft %s -> %s\n", id, o->mm.draft_model);
    return 0;
}

/* The SDK derives the model dir via `parent_path()`, so it needs a *file*
 * path, not a directory path: return a heap path to a regular file inside
 * `path` (preferring `tokenizer.json`, otherwise the lexicographically first
 * regular file). Mirrors `_resolve_local_anchor` in
 * bindings/python/geniex/auto.py:122.
 *
 * A regular file (e.g. an explicit *.gguf) yields NULL — the caller should
 * keep using the original path. Callers must free the returned string. */
char* resolve_local_anchor(const char* path) {
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

/* Release the mm_* paths a cell took ownership of and NULL them out. */
void free_mm_paths(options_t* o) {
    if (o->mm.model_path) {
        geniex_free(o->mm.model_path);
        o->mm.model_path = NULL;
    }
    if (o->mm.mmproj) {
        geniex_free(o->mm.mmproj);
        o->mm.mmproj = NULL;
    }
    if (o->mm.tokenizer) {
        geniex_free(o->mm.tokenizer);
        o->mm.tokenizer = NULL;
    }
    if (o->mm.draft_model) {
        geniex_free(o->mm.draft_model);
        o->mm.draft_model = NULL;
    }
}

void mm_shutdown(void) {
    if (!g_mm_inited) return;
    /* Best-effort: failure here is non-fatal — we already produced the JSON
     * the caller cares about. */
    geniex_model_deinit();
    g_mm_inited = false;
}
