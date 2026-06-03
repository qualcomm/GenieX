/**
 * End-to-end test for the model manager C API.
 *
 * Build (after cmake -DGENIEX_MODEL_MANAGER=ON):
 *
 *   cmake --build <build-dir> --target test_model_manager
 *
 * Run:
 *   LD_LIBRARY_PATH=<build-dir>/src GENIEX_DATADIR=/tmp/geniex-test \
 *       ./<build-dir>/src/test_model_manager
 *
 * Two test modes:
 *   1. LocalFS (always runs): creates a dummy geniex.json and model file in /tmp,
 *      then exercises the full CRUD + path resolution flow.
 *   2. HuggingFace (optional, set GENIEX_TEST_HF=1): downloads a real model from HF.
 *      Requires network access and a geniex.json in the target HF repo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

#include "geniex.h"
#include "geniex_model.h"

/* ---- helpers ---- */

#define CHECK(call)                                             \
    do {                                                        \
        int32_t _rc = (call);                                   \
        if (_rc != GENIEX_SUCCESS) {                            \
            fprintf(stderr, "FAIL  %s  (rc=%d)\n", #call, _rc); \
            return 1;                                           \
        }                                                       \
        printf("OK    %s\n", #call);                            \
    } while (0)

#define EXPECT_FAIL(call, expected_rc)                                                          \
    do {                                                                                        \
        int32_t _rc = (call);                                                                   \
        if (_rc != (expected_rc)) {                                                             \
            fprintf(stderr, "FAIL  %s  expected rc=%d got rc=%d\n", #call, (expected_rc), _rc); \
            return 1;                                                                           \
        }                                                                                       \
        printf("OK    %s  (expected failure rc=%d)\n", #call, _rc);                             \
    } while (0)

static bool progress_cb(const geniex_FileProgress* files, int32_t file_count, void* user_data) {
    int* counter = (int*)user_data;
    (*counter)++;
    for (int i = 0; i < file_count; i++) {
        printf("      progress[%d]: %s  %lld / %lld\n",
            i,
            files[i].file_name ? files[i].file_name : "(null)",
            (long long)files[i].downloaded_bytes,
            (long long)files[i].total_bytes);
    }
    return true; /* never cancel */
}

static void mkdir_p(const char* path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    fputs(content, f);
    fclose(f);
}

/* ---- LocalFS test ---- */

static int test_localfs(const char* data_dir) {
    printf("\n=== LocalFS test ===\n");

    /* 1. Create a fake model source directory */
    const char* src_dir = "/tmp/geniex-localfs-src/NexaAI/TestModel-GGUF";
    mkdir_p(src_dir);

    /* Fake GGUF file */
    char gguf_path[512];
    snprintf(gguf_path, sizeof(gguf_path), "%s/model-Q4_K_M.gguf", src_dir);
    write_file(gguf_path, "fake gguf content");

    /* geniex.json manifest */
    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/geniex.json", src_dir);
    write_file(manifest_path,
        "{"
        "\"Name\":\"NexaAI/TestModel-GGUF\","
        "\"ModelName\":\"test-1b\","
        "\"ModelType\":\"llm\","
        "\"PluginId\":\"llama_cpp\","
        "\"ModelFile\":{\"Q4_K_M\":{\"Name\":\"model-Q4_K_M.gguf\",\"Downloaded\":true,\"Size\":17}},"
        "\"MMProjFile\":{\"Name\":\"\",\"Downloaded\":false,\"Size\":0},"
        "\"TokenizerFile\":{\"Name\":\"\",\"Downloaded\":false,\"Size\":0},"
        "\"ExtraFiles\":[]"
        "}");

    /* 2. Pull from LocalFS */
    int                   progress_calls = 0;
    geniex_ModelPullInput pull_input;
    memset(&pull_input, 0, sizeof(pull_input));
    pull_input.struct_size = sizeof(pull_input);
    pull_input.model_name  = "NexaAI/TestModel-GGUF";
    pull_input.quant       = NULL;
    pull_input.hub         = GENIEX_HUB_LOCALFS;
    pull_input.local_path  = "/tmp/geniex-localfs-src/NexaAI/TestModel-GGUF";
    pull_input.hf_token    = NULL;
    pull_input.on_progress = progress_cb;
    pull_input.user_data   = &progress_calls;
    CHECK(geniex_model_pull(&pull_input));
    if (progress_calls == 0) {
        fprintf(stderr, "FAIL  progress callback never invoked\n");
        return 1;
    }
    printf("      progress callback invoked %d time(s) ✓\n", progress_calls);

    /* 3. List */
    geniex_ModelListDetailedOutput list = {0};
    CHECK(geniex_model_list_detailed(&list));
    printf("      cached models: %d\n", list.count);
    if (list.count < 1) {
        fprintf(stderr, "FAIL  expected at least 1 cached model\n");
        geniex_model_list_detailed_free(&list);
        return 1;
    }
    printf("      [0] %s\n", list.models[0].name);
    geniex_model_list_detailed_free(&list);

    /* 4. Get type */
    geniex_ModelType mtype;
    CHECK(geniex_model_get_type("NexaAI/TestModel-GGUF", &mtype));
    if (mtype != GENIEX_MODEL_TYPE_LLM) {
        fprintf(stderr, "FAIL  expected GENIEX_MODEL_TYPE_LLM (%d), got %d\n", GENIEX_MODEL_TYPE_LLM, mtype);
        return 1;
    }
    printf("      model type: LLM ✓\n");

    /* 5. Get paths */
    geniex_ModelPaths paths = {0};
    CHECK(geniex_model_get_paths("NexaAI/TestModel-GGUF", &paths));
    printf("      model_path: %s\n", paths.model_path ? paths.model_path : "(null)");
    printf("      model_dir:  %s\n", paths.model_dir ? paths.model_dir : "(null)");
    printf("      model_name: %s\n", paths.model_name ? paths.model_name : "(null)");
    printf("      plugin_id:  %s\n", paths.plugin_id ? paths.plugin_id : "(null)");
    if (!paths.model_path || strstr(paths.model_path, "model-Q4_K_M.gguf") == NULL) {
        fprintf(stderr, "FAIL  model_path does not contain expected filename\n");
        geniex_model_paths_free(&paths);
        return 1;
    }
    printf("      model_path contains 'model-Q4_K_M.gguf' ✓\n");
    geniex_model_paths_free(&paths);

    /* 6. Get paths with explicit quant */
    CHECK(geniex_model_get_paths("NexaAI/TestModel-GGUF:Q4_K_M", &paths));
    geniex_model_paths_free(&paths);

    /* 7. Error case: unknown quant */
    EXPECT_FAIL(geniex_model_get_paths("NexaAI/TestModel-GGUF:Q8_0", &paths), GENIEX_ERROR_COMMON_INVALID_INPUT);

    /* 8. Remove */
    CHECK(geniex_model_remove("NexaAI/TestModel-GGUF"));

    /* 9. Verify gone */
    geniex_ModelListDetailedOutput list2 = {0};
    CHECK(geniex_model_list_detailed(&list2));
    if (list2.count != 0) {
        fprintf(stderr, "FAIL  expected 0 models after remove, got %d\n", list2.count);
        geniex_model_list_detailed_free(&list2);
        return 1;
    }
    printf("      list after remove: 0 ✓\n");
    geniex_model_list_detailed_free(&list2);

    printf("=== LocalFS test PASSED ===\n");
    return 0;
}

/* ---- alias test ---- */

static int test_alias(void) {
    printf("\n=== Alias test ===\n");
    char* full = NULL;
    CHECK(geniex_model_resolve_alias("qwen3", &full));
    printf("      qwen3 -> %s\n", full);
    geniex_free(full);

    /* unknown alias should fail */
    EXPECT_FAIL(geniex_model_resolve_alias("nonexistent_model_xyz_abc", &full), GENIEX_ERROR_COMMON_INVALID_INPUT);
    printf("=== Alias test PASSED ===\n");
    return 0;
}

/* ---- main ---- */

int main(void) {
    const char* data_dir = getenv("GENIEX_DATADIR");

    printf("=== geniex_model_init ===\n");
    CHECK(geniex_model_init(data_dir));

    if (test_alias()) return 1;
    if (test_localfs(data_dir ? data_dir : "/tmp/geniex-test")) return 1;

    CHECK(geniex_model_deinit());

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
