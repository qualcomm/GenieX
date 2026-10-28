// SPDX-License-Identifier: BSD-3-Clause
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "geniex_tts.h"

extern "C" int tts_c_header_compiles(void);
void           check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Context {
    geniex_TTS*      handle;
    std::atomic<int> frames{0};
    int              mode = 0;
};
bool audio(const float* pcm, uint32_t count, uint32_t rate, void* user) {
    auto& ctx = *static_cast<Context*>(user);
    check(count == 4 && rate == 24000 && pcm[1] == 0.5f && pcm[2] == -0.5f, "PCM corrupted");
    ++ctx.frames;
    if (ctx.mode == 1) return false;
    if (ctx.mode == 2) check(geniex_tts_cancel(ctx.handle) == 0, "cancel from callback");
    return true;
}
int main() {
    try {
        check(tts_c_header_compiles(), "C/C++ ABI");
        check(geniex_init() == 0, "init");
        geniex_TTS* h = reinterpret_cast<geniex_TTS*>(1);
        check(geniex_tts_create(nullptr, &h) == GENIEX_ERROR_COMMON_INVALID_INPUT && !h, "null create");
        geniex_TtsCreateInput create{sizeof(create), 0, "mock_tts", "unused", "npu"};
        check(geniex_tts_create(&create, nullptr) == GENIEX_ERROR_COMMON_INVALID_INPUT, "null output handle");
        create.struct_size--;
        check(geniex_tts_create(&create, &h) == GENIEX_ERROR_COMMON_INVALID_INPUT && !h, "version guard");
        create.struct_size++;
        create.flags = 2;
        check(geniex_tts_create(&create, &h) == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED, "unknown flag");
        create.flags     = 0;
        create.plugin_id = "absent";
        check(geniex_tts_create(&create, &h) == GENIEX_ERROR_COMMON_PLUGIN_INVALID, "absent plugin");
        create.plugin_id = "legacy";
        check(geniex_tts_create(&create, &h) == GENIEX_ERROR_COMMON_NOT_SUPPORTED && !h, "legacy plugin compatibility");
        create.plugin_id = "mock_tts";
        check(geniex_tts_create(&create, &h) == 0 && h, "create");
        Context                    ctx{h};
        geniex_TtsSynthesizeInput  in{sizeof(in), "test", "", audio, &ctx};
        geniex_TtsSynthesizeOutput out{sizeof(out)};
        check(geniex_tts_synthesize(nullptr, &in, &out) == GENIEX_ERROR_COMMON_INVALID_INPUT, "null synthesis");
        in.on_audio = nullptr;
        check(geniex_tts_synthesize(h, &in, &out) == GENIEX_ERROR_COMMON_INVALID_INPUT, "null callback");
        in.on_audio = audio;
        in.text     = "";
        check(geniex_tts_synthesize(h, &in, &out) == GENIEX_ERROR_COMMON_INVALID_INPUT, "empty text");
        in.text = "test";
        check(geniex_tts_synthesize(h, &in, &out) == 0 && out.samples == 40 && ctx.frames == 10, "stream");
        ctx.mode = 1;
        check(geniex_tts_synthesize(h, &in, &out) == GENIEX_ERROR_COMMON_CANCELLED && out.samples == 4,
            "callback cancellation");
        ctx.mode = 2;
        check(geniex_tts_synthesize(h, &in, &out) == GENIEX_ERROR_COMMON_CANCELLED, "explicit callback cancellation");
        ctx.mode   = 0;
        ctx.frames = 0;
        std::thread stopper([&] {
            while (ctx.frames == 0) std::this_thread::yield();
            geniex_tts_cancel(h);
        });
        int         status = geniex_tts_synthesize(h, &in, &out);
        stopper.join();
        check(status == GENIEX_ERROR_COMMON_CANCELLED, "cross-thread cancellation");
        ctx.frames = 0;
        check(geniex_tts_synthesize(h, &in, &out) == 0 && out.samples == 40, "reuse after cancellation");
        in.text = "throw";
        check(geniex_tts_synthesize(h, &in, &out) == GENIEX_ERROR_COMMON_UNKNOWN, "exception boundary");
        check(geniex_tts_destroy(h) == 0, "destroy");
        check(geniex_tts_destroy(nullptr) == GENIEX_ERROR_COMMON_INVALID_INPUT, "null destroy");
        check(geniex_tts_cancel(nullptr) == GENIEX_ERROR_COMMON_INVALID_INPUT, "null cancel");
        check(geniex_deinit() == 0, "deinit");
        std::cout << "TTS contracts passed (including legacy plugin and threaded cancellation)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
