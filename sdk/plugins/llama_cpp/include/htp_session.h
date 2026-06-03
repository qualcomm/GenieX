#pragma once

namespace geniex {

// HTP session lifecycle helpers shared by all llama.cpp plugin classes.
//
// llama.cpp's Hexagon backend opens FastRPC channels to the CDSP at registry
// construction time (ggml_hexagon_registry). Those channels persist for the
// life of the process, which means a QAIRT plugin spun up after llama.cpp
// tries to open its own dspqueue on the same CDSP domain and collides with
// llama.cpp's still-open libggml-htp-vN.so / libdspqueue_rpc_skel.so handles —
// QAIRT reports "Failed to create device: 1002" or 1007.
//
// Workaround: when the last HTP-bound llama.cpp object is destroyed, call
// ggml_backend_hexagon_release_sessions (exposed by our llama.cpp patch) to
// close those handles. Before the next llama.cpp load we call
// ggml_backend_hexagon_reacquire_sessions so the cached device pointers in
// ggml-backend-reg have live sessions again.
//
// Classes that may load on HTP (LlamaLlm / LlamaVlm / LlamaCppEmbedding /
// LlamaCppReranker) own an htp::SessionGuard with RAII semantics: call
// reacquire_before_load() before llama_model_load_from_file, mark_htp() when
// the HTP backend is registered, and the destructor releases when it was the
// last live user. Tracking is registry-scoped (not device-scoped) because
// llama.cpp opens HTP FastRPC channels at ggml_hexagon_registry construction
// time regardless of per-load device_id/n_gpu_layers.
namespace htp {

// Recreate any HTP sessions that were released by a prior handoff. No-op if
// the HTP backend is absent or sessions are already live. Safe to call on
// every llama.cpp load.
void reacquire_before_load();

// Returns true iff the ggml backend registry has a backend named "HTP".
// When true, llama.cpp has live FastRPC channels to CDSP that need to be
// torn down before a QAIRT plugin can take over.
bool htp_backend_present();

class SessionGuard {
   public:
    SessionGuard() = default;
    ~SessionGuard() { release_if_last(); }

    SessionGuard(const SessionGuard&)            = delete;
    SessionGuard& operator=(const SessionGuard&) = delete;

    // Call after deciding uses_htp (post model-load so we know the real
    // device selection). Safe to call multiple times — only the first flip
    // from false→true increments the shared refcount.
    void mark_htp();

    bool uses_htp() const { return uses_htp_; }

   private:
    void release_if_last();

    bool uses_htp_ = false;
};

}  // namespace htp
}  // namespace geniex
