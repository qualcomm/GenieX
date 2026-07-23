use crate::ffi;
use std::ffi::CString;
use std::os::raw::c_char;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_TRACE as i32,
    Debug = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_DEBUG as i32,
    Info = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_INFO as i32,
    Warn = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_WARN as i32,
    Error = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_ERROR as i32,
}

#[derive(Debug, Clone, Default)]
pub struct ProfileData {
    pub ttft: i64,
    pub prompt_time: i64,
    pub decode_time: i64,
    pub prompt_tokens: i64,
    pub generated_tokens: i64,
    pub audio_duration: i64,
    pub prefill_speed: f64,
    pub decoding_speed: f64,
    pub real_time_factor: f64,
    pub stop_reason: String,
}

impl From<&ffi::geniex_ProfileData> for ProfileData {
    fn from(raw: &ffi::geniex_ProfileData) -> Self {
        let stop_reason = if raw.stop_reason.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(raw.stop_reason).to_string_lossy().into_owned() }
        };
        Self {
            ttft: raw.ttft,
            prompt_time: raw.prompt_time,
            decode_time: raw.decode_time,
            prompt_tokens: raw.prompt_tokens,
            generated_tokens: raw.generated_tokens,
            audio_duration: raw.audio_duration,
            prefill_speed: raw.prefill_speed,
            decoding_speed: raw.decoding_speed,
            real_time_factor: raw.real_time_factor,
            stop_reason,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: i32,
    pub grammar_path: Option<String>,
    pub grammar_string: Option<String>,
    pub enable_json: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: -1,
            grammar_path: None,
            grammar_string: None,
            enable_json: false,
        }
    }
}

pub(crate) struct RawSamplerConfig {
    pub raw: ffi::geniex_SamplerConfig,
    _grammar_path: Option<CString>,
    _grammar_string: Option<CString>,
}

impl SamplerConfig {
    pub(crate) fn to_raw(&self) -> RawSamplerConfig {
        let grammar_path_c = self.grammar_path.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        let grammar_string_c = self.grammar_string.as_ref().map(|s| CString::new(s.as_str()).unwrap());

        let raw = ffi::geniex_SamplerConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            grammar_path: grammar_path_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            grammar_string: grammar_string_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            enable_json: self.enable_json,
        };

        RawSamplerConfig {
            raw,
            _grammar_path: grammar_path_c,
            _grammar_string: grammar_string_c,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenerationConfig {
    pub max_tokens: i32,
    pub stop: Vec<String>,
    pub n_past: i32,
    pub sampler_config: Option<SamplerConfig>,
    pub image_paths: Vec<String>,
    pub image_max_length: i32,
    pub audio_paths: Vec<String>,
    pub sliding_window: bool,
    pub sliding_window_n_keep: i32,
}

pub(crate) struct RawGenerationConfig {
    pub raw: ffi::geniex_GenerationConfig,
    _raw_sampler: Option<Box<RawSamplerConfig>>,
    _stops: Vec<CString>,
    _stop_ptrs: Vec<*const c_char>,
    _image_paths: Vec<CString>,
    _image_path_ptrs: Vec<*const c_char>,
    _audio_paths: Vec<CString>,
    _audio_path_ptrs: Vec<*const c_char>,
}

impl GenerationConfig {
    pub(crate) fn to_raw(&self) -> RawGenerationConfig {
        let raw_sampler = self.sampler_config.as_ref().map(|s| Box::new(s.to_raw()));
        let sampler_ptr = raw_sampler.as_ref().map_or(std::ptr::null_mut(), |s| &s.raw as *const _ as *mut _);

        let stops: Vec<CString> = self.stop.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
        let stop_ptrs: Vec<*const c_char> = stops.iter().map(|s| s.as_ptr()).collect();

        let image_paths: Vec<CString> = self.image_paths.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
        let image_path_ptrs: Vec<*const c_char> = image_paths.iter().map(|s| s.as_ptr()).collect();

        let audio_paths: Vec<CString> = self.audio_paths.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
        let audio_path_ptrs: Vec<*const c_char> = audio_paths.iter().map(|s| s.as_ptr()).collect();

        let raw = ffi::geniex_GenerationConfig {
            max_tokens: self.max_tokens,
            stop: if stop_ptrs.is_empty() { std::ptr::null_mut() } else { stop_ptrs.as_ptr() as *mut *const c_char },
            stop_count: stop_ptrs.len() as i32,
            n_past: self.n_past,
            sampler_config: sampler_ptr,
            image_paths: if image_path_ptrs.is_empty() { std::ptr::null_mut() } else { image_path_ptrs.as_ptr() as *mut *const c_char },
            image_count: image_path_ptrs.len() as i32,
            image_max_length: self.image_max_length,
            audio_paths: if audio_path_ptrs.is_empty() { std::ptr::null_mut() } else { audio_path_ptrs.as_ptr() as *mut *const c_char },
            audio_count: audio_path_ptrs.len() as i32,
            sliding_window: self.sliding_window,
            sliding_window_n_keep: self.sliding_window_n_keep,
        };

        RawGenerationConfig {
            raw,
            _raw_sampler: raw_sampler,
            _stops: stops,
            _stop_ptrs: stop_ptrs,
            _image_paths: image_paths,
            _image_path_ptrs: image_path_ptrs,
            _audio_paths: audio_paths,
            _audio_path_ptrs: audio_path_ptrs,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub n_ctx: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_seq_max: i32,
    pub n_gpu_layers: i32,
    pub chat_template_path: Option<String>,
    pub chat_template_content: Option<String>,
    pub system_prompt: Option<String>,
    pub enable_sampling: bool,
    pub grammar_str: Option<String>,
    pub max_tokens: i32,
    pub enable_thinking: bool,
    pub verbose: bool,
}

pub(crate) struct RawModelConfig {
    pub raw: ffi::geniex_ModelConfig,
    _chat_template_path: Option<CString>,
    _chat_template_content: Option<CString>,
    _system_prompt: Option<CString>,
    _grammar_str: Option<CString>,
}

impl ModelConfig {
    pub(crate) fn to_raw(&self) -> RawModelConfig {
        let chat_template_path_c = self.chat_template_path.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        let chat_template_content_c = self.chat_template_content.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        let system_prompt_c = self.system_prompt.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        let grammar_str_c = self.grammar_str.as_ref().map(|s| CString::new(s.as_str()).unwrap());

        let raw = ffi::geniex_ModelConfig {
            n_ctx: self.n_ctx,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_gpu_layers: self.n_gpu_layers,
            chat_template_path: chat_template_path_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            chat_template_content: chat_template_content_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            system_prompt: system_prompt_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            enable_sampling: self.enable_sampling,
            grammar_str: grammar_str_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
            verbose: self.verbose,
        };

        RawModelConfig {
            raw,
            _chat_template_path: chat_template_path_c,
            _chat_template_content: chat_template_content_c,
            _system_prompt: system_prompt_c,
            _grammar_str: grammar_str_c,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub(crate) struct RawLlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_LlmChatMessage>,
    _roles: Vec<CString>,
    _contents: Vec<CString>,
}

impl ChatMessage {
    pub(crate) fn vec_to_raw(messages: &[ChatMessage]) -> RawLlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut contents = Vec::with_capacity(messages.len());
        let mut raw_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role_c = CString::new(msg.role.as_str()).unwrap();
            let content_c = CString::new(msg.content.as_str()).unwrap();
            raw_messages.push(ffi::geniex_LlmChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
            });
            roles.push(role_c);
            contents.push(content_c);
        }

        RawLlmChatMessages {
            raw_messages,
            _roles: roles,
            _contents: contents,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VlmContent {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct VlmChatMessage {
    pub role: String,
    pub contents: Vec<VlmContent>,
}

pub(crate) struct RawVlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_VlmChatMessage>,
    _roles: Vec<CString>,
    _content_structs: Vec<Vec<ffi::geniex_VlmContent>>,
    _type_cstrings: Vec<Vec<CString>>,
    _text_cstrings: Vec<Vec<CString>>,
}

impl VlmChatMessage {
    pub(crate) fn vec_to_raw(messages: &[VlmChatMessage]) -> RawVlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut content_structs = Vec::with_capacity(messages.len());
        let mut type_cstrings = Vec::with_capacity(messages.len());
        let mut text_cstrings = Vec::with_capacity(messages.len());
        let mut raw_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role_c = CString::new(msg.role.as_str()).unwrap();
            let mut sub_types = Vec::with_capacity(msg.contents.len());
            let mut sub_texts = Vec::with_capacity(msg.contents.len());
            let mut sub_structs = Vec::with_capacity(msg.contents.len());

            for c in &msg.contents {
                let t_c = CString::new(c.r#type.as_str()).unwrap();
                let txt_c = CString::new(c.text.as_str()).unwrap();
                sub_structs.push(ffi::geniex_VlmContent {
                    type_: t_c.as_ptr(),
                    text: txt_c.as_ptr(),
                });
                sub_types.push(t_c);
                sub_texts.push(txt_c);
            }

            raw_messages.push(ffi::geniex_VlmChatMessage {
                role: role_c.as_ptr(),
                contents: if sub_structs.is_empty() { std::ptr::null_mut() } else { sub_structs.as_ptr() as *mut _ },
                content_count: sub_structs.len() as i64,
            });

            roles.push(role_c);
            content_structs.push(sub_structs);
            type_cstrings.push(sub_types);
            text_cstrings.push(sub_texts);
        }

        RawVlmChatMessages {
            raw_messages,
            _roles: roles,
            _content_structs: content_structs,
            _type_cstrings: type_cstrings,
            _text_cstrings: text_cstrings,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VlmCapabilities {
    pub supports_vision: bool,
    pub supports_audio: bool,
}

impl From<&ffi::geniex_VlmCapabilities> for VlmCapabilities {
    fn from(raw: &ffi::geniex_VlmCapabilities) -> Self {
        Self {
            supports_vision: raw.supports_vision,
            supports_audio: raw.supports_audio,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LlmModelInfo {
    pub vocab_size: i32,
    pub bos_token: i32,
    pub add_bos: i32,
}

impl From<&ffi::geniex_LlmModelInfo> for LlmModelInfo {
    fn from(raw: &ffi::geniex_LlmModelInfo) -> Self {
        Self {
            vocab_size: raw.vocab_size,
            bos_token: raw.bos_token,
            add_bos: raw.add_bos,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolveDeviceInput {
    pub plugin_id: String,
    pub model_name: Option<String>,
    pub mode: Option<String>,
    pub ngl_default: i32,
}

#[derive(Debug, Clone, Default)]
pub struct ResolveDeviceOutput {
    pub device_id: Option<String>,
    pub ngl: i32,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceList {
    pub device_ids: Vec<String>,
    pub device_names: Vec<String>,
}
