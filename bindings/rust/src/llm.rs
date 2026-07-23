use crate::error::{GeniexError, Result};
use crate::ffi;
use crate::types::{
    ChatMessage, GenerationConfig, LlmModelInfo, ModelConfig, ProfileData,
};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

pub struct Llm {
    handle: *mut ffi::geniex_LLM,
}

unsafe impl Send for Llm {}
unsafe impl Sync for Llm {}

impl Llm {
    pub fn create(
        model_path: &str,
        plugin_id: &str,
        config: &ModelConfig,
        model_name: Option<&str>,
        tokenizer_path: Option<&str>,
        device_id: Option<&str>,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_model_name = model_name.map(|s| CString::new(s).unwrap());
        let c_tokenizer_path = tokenizer_path.map(|s| CString::new(s).unwrap());
        let c_device_id = device_id.map(|s| CString::new(s).unwrap());

        let raw_config = config.to_raw();

        let raw_input = ffi::geniex_LlmCreateInput {
            model_name: c_model_name.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            model_path: c_model_path.as_ptr(),
            tokenizer_path: c_tokenizer_path.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            config: raw_config.raw,
            plugin_id: c_plugin_id.as_ptr(),
            device_id: c_device_id.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        };

        let mut handle: *mut ffi::geniex_LLM = ptr::null_mut();
        let code = unsafe { ffi::geniex_llm_create(&raw_input, &mut handle) };
        GeniexError::check(code)?;

        if handle.is_null() {
            Err(GeniexError::CommonMemoryAllocation)
        } else {
            Ok(Self { handle })
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        let code = unsafe { ffi::geniex_llm_reset(self.handle) };
        GeniexError::check(code)
    }

    pub fn save_kv_cache(&mut self, path: &str) -> Result<()> {
        let c_path = CString::new(path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let input = ffi::geniex_KvCacheSaveInput { path: c_path.as_ptr() };
        let mut output = ffi::geniex_KvCacheSaveOutput { reserved: ptr::null_mut() };
        let code = unsafe { ffi::geniex_llm_save_kv_cache(self.handle, &input, &mut output) };
        GeniexError::check(code)
    }

    pub fn load_kv_cache(&mut self, path: &str) -> Result<()> {
        let c_path = CString::new(path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let input = ffi::geniex_KvCacheLoadInput { path: c_path.as_ptr() };
        let mut output = ffi::geniex_KvCacheLoadOutput { reserved: ptr::null_mut() };
        let code = unsafe { ffi::geniex_llm_load_kv_cache(self.handle, &input, &mut output) };
        GeniexError::check(code)
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        tools: Option<&str>,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let raw_messages = ChatMessage::vec_to_raw(messages);
        let c_tools = tools.map(|s| CString::new(s).unwrap());

        let input = ffi::geniex_LlmApplyChatTemplateInput {
            messages: raw_messages.raw_messages.as_ptr() as *mut _,
            message_count: raw_messages.raw_messages.len() as i32,
            tools: c_tools.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            enable_thinking,
            add_generation_prompt,
        };

        let mut output = ffi::geniex_LlmApplyChatTemplateOutput {
            formatted_text: ptr::null_mut(),
        };

        let code = unsafe { ffi::geniex_llm_apply_chat_template(self.handle, &input, &mut output) };
        GeniexError::check(code)?;

        if output.formatted_text.is_null() {
            Ok(String::new())
        } else {
            let result = unsafe { CStr::from_ptr(output.formatted_text).to_string_lossy().into_owned() };
            unsafe { ffi::geniex_free(output.formatted_text as *mut _) };
            Ok(result)
        }
    }

    pub fn generate<F>(
        &mut self,
        prompt: Option<&str>,
        input_ids: Option<&[i32]>,
        config: Option<&GenerationConfig>,
        mut callback: Option<F>,
    ) -> Result<(String, ProfileData)>
    where
        F: FnMut(&str) -> bool,
    {
        let c_prompt = prompt.map(|s| CString::new(s).unwrap());
        let raw_config = config.map(|c| c.to_raw());

        extern "C" fn token_trampoline<F: FnMut(&str) -> bool>(
            token: *const c_char,
            user_data: *mut c_void,
        ) -> bool {
            if token.is_null() || user_data.is_null() {
                return true;
            }
            unsafe {
                let cb = &mut *(user_data as *mut F);
                let s = CStr::from_ptr(token).to_string_lossy();
                cb(&s)
            }
        }

        let (cb_ptr, user_data_ptr): (ffi::geniex_token_callback, *mut c_void) = if let Some(ref mut cb) = callback {
            (Some(token_trampoline::<F>), cb as *mut F as *mut c_void)
        } else {
            (None, ptr::null_mut())
        };

        let (ids_ptr, ids_count) = if let Some(ids) = input_ids {
            (ids.as_ptr(), ids.len() as i32)
        } else {
            (ptr::null(), 0)
        };

        let input = ffi::geniex_LlmGenerateInput {
            prompt_utf8: c_prompt.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            config: raw_config.as_ref().map_or(ptr::null(), |c| &c.raw),
            on_token: cb_ptr,
            user_data: user_data_ptr,
            input_ids: ids_ptr,
            input_ids_count: ids_count,
        };

        let mut output = ffi::geniex_LlmGenerateOutput {
            full_text: ptr::null_mut(),
            profile_data: unsafe { std::mem::zeroed() },
        };

        let code = unsafe { ffi::geniex_llm_generate(self.handle, &input, &mut output) };
        GeniexError::check(code)?;

        let full_text = if output.full_text.is_null() {
            String::new()
        } else {
            let s = unsafe { CStr::from_ptr(output.full_text).to_string_lossy().into_owned() };
            unsafe { ffi::geniex_free(output.full_text as *mut _) };
            s
        };

        let profile_data = ProfileData::from(&output.profile_data);
        Ok((full_text, profile_data))
    }

    pub fn get_model_info(&self) -> Result<LlmModelInfo> {
        let mut raw = unsafe { std::mem::zeroed::<ffi::geniex_LlmModelInfo>() };
        let code = unsafe { ffi::geniex_llm_get_model_info(self.handle, &mut raw) };
        GeniexError::check(code)?;
        Ok(LlmModelInfo::from(&raw))
    }

    pub fn raw_handle(&self) -> *mut ffi::geniex_LLM {
        self.handle
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::geniex_llm_destroy(self.handle);
            }
        }
    }
}
