use crate::ffi;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeniexError {
    CommonUnknown,
    CommonInvalidInput,
    CommonInvalidDevice,
    CommonMemoryAllocation,
    CommonFileNotFound,
    CommonNetwork,
    CommonCancelled,
    CommonNotInitialized,
    CommonAlreadyInitialized,
    CommonAuth,
    CommonHubModelNotFound,
    CommonRateLimited,
    CommonHubServer,
    CommonNotSupported,
    CommonManifestParse,
    CommonChipsetUnavailable,
    CommonParamNotSupported,
    CommonModelLoad,
    CommonModelInvalid,
    CommonPluginLoad,
    CommonPluginInvalid,
    LlmTokenizationFailed,
    LlmTokenizationContextLength,
    LlmGenerationFailed,
    LlmGenerationPromptTooLong,
    VlmImageLoad,
    VlmImageFormat,
    VlmAudioLoad,
    VlmAudioFormat,
    VlmGenerationFailed,
    Unknown(i32),
}

impl GeniexError {
    pub fn check(code: i32) -> Result<()> {
        if code == ffi::geniex_ErrorCode_GENIEX_SUCCESS as i32 {
            Ok(())
        } else {
            Err(Self::from_i32(code))
        }
    }

    pub fn from_i32(code: i32) -> Self {
        if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_UNKNOWN as i32 {
            Self::CommonUnknown
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_INPUT as i32 {
            Self::CommonInvalidInput
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_DEVICE as i32 {
            Self::CommonInvalidDevice
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MEMORY_ALLOCATION as i32 {
            Self::CommonMemoryAllocation
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_FILE_NOT_FOUND as i32 {
            Self::CommonFileNotFound
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NETWORK as i32 {
            Self::CommonNetwork
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CANCELLED as i32 {
            Self::CommonCancelled
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_INITIALIZED as i32 {
            Self::CommonNotInitialized
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_ALREADY_INITIALIZED as i32 {
            Self::CommonAlreadyInitialized
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_AUTH as i32 {
            Self::CommonAuth
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND as i32 {
            Self::CommonHubModelNotFound
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_RATE_LIMITED as i32 {
            Self::CommonRateLimited
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_SERVER as i32 {
            Self::CommonHubServer
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_SUPPORTED as i32 {
            Self::CommonNotSupported
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MANIFEST_PARSE as i32 {
            Self::CommonManifestParse
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE as i32 {
            Self::CommonChipsetUnavailable
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED as i32 {
            Self::CommonParamNotSupported
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_LOAD as i32 {
            Self::CommonModelLoad
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_INVALID as i32 {
            Self::CommonModelInvalid
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_LOAD as i32 {
            Self::CommonPluginLoad
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_INVALID as i32 {
            Self::CommonPluginInvalid
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_FAILED as i32 {
            Self::LlmTokenizationFailed
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH as i32 {
            Self::LlmTokenizationContextLength
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_FAILED as i32 {
            Self::LlmGenerationFailed
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG as i32 {
            Self::LlmGenerationPromptTooLong
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_LOAD as i32 {
            Self::VlmImageLoad
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_FORMAT as i32 {
            Self::VlmImageFormat
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_LOAD as i32 {
            Self::VlmAudioLoad
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_FORMAT as i32 {
            Self::VlmAudioFormat
        } else if code == ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_GENERATION_FAILED as i32 {
            Self::VlmGenerationFailed
        } else {
            Self::Unknown(code)
        }
    }

    pub fn message(&self) -> String {
        let code = match self {
            Self::CommonUnknown => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_UNKNOWN,
            Self::CommonInvalidInput => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_INPUT,
            Self::CommonInvalidDevice => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_DEVICE,
            Self::CommonMemoryAllocation => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MEMORY_ALLOCATION,
            Self::CommonFileNotFound => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
            Self::CommonNetwork => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NETWORK,
            Self::CommonCancelled => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CANCELLED,
            Self::CommonNotInitialized => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_INITIALIZED,
            Self::CommonAlreadyInitialized => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_ALREADY_INITIALIZED,
            Self::CommonAuth => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_AUTH,
            Self::CommonHubModelNotFound => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND,
            Self::CommonRateLimited => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_RATE_LIMITED,
            Self::CommonHubServer => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_SERVER,
            Self::CommonNotSupported => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_SUPPORTED,
            Self::CommonManifestParse => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MANIFEST_PARSE,
            Self::CommonChipsetUnavailable => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE,
            Self::CommonParamNotSupported => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED,
            Self::CommonModelLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_LOAD,
            Self::CommonModelInvalid => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_INVALID,
            Self::CommonPluginLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_LOAD,
            Self::CommonPluginInvalid => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_INVALID,
            Self::LlmTokenizationFailed => ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_FAILED,
            Self::LlmTokenizationContextLength => ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH,
            Self::LlmGenerationFailed => ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_FAILED,
            Self::LlmGenerationPromptTooLong => ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG,
            Self::VlmImageLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_LOAD,
            Self::VlmImageFormat => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_FORMAT,
            Self::VlmAudioLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_LOAD,
            Self::VlmAudioFormat => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_FORMAT,
            Self::VlmGenerationFailed => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_GENERATION_FAILED,
            Self::Unknown(c) => *c,
        };
        unsafe {
            let ptr = ffi::geniex_get_error_message(code);
            if ptr.is_null() {
                format!("Unknown error code ({})", code)
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }
}

impl fmt::Display for GeniexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for GeniexError {}

pub type Result<T> = std::result::Result<T, GeniexError>;
