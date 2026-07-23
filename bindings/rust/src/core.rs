use crate::error::{GeniexError, Result};
use crate::ffi;
use crate::types::{DeviceList, ResolveDeviceInput, ResolveDeviceOutput};
use std::ffi::{CStr, CString};

pub fn init() -> Result<()> {
    let code = unsafe { ffi::geniex_init() };
    GeniexError::check(code)
}

pub fn deinit() -> Result<()> {
    let code = unsafe { ffi::geniex_deinit() };
    GeniexError::check(code)
}

pub fn version() -> String {
    unsafe {
        let ptr = ffi::geniex_version();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub fn get_plugin_version(plugin_id: &str) -> Option<String> {
    let c_id = CString::new(plugin_id).ok()?;
    unsafe {
        let ptr = ffi::geniex_get_plugin_version(c_id.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

pub fn get_plugin_list() -> Result<Vec<String>> {
    let mut output = ffi::geniex_GetPluginListOutput {
        plugin_ids: std::ptr::null_mut(),
        plugin_count: 0,
    };
    let code = unsafe { ffi::geniex_get_plugin_list(&mut output) };
    GeniexError::check(code)?;

    let mut result = Vec::new();
    if !output.plugin_ids.is_null() && output.plugin_count > 0 {
        let slice = unsafe { std::slice::from_raw_parts(output.plugin_ids, output.plugin_count as usize) };
        for &ptr in slice {
            if !ptr.is_null() {
                let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                result.push(s);
            }
        }
        unsafe {
            ffi::geniex_free(output.plugin_ids as *mut _);
        }
    }
    Ok(result)
}

pub fn get_device_list(plugin_id: &str) -> Result<DeviceList> {
    let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
    let input = ffi::geniex_GetDeviceListInput {
        plugin_id: c_plugin_id.as_ptr(),
    };
    let mut output = ffi::geniex_GetDeviceListOutput {
        device_ids: std::ptr::null_mut(),
        device_names: std::ptr::null_mut(),
        device_count: 0,
    };

    let code = unsafe { ffi::geniex_get_device_list(&input, &mut output) };
    GeniexError::check(code)?;

    let mut device_ids = Vec::new();
    let mut device_names = Vec::new();

    if output.device_count > 0 {
        if !output.device_ids.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(output.device_ids, output.device_count as usize) };
            for &ptr in slice {
                if !ptr.is_null() {
                    device_ids.push(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() });
                }
            }
            unsafe {
                ffi::geniex_free(output.device_ids as *mut _);
            }
        }
        if !output.device_names.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(output.device_names, output.device_count as usize) };
            for &ptr in slice {
                if !ptr.is_null() {
                    device_names.push(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() });
                }
            }
            unsafe {
                ffi::geniex_free(output.device_names as *mut _);
            }
        }
    }

    Ok(DeviceList {
        device_ids,
        device_names,
    })
}

pub fn resolve_device(input: &ResolveDeviceInput) -> Result<ResolveDeviceOutput> {
    let c_plugin_id = CString::new(input.plugin_id.as_str()).map_err(|_| GeniexError::CommonInvalidInput)?;
    let c_model_name = input.model_name.as_ref().map(|s| CString::new(s.as_str()).unwrap());
    let c_mode = input.mode.as_ref().map(|s| CString::new(s.as_str()).unwrap());

    let raw_input = ffi::geniex_ResolveDeviceInput {
        plugin_id: c_plugin_id.as_ptr(),
        model_name: c_model_name.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        mode: c_mode.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        ngl_default: input.ngl_default,
    };

    let mut raw_output = ffi::geniex_ResolveDeviceOutput {
        device_id: std::ptr::null_mut(),
        ngl: 0,
        warning: std::ptr::null_mut(),
    };

    let code = unsafe { ffi::geniex_resolve_device(&raw_input, &mut raw_output) };
    GeniexError::check(code)?;

    let device_id = if !raw_output.device_id.is_null() {
        let s = unsafe { CStr::from_ptr(raw_output.device_id).to_string_lossy().into_owned() };
        unsafe { ffi::geniex_free(raw_output.device_id as *mut _) };
        Some(s)
    } else {
        None
    };

    let warning = if !raw_output.warning.is_null() {
        let s = unsafe { CStr::from_ptr(raw_output.warning).to_string_lossy().into_owned() };
        unsafe { ffi::geniex_free(raw_output.warning as *mut _) };
        Some(s)
    } else {
        None
    };

    Ok(ResolveDeviceOutput {
        device_id,
        ngl: raw_output.ngl,
        warning,
    })
}

pub fn set_log_callback(callback: ffi::geniex_log_callback) -> Result<()> {
    let code = unsafe { ffi::geniex_set_log(callback) };
    GeniexError::check(code)
}

pub fn register_plugin(
    plugin_id_func: ffi::geniex_plugin_id_func,
    create_func: ffi::geniex_create_plugin_func,
) -> Result<()> {
    let code = unsafe { ffi::geniex_register_plugin(plugin_id_func, create_func) };
    GeniexError::check(code)
}

pub fn free_ptr(ptr: *mut std::os::raw::c_void) {
    if !ptr.is_null() {
        unsafe {
            ffi::geniex_free(ptr);
        }
    }
}
