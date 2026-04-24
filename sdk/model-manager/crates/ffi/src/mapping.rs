use std::os::raw::c_char;

use model_manager_core::mapping::resolve_alias;

use crate::types::*;

#[no_mangle]
pub extern "C" fn geniex_model_resolve_alias(
    alias: *const c_char,
    out_full_name: *mut *mut c_char,
) -> i32 {
    ffi_guard(|| {
        if out_full_name.is_null() {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }
        let alias_str = match unsafe { cstr_to_str(alias) } {
            Some(s) => s,
            None => return GENIEX_ERROR_COMMON_INVALID_INPUT,
        };
        match resolve_alias(alias_str) {
            Some(full) => {
                unsafe {
                    *out_full_name = str_to_cptr(&full);
                }
                GENIEX_SUCCESS
            }
            None => GENIEX_ERROR_COMMON_INVALID_INPUT,
        }
    })
}
