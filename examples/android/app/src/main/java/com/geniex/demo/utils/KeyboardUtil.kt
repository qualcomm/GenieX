package com.geniex.demo.utils

import android.content.Context
import android.view.inputmethod.InputMethodManager
import android.widget.EditText

class KeyboardUtil {
    companion object {
        fun hide(editText: EditText) {
            val context = editText.context
            val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(editText.windowToken, 0)
        }
    }
}