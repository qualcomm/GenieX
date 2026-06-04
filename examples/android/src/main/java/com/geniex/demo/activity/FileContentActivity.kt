// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.activity

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import com.gyf.immersionbar.ktx.immersionBar
import com.geniex.demo.databinding.ActivityFileContentBinding
import com.geniex.demo.utils.inflate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class FileContentActivity : Activity() {

    private val binding by inflate<ActivityFileContentBinding>()
    private var filePath: String? = null
    private var promptContent: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        immersionBar {
            statusBarColorInt(Color.WHITE)
            statusBarDarkFont(true)
            fitsSystemWindows(true)
        }
        filePath = intent.getStringExtra(KEY_FILE_PATH)
        promptContent = intent.getStringExtra(KEY_PROMPT_CONTENT)
        
        binding.btnBack.setOnClickListener {
            finish()
        }
        
        // Handle either file path or prompt content
        if (promptContent != null) {
            // Display prompt content directly
            binding.tvContent.text = promptContent
        } else if (filePath != null) {
            // Read file content
            CoroutineScope(Dispatchers.IO).launch {
                val text = File(filePath).readText()
                runOnUiThread {
                    binding.tvContent.text = text
                }
            }
        }
    }

    companion object {
        const val KEY_FILE_PATH = "key_file_path"
        const val KEY_PROMPT_CONTENT = "key_prompt_content"
    }
}