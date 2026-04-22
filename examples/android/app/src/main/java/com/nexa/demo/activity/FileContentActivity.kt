// Copyright 2024-2026 Nexa AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.nexa.demo.activity

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import com.gyf.immersionbar.ktx.immersionBar
import com.nexa.demo.databinding.ActivityFileContentBinding
import com.nexa.demo.utils.inflate
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