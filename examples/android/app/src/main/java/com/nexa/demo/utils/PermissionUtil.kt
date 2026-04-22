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

package com.nexa.demo.utils

import android.Manifest
import android.content.Context
import android.content.DialogInterface
import android.content.DialogInterface.BUTTON_POSITIVE
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat

class PermissionUtil {
    companion object {

        fun requestManageStoragePermission(activity: ComponentActivity) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                try {
                    val intent =
                        Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.data = Uri.parse("package:${activity.packageName}")
                    activity.startActivity(intent)
                } catch (e: Exception) {
                    val intent =
                        Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    activity.startActivity(intent)
                }
            } else {
                activity.registerForActivityResult(
                    ActivityResultContracts.RequestPermission()
                ) {}
            }
        }

        fun showRequestManageStoragePermissionDialog(activity: ComponentActivity) {
            val onClickListener = DialogInterface.OnClickListener { dialog, which ->
                when (which) {
                    BUTTON_POSITIVE -> {
                        requestManageStoragePermission(activity = activity)
                    }

                    else -> {}
                }
                dialog?.dismiss()
            }
            AlertDialog.Builder(activity)
                .setMessage("Index files need MANAGE_EXTERNAL_STORAGE permission, please agree it.")
                .setNegativeButton("cancel", onClickListener)
                .setPositiveButton("sure", onClickListener)
                .show()
        }

        fun checkManageStoragePermission(context: Context): Boolean {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
    }
}