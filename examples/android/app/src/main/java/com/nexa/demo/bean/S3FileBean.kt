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

package com.nexa.demo.bean

import android.util.Log
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.TimeZone

data class S3FileBean(val url: String, val startDate: String) {



    companion object {
        private const val TAG = "S3FileBean"
        /**
         * Default valid for 5 hours before expiration
         */
        fun isValid(startDate: String): Boolean {
            val allHours = 24
            val delayHours = 5
            try {
                val tz: TimeZone = TimeZone.getTimeZone("UTC")
                val sdf = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'")
                sdf.timeZone = tz
                return Calendar.getInstance(tz).timeInMillis - sdf.parse(startDate).time < (allHours - delayHours) * 60 * 60 * 1000
            } catch (e: Exception) {
            }
            return false
        }
        fun getStartDateFromUrl(url: String): String {
            try {
                val tag = "X-Amz-Date="
                val start = url.indexOf(tag) + tag.length
                return url.substring(start, start + 16)
            } catch (e: Exception) {
                Log.e(TAG, "getStartDateFromUrl failed:${e.message}")
            }
            return ""
        }
    }
}