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

package com.nexa.demo.listeners

import android.app.AlertDialog
import android.content.DialogInterface
import android.view.View

abstract class CustomDialogInterface : DialogInterface {

    abstract class OnClickListener : View.OnClickListener, DialogInterface.OnClickListener {
        private var dialog: DialogInterface? = null
        protected val SUBMIT: Int = DialogInterface.BUTTON_POSITIVE
        protected val CANCLE: Int = DialogInterface.BUTTON_NEGATIVE

        constructor()

        override fun onClick(v: View?) {
            dialog?.let {
                var positiveBtn: View? = null
                var negativeBtn: View? = null
                var neutralBtn: View? = null
                if (it is AlertDialog) {
                    (dialog as AlertDialog).let {
                        positiveBtn = it.getButton(DialogInterface.BUTTON_POSITIVE)
                        negativeBtn = it.getButton(DialogInterface.BUTTON_NEGATIVE)
                        neutralBtn = it.getButton(DialogInterface.BUTTON_NEUTRAL)
                    }
                } else if (it is androidx.appcompat.app.AlertDialog) {
                    (dialog as androidx.appcompat.app.AlertDialog).let {
                        positiveBtn = it.getButton(DialogInterface.BUTTON_POSITIVE)
                        negativeBtn = it.getButton(DialogInterface.BUTTON_NEGATIVE)
                        neutralBtn = it.getButton(DialogInterface.BUTTON_NEUTRAL)
                    }
                }

                if (v === positiveBtn) {
                    onClick(dialog, DialogInterface.BUTTON_POSITIVE)
                } else if (v === negativeBtn) {
                    onClick(dialog, DialogInterface.BUTTON_NEGATIVE)
                } else if (v === neutralBtn) {
                    onClick(dialog, DialogInterface.BUTTON_NEUTRAL)
                }
            }
        }

        fun resetPositiveButton(dialog: DialogInterface) {
            resetButton(dialog, DialogInterface.BUTTON_POSITIVE)
        }

        private fun resetButton(dialog: DialogInterface, which: Int) {
            this.dialog = dialog
            if (dialog is AlertDialog) {
                dialog.getButton(which)
            } else if (dialog is androidx.appcompat.app.AlertDialog) {
                dialog.getButton(which)
            } else {
                null
            }?.setOnClickListener(this)
        }
    }

    override fun cancel() {
    }

    override fun dismiss() {
    }
}

