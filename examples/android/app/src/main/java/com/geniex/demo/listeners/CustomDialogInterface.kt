package com.geniex.demo.listeners

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

