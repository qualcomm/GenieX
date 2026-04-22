package com.geniex.demo

import android.app.Application
import com.hjq.toast.Toaster

class MyApplication: Application() {

    override fun onCreate() {
        super.onCreate()
        Toaster.init(this)
    }
}