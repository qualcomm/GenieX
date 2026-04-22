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

import android.app.Activity
import android.app.Dialog
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleObserver
import androidx.lifecycle.OnLifecycleEvent
import androidx.viewbinding.ViewBinding
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty

class KotlinUtil {
    companion object {
        const val TAG = "KotlinUtil"
    }
}

inline fun <reified VB : ViewBinding> Activity.inflate() =
    lazy(LazyThreadSafetyMode.NONE) {
        inflateBinding<VB>(layoutInflater).apply { setContentView(root) }
    }

inline fun <T : ViewBinding> Activity.viewBinding(crossinline bindingInflater: (LayoutInflater) -> T) =
    lazy(LazyThreadSafetyMode.NONE) {
        val invoke = bindingInflater.invoke(layoutInflater)
        setContentView(invoke.root) //可选
        invoke
    }

inline fun <reified VB : ViewBinding> Dialog.inflate() = lazy {
    inflateBinding<VB>(layoutInflater).apply { setContentView(root) }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified VB : ViewBinding> inflateBinding(layoutInflater: LayoutInflater) =
    VB::class.java.getMethod("inflate", LayoutInflater::class.java)
        .invoke(null, layoutInflater) as VB

inline fun <reified VB : ViewBinding> Fragment.bindView() =
    FragmentBindingDelegate(VB::class.java)

class FragmentBindingDelegate<VB : ViewBinding>(private val clazz: Class<VB>) :
    ReadOnlyProperty<Fragment, VB> {

    private var isInitialized = false
    private var _binding: VB? = null
    private val binding: VB get() = _binding!!

    override fun getValue(thisRef: Fragment, property: KProperty<*>): VB {
        if (!isInitialized) {
            thisRef.viewLifecycleOwner.lifecycle.addObserver(object : LifecycleObserver {
                @OnLifecycleEvent(Lifecycle.Event.ON_DESTROY)
                fun onDestroyView() {
                    Log.d(KotlinUtil.TAG, "KotlinUtil.FragmentBindingDelegate.ON_DESTROY $thisRef")
                    _binding = null
                }
            })
            _binding = clazz.getMethod("bind", View::class.java)
                .invoke(null, thisRef.view) as VB
            isInitialized = true
        } else {
            if (_binding == null) {
                _binding = clazz.getMethod("bind", View::class.java)
                    .invoke(null, thisRef.view) as VB
            }
        }
        return binding
    }
}

