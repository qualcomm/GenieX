// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package service

import "runtime"

type inferenceThreadTask struct {
	run  func()
	done chan struct{}
}

var inferenceThreadTasks = make(chan inferenceThreadTask)

func init() {
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for task := range inferenceThreadTasks {
			func() {
				defer close(task.done)
				task.run()
			}()
		}
	}()
}

// RunOnInferenceThread serializes SDK work on one persistent OS thread. The
// llama_cpp CPU backend uses an OpenMP worker team associated with its calling
// thread; allowing Go to move consecutive requests between OS threads can
// leave the first team parked and make later requests run near single-thread
// speed.
func RunOnInferenceThread(run func()) {
	task := inferenceThreadTask{run: run, done: make(chan struct{})}
	inferenceThreadTasks <- task
	<-task.done
}

// RunOnInferenceThreadResult is the result-returning form of
// RunOnInferenceThread.
func RunOnInferenceThreadResult[T any](run func() (T, error)) (result T, err error) {
	RunOnInferenceThread(func() {
		result, err = run()
	})
	return result, err
}
