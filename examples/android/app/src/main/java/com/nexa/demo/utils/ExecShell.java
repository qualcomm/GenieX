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

package com.nexa.demo.utils;
/**
 * Created by fuli.niu on 2016/8/19.
 */

import android.util.Log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

/**
 * Created by fuli.niu 2016/8/19
 */
public class ExecShell {
    private static String LOG_TAG = ExecShell.class.getName();

    public enum SHELL_CMD {
        check_su_binary(new String[]{"/system/xbin/which", "su"});
        String[] command;

        SHELL_CMD(String[] command) {
            this.command = command;
        }
    }

    public ArrayList<String> executeCommand(String[] commands) {
        String line = null;
        ArrayList<String> fullResponse = new ArrayList<String>();
        Process localProcess = null;
        try {
            localProcess = Runtime.getRuntime().exec(commands);
        } catch (Exception e) {
            Log.e("nfl","Command line execution failed");
            e.printStackTrace();
            return null;
        }
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(localProcess.getOutputStream()));
        BufferedReader in = new BufferedReader(new InputStreamReader(localProcess.getInputStream()));
        try {
            while ((line = in.readLine()) != null) {
                fullResponse.add(line);
            }
        } catch (Exception e) {
            Log.e("nfl", "Command line result processing failed");
            e.printStackTrace();
        }
        return fullResponse;
    }

    public ArrayList<String> executeCommand(SHELL_CMD shellCmd) {
        return executeCommand(shellCmd.command);
    }
}
