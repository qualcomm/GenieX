# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess
import sys
import traceback

import psutil
from cases import BaseCase
from scripts import config, log, utils

# plugin, model, testcases
testcases: list[config.case]


def init_benchmark():
    global testcases
    log.print('========= Init Benchmark =========')

    plugins = config.get_plugins()
    log.print(f'Plugins: {plugins}')
    if len(plugins) == 0:
        raise Exception('No supported plugins found')

    testcases = config.get_testcases(plugins)
    # count total Testcases
    log.print(f'Found {len(testcases)} Models with {sum(len(tc[2]) for tc in testcases)} TestCases')
    if len(testcases) == 0:
        raise Exception('No TestCases found')


def check_models():
    global testcases
    log.print('========== Check Models ==========')

    res = utils.execute_geniex(['list'])
    if res.returncode != 0:
        for line in res.stdout.splitlines():
            log.print(line)
        for line in res.stderr.splitlines():
            log.print(line)
        raise RuntimeError(f'Non-zero exit code: {res.returncode}')

    exist_models: set[str] = set()
    for line in res.stdout.splitlines():
        log.print(line)
        fields = line.split('│')
        if len(fields) < 5:
            continue
        name = fields[1].strip()
        quant = fields[3].strip()
        if quant == '':
            exist_models.add(name)
        else:
            for quant in quant.split(','):
                exist_models.add(f'{name}:{quant.strip()}')

    for i, (_, model, type, _) in enumerate(testcases):
        mp = f'{i + 1:0{len(str(len(testcases)))}}/{len(testcases)}'

        if model in exist_models:
            log.print(f'  --> [{mp}] {model} already exists, skip download')
            continue
        log.print(f'  --> [{mp}] {model} not found, downloading...')
        res = utils.execute_geniex(
            [
                'pull',
                model,
                '--model-type',
                type,
            ],
            timeout=3600,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res.returncode != 0:
            raise RuntimeError(f'Failed to download model {model}, exit code: {res.returncode}')


def _parse_stderr(stderr: str) -> tuple[list[str], list[str]]:
    out: list[str] = []
    err: list[str] = []
    for line in stderr.splitlines():
        try:
            json.loads(line)
        except Exception:
            out.append(line)
        else:
            err.append(line)
    return out, err


def _do_case(type: str, model: str, tc: BaseCase, tc_log: pathlib.Path) -> None | str:
    of = None
    try:
        of = open(f'{tc_log}.{type}.log', 'w', encoding='utf-8')
        of.write('===== Output Log =====\n')
        of.flush()
        p = utils.start_geniex([type, model] + tc.param(), debug_log=True, stdout=of)
        _, stderr = p.communicate(timeout=600)

        stdout, stderr = _parse_stderr(stderr)
        of.write('\n====== Debug Log ======\n')
        for line in stdout:
            of.write(f'{line}\n')

        if p.returncode != 0:
            raise RuntimeError(f'Non-zero exit code: {p.returncode}')

        of.write('\n====== Json Log =======\n')
        failed = False
        for line in stderr:
            of.write(f'{line}\n')
            if tc.check(json.loads(line)):
                of.write('  --> Passed\n')
                continue
            else:
                of.write('  --> Failed\n')
                failed = True

        if failed:
            return 'Check Failed'
        else:
            return None

    except Exception as _:
        if of is not None:
            of.write('\n====== Exception Log =======\n')
            of.write(traceback.format_exc())
        return 'Execution Failed'

    finally:
        if of is not None:
            of.close()


def _start_server(tc_log: pathlib.Path):
    of = None
    try:
        of = open(f'{tc_log}.serve.log', 'w', encoding='utf-8')
        of.write('===== Output Log =====\n')
        of.flush()
        return utils.start_geniex(['serve'], debug_log=True, stdout=of, stderr=of)

    except Exception as _:
        if of is not None:
            of.write('\n====== Exception Log =======\n')
            of.write(traceback.format_exc())
            of.close()
        return 'Serve Failed'


def _stop_server():
    for proc in psutil.process_iter():  # pyright: ignore[reportUnknownMemberType]
        if 'geniex-cli' in proc.name() and 'serve' in proc.cmdline():
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
                proc.wait()


def run_benchmark():
    log.print('========== Run Benchmark =========')

    # plugin, model, tc, reason, log_file
    failed_cases: list[tuple[str, str, str, str, str]] = []

    for i, (plugin, model, _, tcs) in enumerate(testcases):
        os.makedirs(log.log_dir / plugin, exist_ok=True)
        mp = f'{i + 1:0{len(str(len(testcases)))}}/{len(testcases)}'
        log.print(f'==> [{mp}] Plugin: {plugin}, Model: {model}')

        for j, tcc in enumerate(tcs):
            tc = tcc()
            tcp = f'{j + 1:0{len(str(len(tcs)))}}/{len(tcs)}'
            tc_log = (
                log.log_dir
                / plugin
                / f'{mp.split("/")[0]}-{tcp.split("/")[0]}-{model.replace("/", "-").replace(":", "-")}-{tc.name()}'
            )

            res = _do_case('infer', model, tc, tc_log)
            if res is None:
                log.print(f'  --> [{mp}][{tcp}] {tc.name()} Infer: Success')
            else:
                log.print(f'  --> [{mp}][{tcp}] {tc.name()} Infer: {res}')
                failed_cases.append((plugin, model, tc.name(), res, str(tc_log) + '.infer.log'))

            # use finally to ensure server is stopped
            tc = tcc()
            try:
                serve = _start_server(tc_log)
                if not isinstance(serve, subprocess.Popen):
                    log.print(f'  --> [{mp}][{tcp}] {tc.name()} Run: {serve}')
                    failed_cases.append((plugin, model, tc.name(), serve, str(tc_log) + '.serve.log'))
                    continue

                res = _do_case('run', model, tc, tc_log)
                if res is None:
                    log.print(f'  --> [{mp}][{tcp}] {tc.name()} Run: Success')
                else:
                    log.print(f'  --> [{mp}][{tcp}] {tc.name()} Run: {res}')
                    failed_cases.append((plugin, model, tc.name(), res, str(tc_log) + '.run.log'))
            finally:
                _stop_server()

    log.print('======== Benchmark Result ========')
    if len(failed_cases) == 0:
        log.print('All TestCases passed')
    else:
        for plugin, model, tc, reason, log_file in failed_cases:
            log.print(f'==> {reason}: [{plugin}] [{model}] [{tc}] {log_file}')
    log.print(f'Logs saved to {log.log_dir}')


def main():
    log.init()
    init_benchmark()
    check_models()
    run_benchmark()


if __name__ == '__main__':
    main()
