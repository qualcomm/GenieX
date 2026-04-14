WINDOWS_CMAKE_OUTS = [
    "common.lib",
    "cpp-httplib.lib",
    "ggml-base.dll",
    "ggml-base.lib",
    "ggml-cpu.dll",
    "ggml-cpu.lib",
    "ggml.dll",
    "ggml.lib",
    "ggml-opencl.dll",
    "ggml-opencl.lib",
    "ggml-hexagon.dll",
    "ggml-hexagon.lib",
    "libggml-htp-v68.so",
    "libggml-htp-v69.so",
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so",
    "libggml-htp.cat",
    "llama.dll",
    "llama.lib",
    "mtmd.dll",
    "mtmd.lib",
]

WINDOWS_RUNTIME_DLLS = [
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml.dll",
    "ggml-opencl.dll",
    "ggml-hexagon.dll",
    "libggml-htp-v68.so",
    "libggml-htp-v69.so",
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so",
    "libggml-htp.cat",
    "llama.dll",
    "mtmd.dll",
]

WINDOWS_BACKEND_RUNTIME_FILES = [
    "ggml-opencl.dll",
    "ggml-hexagon.dll",
    "libggml-htp-v68.so",
    "libggml-htp-v69.so",
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so",
    "libggml-htp.cat",
]

def artifact_labels(repo_prefix, artifacts):
    return ["%s:%s" % (repo_prefix, artifact) for artifact in artifacts]

def windows_copy_cmd(repo_prefix, artifacts):
    commands = []
    for artifact in artifacts:
        commands.append(
            "copy /Y \"$(location %s:%s)\" \"$(@D)\\%s\" >NUL" % (
                repo_prefix,
                artifact,
                artifact,
            ),
        )
    return " && ".join(commands)
