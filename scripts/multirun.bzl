"""Cross-platform alternative to rules_multirun.

rules_multirun only emits a bash wrapper, which does not run on Windows.
This macro emits a py_binary wrapper that subprocess-invokes each
dependency via runfiles, so `bazel run` works on Linux, macOS, and Windows.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_python//python:defs.bzl", "py_binary")

_RUNNER = [
    "import subprocess, sys",
    "from python.runfiles import runfiles",
    "r = runfiles.Create()",
    "for rloc in sys.argv[1:]:",
    "    path = r.Rlocation(rloc)",
    "    if path is None:",
    "        sys.exit('multirun: could not locate ' + rloc + ' in runfiles')",
    "    rc = subprocess.call([path])",
    "    if rc != 0:",
    "        sys.exit(rc)",
]

def multirun(name, commands):
    """Runs `commands` sequentially under `bazel run :{name}`.

    Args:
      name: target name.
      commands: list of labels pointing to executable targets.
    """
    runner_src = "_{}_runner.py".format(name)
    write_file(
        name = "_{}_runner".format(name),
        out = runner_src,
        content = _RUNNER,
    )
    py_binary(
        name = name,
        srcs = [runner_src],
        main = runner_src,
        args = ["$(rlocationpath {})".format(c) for c in commands],
        data = commands,
        deps = ["@rules_python//python/runfiles"],
    )
