---
name: coverage
description: Generate an HTML coverage report for all Go tests via bazel + genhtml, then open it in the browser.
---

# Coverage

> **Before rendering, warn the user if `genhtml` is lcov 1.x** (always
> the case on Windows msys2; also on older Linux distros). The output
> is degraded: `LCOV_EXCL_*` markers ignored (cgo helpers in
> `bindings/go/*` show 0%), flat index, all entries as absolute paths.
> Recommend rendering on Linux with lcov 2.x for the clean view.

## 1. Run coverage

```
bazelisk coverage --combined_report=lcov //...
```

Combined LCOV is at
`<bazel-cache>/execroot/_main/bazel-out/_coverage/_coverage_report.dat`
(printed at the end of the run). `//cli/release/...` installer targets
are gated by `target_compatible_with`, so wrong-host toolchains skip
automatically.

## 2. Render HTML

`genhtml` ships with `lcov`. Check the version with `genhtml --version`
to pick the right command path below.

- Linux: `apt install lcov` / `dnf install lcov` / etc. 2.x available
  on Ubuntu 24.04+, Debian trixie+, Fedora 39+, current nixpkgs.
- Windows (msys2): `pacman -S --noconfirm mingw-w64-clang-aarch64-lcov`
  (or matching `mingw-w64-*-lcov`). Only ships 1.16.

### Linux, lcov 2.x (preferred)

```
rm -rf /tmp/coverage-html && mkdir -p /tmp/coverage-html
unset SOURCE_DATE_EPOCH
genhtml --filter region --rc c_file_extensions=c,h,i,C,H,I,icc,cpp,cc,cxx,hh,hpp,hxx,go \
  --prefix "$(pwd)" --hierarchical \
  <lcov-path> -o /tmp/coverage-html
```

- `--filter region` + `--rc c_file_extensions=...,go` together honor
  `// LCOV_EXCL_START/STOP/LINE` markers. Drop either and the markers
  stop working.
- Unset `SOURCE_DATE_EPOCH` (set by nix / dpkg-buildpackage / etc. to
  a fixed epoch, used by lcov as "Test Date"). No-op in normal shells.
- `--prefix "$(pwd)"` anchors relative `SF:` paths to the repo root.
  Without it, lcov 2.x strips the longest common prefix (`cli/`),
  pushing `bindings/go/*` to absolute paths.
- `--hierarchical` renders a directory tree. Drop for a single flat
  index.

### Linux lcov 1.x / Windows msys2 (degraded)

`--filter`, `--rc`, `--hierarchical` are 2.x-only — drop them.
`LCOV_EXCL_*` markers are ignored, index is flat. Use **`--no-prefix`**
(not `--prefix`): the auto prefix detection picks `cli/` as the common
root and renders `bindings/go/*` as absolute paths while everything
else is relative — a half-stripped, inconsistent index. On Windows the
mixed `\` / `/` separators break it further. `--no-prefix` disables
stripping entirely so every entry renders as a full absolute path —
ugly but uniform.

Linux:

```
rm -rf /tmp/coverage-html && mkdir -p /tmp/coverage-html
unset SOURCE_DATE_EPOCH
genhtml --no-prefix <lcov-path> -o /tmp/coverage-html
```

Windows (msys2 bash, **not** PowerShell / cmd — genhtml is a Perl
script and path translation happens inside msys2):

```
cd /c/Users/<you>/path/to/geniex
rm -rf /tmp/coverage-html && mkdir -p /tmp/coverage-html
/clangarm64/bin/genhtml --no-prefix <lcov-path> -o /tmp/coverage-html
```

Use `/tmp` (msys2 maps it to `C:\msys64\tmp\`), **not** `$TEMP` —
`$TEMP` / `$TMP` are unset in msys2 bash, so `-o "$TEMP/coverage-html"`
silently writes to `/coverage-html`. PowerShell's `$env:TEMP` is a
different directory entirely.

## 3. Open the report

- Linux: `xdg-open /tmp/coverage-html/index.html`
- Windows (msys2 bash): `start /tmp/coverage-html/index.html`
  (or open `C:\msys64\tmp\coverage-html\index.html` from Explorer)

## Notes

- For a single package use the target directly,
  e.g. `//cli/internal/types:types_test`.
