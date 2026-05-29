---
name: coverage
description: Generate an HTML coverage report for all Go tests via bazel + genhtml, then open it in the browser.
---

# Coverage

Supports Linux and Windows. The bazel step is identical on both.

> **Linux caveat**: confirm `genhtml --version` reports lcov 2.x. lcov
> 1.x (still default on older Debian/Ubuntu) lacks `--filter region` and
> `--hierarchical`, so `// LCOV_EXCL_*` markers are ignored and the
> index is a flat absolute-path list. Upgrade (Ubuntu 24.04+, Debian
> trixie+, Fedora 39+, current nixpkgs all ship 2.x), or accept the
> degraded view.
>
> **Windows caveat**: msys2 only ships lcov 1.16, with the same gaps as
> Linux 1.x — render on Linux for the clean filtered view. When you
> invoke this skill on Windows, surface this caveat to the user before
> generating the report.

## 1. Run coverage

From the repo root:

```
bazelisk coverage --combined_report=lcov //...
```

Bazel auto-derives `--instrumentation_filter` from the target pattern,
and `//cli/release/...` targets are gated by `target_compatible_with`,
so the wrong-host installer toolchains are skipped automatically.

The combined LCOV file path is printed at the end of the run, under
`<bazel-cache>/execroot/_main/bazel-out/_coverage/_coverage_report.dat`.

## 2. Render HTML

`genhtml` is a Perl script that ships with `lcov`.

- Linux: `apt install lcov` / `dnf install lcov` / etc. — must be 2.x for
  `--filter region` to honor `// LCOV_EXCL_*` markers in source.
- Windows (msys2): `pacman -S --noconfirm mingw-w64-clang-aarch64-lcov`
  (or the matching `mingw-w64-*-lcov` for your target). msys2 only
  ships lcov 1.16, which **does not** support `--filter region`, so
  `LCOV_EXCL_*` markers in `bindings/go/*` and friends are ignored —
  cgo helper coverage will show as 0% in the report. For the cleaner
  filtered view, render on Linux.

Output goes to the OS temp dir so it stays out of the repo.

### Linux

Check `genhtml --version` first.

#### lcov 2.x

```
rm -rf /tmp/coverage-html && mkdir -p /tmp/coverage-html
unset SOURCE_DATE_EPOCH
genhtml --filter region --rc c_file_extensions=c,h,i,C,H,I,icc,cpp,cc,cxx,hh,hpp,hxx,go \
  --prefix "$(pwd)" --hierarchical \
  <lcov-path> -o /tmp/coverage-html
```

- `--filter region` honors `// LCOV_EXCL_START` / `// LCOV_EXCL_STOP` /
  `// LCOV_EXCL_LINE` markers. lcov gates this on the file extension,
  so register `.go` via `--rc c_file_extensions=...,go` — drop either
  flag and the markers stop working.
- Reproducible-build environments (`nix shell`, Nixpkgs/Guix sandboxes,
  `dpkg-buildpackage`, etc.) set `SOURCE_DATE_EPOCH` to a fixed epoch
  (often 1980-01-01), which lcov uses as the report "Test Date". Unset
  it. No-op in normal shells.
- `--prefix` is the repo root that LCOV's relative `SF:` paths resolve
  against. Without it, genhtml strips the longest *common* prefix, which
  on this repo is `cli/` — so files outside `cli/` (like `bindings/go/*`)
  fall back to absolute paths and land under `home/<user>/...` in the
  output tree. Setting `--prefix "$(pwd)"` keeps everything aligned.
- `--hierarchical` renders the report as a directory tree (one index
  per level, drill down). Without it, genhtml emits a single flat list
  of every leaf directory, which gets unwieldy at this repo's depth.
  Drop the flag if you want everything on one page.

#### lcov 1.x (degraded)

The 2.x-only flags (`--filter`, `--rc`, `--hierarchical`) error out, so
drop them. `// LCOV_EXCL_*` markers are ignored and the index is flat —
same caveat as Windows.

```
rm -rf /tmp/coverage-html && mkdir -p /tmp/coverage-html
unset SOURCE_DATE_EPOCH
genhtml --prefix "$(pwd)" <lcov-path> -o /tmp/coverage-html
```

### Windows (msys2 bash)

Open the msys2 shell (not PowerShell / cmd — `genhtml` is a Perl script
and the path translation happens inside msys2):

```
cd /c/Users/<you>/path/to/geniex
/clangarm64/bin/genhtml <lcov-path> -o "$TEMP/coverage-html"
```

lcov 1.16 doesn't have `--hierarchical`, and `--prefix` is broken on
mixed Windows path separators, so directory entries render as full
absolute paths. For the cleaner view, render on Linux.

## 3. Open the report

- Linux: `xdg-open /tmp/coverage-html/index.html`
- Windows: `Invoke-Item "$env:TEMP\coverage-html\index.html"` (PowerShell)
  or `start "" "%TEMP%\coverage-html\index.html"` (cmd)

## Notes

- For a single package use the target directly,
  e.g. `//cli/internal/types:types_test`.
