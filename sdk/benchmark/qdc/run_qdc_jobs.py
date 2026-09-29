# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Run geniex-bench on a QDC device and render a bench report.

Builds an artifact (SDK pkg + entry script), submits it as a QDC job, downloads
the per-cell JSON geniex-bench emits, and writes a markdown bench report to
GITHUB_STEP_SUMMARY. Linux (QCS9075M, BASH), Windows (SC8380XP, PowerShell), and
Android (SM8850, APPIUM via adb) are implemented.

`--accuracy` swaps the timing sweep for one `geniex-bench --accuracy` pass over
the committed prompt set and harvests the generated text from the device log
into grading items plus a Breeze payload. Linux only for now.
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The QDC SDK is only needed in run mode; render mode (the aggregate job) has no
# wheel installed, so import the shared primitives optionally and fail loudly
# only when run mode uses them.
try:
    import _qdc
except ImportError:
    _qdc = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).parent
MODELS_FILE = HERE / "bench-models.json"
# Host poll and QDC reservation seconds. 330 min leaves headroom for log upload
# under the GitHub-hosted 6h job cap.
JOB_TIMEOUT = 19800

# release_assets.json is published per model on the qualcomm HuggingFace org;
# the download_url values inside it point at the qaihub-public-assets S3 bucket.
HF_BASE = "https://huggingface.co/qualcomm"
# Default precision for qairt release_assets.json lookups.
AIHUB_PRECISION = "w4a16"
# Staged into every artifact and fed to VLM cells; reuses the committed VLM
# e2e fixture (tests/conftest.py TEST_IMAGE_PATH).
TEST_IMAGE = HERE.parents[2] / "cli" / "server" / "docs" / "ui" / "favicon-32x32.png"
# QDC device code -> AI Hub chipset slug (the keys under
# precisions.<precision>.chipset_assets in release_assets.json).
CHIPSET = {
    "QCS8275": "qualcomm-qcs8275",
    "QCS9075M": "qualcomm-qcs9075",
    "SC8380XP": "qualcomm-snapdragon-x-elite",
    "SC8480XP": "qualcomm-snapdragon-x2-elite",
    "X1P42100": "qualcomm-snapdragon-x-plus-8-core",
    "SM8650": "qualcomm-snapdragon-8gen3",
    "SM8750": "qualcomm-snapdragon-8-elite",
    "SM8850": "qualcomm-snapdragon-8-elite-gen5",
}


def platform_for(device: str) -> str:
    if device.startswith("QCS"):
        return "linux"
    if device.startswith("SM"):
        return "android"
    if device.startswith(("SC", "CRD", "X")):
        return "windows"
    raise SystemExit(f"unknown device chipset: {device}")


def _resolve_aihub_url(m: dict, device: str) -> str | None:
    """Resolve a qairt genie bundle download URL from a model's
    release_assets.json, published on the qualcomm HuggingFace org.

    Reads the current schema:
        precisions.<precision>.chipset_assets.<slug>.genie.download_url

    The returned URL points at the qaihub-public-assets S3 bucket. Used only
    for the host-side bench report / chipset probe — the device-side mm pull
    does the actual download via the model's "qualcomm/<id>" alias."""
    slug = CHIPSET.get(device)
    if slug is None:
        return None
    hf_repo = m.get("hf_repo")
    if not hf_repo:
        raise SystemExit(f"{m.get('name')}: missing hf_repo for aihub model")
    url = f"{HF_BASE}/{hf_repo}/resolve/main/release_assets.json"
    with urllib.request.urlopen(url) as r:
        doc = json.load(r)
    precision = m.get("precision", AIHUB_PRECISION)
    chipset_assets = (
        doc.get("precisions", {}).get(precision, {}).get("chipset_assets", {})
    )
    asset = chipset_assets.get(slug)
    if not asset:
        return None
    return asset.get("genie", {}).get("download_url")


def _aihub_chipset_supported(m: dict, device: str) -> bool:
    """Probe AI Hub's release manifest to confirm the model advertises an
    asset for `device`'s chipset slug. Cheap (single JSON over HTTPS) and
    keeps us from emitting a row that the device-side mm pull would just
    error on, which is the only behaviour the host can know up front."""
    if CHIPSET.get(device) is None:
        raise SystemExit(f"no chipset slug for {device}")
    return _resolve_aihub_url(m, device) is not None


def resolve_model_url(m: dict, device: str) -> str | None:
    """Best-effort public download URL for the bench report `Build & models`
    block. None when no asset matches (QAIRT bundles on unsupported chipsets).
    Not used for the actual download — the device-side mm pull does that."""
    if m.get("hub") == "aihub":
        return _resolve_aihub_url(m, device)
    return m.get("url")


def _resolve_draft_model_id(models: list[dict], draft_name: str) -> str:
    for m in models:
        if m["name"] == draft_name:
            return m["model_id"]
    raise SystemExit(f"draft model {draft_name!r} not found in bench-models.json")


def model_rows(models: list[dict], device: str) -> list[str]:
    """One pipe-delimited row per model, consumed by the device-side run
    scripts. Schema:

        name | plugin | csv_devices | model_id | vlm | image
             | spec_type | draft_model_id | draft_tokens

    Trailing three fields carry spec-decoding params or empty strings so
    every script parses the same column count. Entries with empty
    ``devices`` are catalog-only (e.g. spec draft models) and skipped.
    Rows for AI Hub models whose chipset isn't advertised are dropped
    upfront so the device doesn't waste time on a guaranteed-fail pull."""
    rows = []
    for m in models:
        if "model_id" not in m:
            raise SystemExit(f"{m['name']}: missing model_id in bench-models.json")
        if not m["devices"]:
            continue
        if m.get("hub") == "aihub" and not _aihub_chipset_supported(m, device):
            log.warning("no %s asset for %s, skipping", device, m["name"])
            continue
        vlm = "1" if m.get("vlm") else ""
        image = "1" if m.get("image") else ""
        spec = m.get("spec") or {}
        spec_type = spec.get("type", "")
        draft_id = _resolve_draft_model_id(models, spec["draft"]) if spec else ""
        draft_tokens = str(spec.get("n_max", "")) if spec else ""
        rows.append(
            f"{m['name']}|{m['plugin']}|{','.join(m['devices'])}|{m['model_id']}"
            f"|{vlm}|{image}|{spec_type}|{draft_id}|{draft_tokens}"
        )
    return rows


PROMPTS = HERE / "prompts"
ACCURACY_PROMPTS = PROMPTS / "accuracy_prompts.txt"


def load_accuracy_prompts(limit: int = 0) -> list[str]:
    """The committed prompt set, split the way geniex-bench splits it (on lines
    that are exactly `---`). `limit` trims it for cheap runs."""
    segs = [
        s.strip() for s in ACCURACY_PROMPTS.read_text().split("\n---\n") if s.strip()
    ]
    return segs[:limit] if limit else segs


_SEP_RE = re.compile(r"^\[sep \] prompt (\d+)/\d+")
_GEN_RE = re.compile(r"^\[gen \] ?(.*)$")
# geniex-bench prints this once per matrix row, naming the compute unit that
# just finished; a multi-device accuracy run puts more than one in one log.
_OK_RE = re.compile(r"^\[ok  \] \S+\s+plugin=\S+ device=([^\s(]+)")


def decode_log(data: bytes) -> str:
    """Device logs are UTF-8 everywhere except Windows, where PowerShell 5.1's
    Start-Transcript writes UTF-16 with a BOM."""
    for bom, enc in (
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if data.startswith(bom):
            return data.decode(enc, errors="replace")
    return data.decode("utf-8", errors="replace")


def _cell_items(
    blocks: dict[int, list[str]], prompts: list[str], compute: str
) -> list[dict]:
    out = []
    for i, prompt in enumerate(prompts):
        text = "\n".join(blocks.get(i, [])).strip()
        if text:
            out.append({"prompt": prompt, "response": text, "compute": compute})
        else:
            where = f" ({compute})" if compute else ""
            log.warning(
                "no generated text for prompt %d%s: %r", i + 1, where, prompt[:60]
            )
    return out


def parse_items(log_text: str, prompts: list[str]) -> list[dict]:
    """Pair each prompt with the `[gen ]` block geniex-bench printed for it.
    A row is a cell that ends in its own `[ok  ]` line; flush on every one
    instead of assuming there's exactly one, so a multi-device accuracy run
    (more than one compute unit for the same model) keeps every cell's text
    instead of only the first."""
    items: list[dict] = []
    blocks: dict[int, list[str]] = {}
    cur = 0
    for line in log_text.splitlines():
        if m := _OK_RE.match(line):
            items += _cell_items(blocks, prompts, m.group(1))
            blocks, cur = {}, 0
        elif m := _SEP_RE.match(line):
            cur = int(m.group(1)) - 1
        elif m := _GEN_RE.match(line):
            blocks.setdefault(cur, []).append(m.group(1))
    if blocks:
        # No closing [ok  ] seen (e.g. the device crashed mid-cell) -- whatever
        # text made it out is still worth keeping, just unattributed.
        items += _cell_items(blocks, prompts, "")
    return items


GRADE_RULES = PROMPTS / "grade-rules.md"


def render_item_map(items: list[dict]) -> str:
    """The human-facing copy of what was sent for grading -- same prompt and
    response as the payload, but tagged with the cell it withholds, so a
    reader can trace a rating back to a model without the grader ever
    seeing that mapping."""
    out = []
    for i, item in enumerate(items):
        out += [
            f"========== ITEM {i} — {item['cell']} ==========",
            "",
            "--- Prompt ---",
            item["prompt"],
            "--- Response ---",
            item["response"],
            "",
        ]
    return "\n".join(out) + "\n"


def render_payload(items: list[dict]) -> str:
    """The Breeze grading payload: the rubric followed by the numbered items.
    Rides a workflow artifact — an issue body caps at 65536 characters.

    Item numbers are global across cells; which cell an item came from is
    deliberately withheld so the grader cannot anchor on the model."""
    out = [GRADE_RULES.read_text().strip(), ""]
    for i, item in enumerate(items):
        out += [
            f"========== ITEM {i} ==========",
            "",
            "--- Prompt ---",
            item["prompt"],
            "--- Response to grade ---",
            item["response"],
            "--- End of response ---",
            "",
        ]
    out += [
        f"Grade all {len(items)} items above. Emit one markdown table, one row per",
        "item, with the columns Item | Rating | Category | Note. Nothing else.",
    ]
    return "\n".join(out) + "\n"


# --- grading round trip: merge cells -> payload, grader table -> two tables ---

BANDS = (("0-3", 0, 3), ("4-6", 4, 6), ("7-9", 7, 9), ("10", 10, 10))
# Closed vocabulary from prompts/grade-rules.md; anything else counts as other.
CATEGORIES = frozenset(
    (
        "junk-tokens mojibake foreign-script loop truncation "
        "duplicate-word markup misspelling code-error factual reasoning none"
    ).split()
)


def merge_items(items_dir: Path) -> list[dict]:
    """Flatten every generate job's `items.json` into one globally numbered
    list. Each item already carries the cell it came from -- a multi-device
    accuracy run can produce more than one cell per file."""
    merged = []
    for f in sorted(items_dir.rglob("*.json")):
        merged += json.loads(f.read_text())["items"]
    return merged


_GRADE_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*`?([A-Za-z-]+)`?\s*\|\s*(.*?)\s*\|?\s*$"
)


def parse_grades(comment: str) -> dict[int, dict]:
    """Pull `| item | rating | category | note |` rows out of the grader's
    comment, ignoring the header separator and any stray prose."""
    grades = {}
    for line in comment.splitlines():
        if m := _GRADE_ROW.match(line.strip()):
            item, rating = int(m.group(1)), int(m.group(2))
            if 0 <= rating <= 10:
                grades[item] = {
                    "rating": rating,
                    "category": m.group(3).lower(),
                    "note": m.group(4),
                }
    return grades


def _score_table(graded: list[tuple[int, dict, dict]], cells: list[str]) -> list[str]:
    """One row per cell: mean rating and the band histogram. `cells` may
    include ones with no graded items (a generate job that produced
    nothing) -- those get a dash row instead of silently vanishing."""
    per_cell: dict[str, list[int]] = {}
    for _, item, g in graded:
        per_cell.setdefault(item["cell"], []).append(g["rating"])
    lines = [
        "| Cell | Items | Mean | " + " | ".join(b for b, _, _ in BANDS) + " |",
        "|------|------:|-----:|" + "".join("----:|" for _ in BANDS),
    ]
    for cell in cells:
        ratings = per_cell.get(cell)
        if not ratings:
            lines.append(f"| {cell} | 0 | - | " + " | ".join("-" for _ in BANDS) + " |")
            continue
        counts = [sum(1 for r in ratings if lo <= r <= hi) for _, lo, hi in BANDS]
        mean = sum(ratings) / len(ratings)
        lines.append(
            f"| {cell} | {len(ratings)} | {mean:.1f} | "
            + " | ".join(str(c) for c in counts)
            + " |"
        )
    return lines


def _category_table(
    graded: list[tuple[int, dict, dict]], cells: list[str]
) -> list[str]:
    """Deduction counts, cell rows x category columns -- same orientation as
    the score table above it. `none` is a perfect 10, not a deduction."""
    counts: dict[str, dict[str, int]] = {c: {} for c in cells}
    totals: dict[str, int] = {}
    for _, item, g in graded:
        cat = g["category"] if g["category"] in CATEGORIES else "other"
        if cat == "none":
            continue
        counts[item["cell"]][cat] = counts[item["cell"]].get(cat, 0) + 1
        totals[cat] = totals.get(cat, 0) + 1
    if not totals:
        return ["> Every item scored a perfect 10 -- no deductions."]
    cats = sorted(totals, key=lambda c: -totals[c])
    lines = [
        "| Cell | " + " | ".join(f"`{c}`" for c in cats) + " |",
        "|------|" + "".join("---:|" for _ in cats),
    ]
    for cell in cells:
        row = [str(counts[cell][cat]) if cat in counts[cell] else "-" for cat in cats]
        lines.append(f"| {cell} | " + " | ".join(row) + " |")
    return lines


def render_grades(
    items: list[dict], comment: str, expected_cells: list[str] = ()
) -> str:
    """Two tables instead of 100 rows of prose: scores per cell, then a
    breakdown of what the deductions were for.

    `expected_cells` are the device x model matrix's base cell names
    (`{model}-{device}`, no compute suffix -- the caller doesn't know
    ahead of time whether a job would have split into more than one). Any
    not matched by an actual item, exactly or as a `-{compute}` prefix,
    get a dash row: a generate job that produced nothing should show up
    as missing, not vanish from the table."""
    grades = parse_grades(comment)
    graded = [(i, item, grades[i]) for i, item in enumerate(items) if i in grades]
    present = {item["cell"] for item in items}
    missing_cells = [
        c
        for c in expected_cells
        if c not in present and not any(p.startswith(c + "-") for p in present)
    ]
    cells = sorted({item["cell"] for _, item, _ in graded} | set(missing_cells))
    lines = ["## Breeze grading results", ""]
    if not graded and not cells:
        lines += [
            "> No parseable `| Item | Rating | Category | Note |` rows in the",
            "> grader comment — see the issue for the raw agent output.",
            "",
        ]
        return "\n".join(lines)
    missing = len(items) - len(graded)
    lines += _score_table(graded, cells) + [""]
    if missing:
        lines += [f"> {missing} of {len(items)} items came back ungraded.", ""]
    lines += ["### Deductions", ""]
    lines += _category_table(graded, cells) + [""]
    return "\n".join(lines)


DEFAULT_CTX = [512, 1024, 4096]
DEFAULT_TG_PER_CELL = 128


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_sweep(
    ctx_arg: str, pp_arg: str, tg_arg: str
) -> tuple[list[int], list[int], list[int]]:
    """Turn the workflow's raw --ctx/--pp/--tg strings into three parallel
    int lists of equal length. Empty --ctx picks {512, 1024, 4096}; empty
    --tg picks 128 per cell; empty --pp derives ctx-tg per cell."""
    ctx = _parse_int_list(ctx_arg) or list(DEFAULT_CTX)
    tg = _parse_int_list(tg_arg) or [DEFAULT_TG_PER_CELL] * len(ctx)
    pp = _parse_int_list(pp_arg) or [c - t for c, t in zip(ctx, tg)]
    if len(pp) != len(ctx) or len(tg) != len(ctx):
        raise SystemExit(f"--ctx/--pp/--tg length mismatch: ctx={ctx} pp={pp} tg={tg}")
    if any(p < 1 for p in pp):
        raise SystemExit(
            f"derived pp has non-positive value: pp={pp} (ctx={ctx}, tg={tg})"
        )
    return ctx, pp, tg


def _sweep_placeholders(ctx: list[int], pp: list[int], tg: list[int]) -> dict[str, str]:
    """Comma-separated sweep lists shared by both bash and PowerShell templates
    (both parse the same string form)."""
    return {
        "{CTX_LIST}": ",".join(map(str, ctx)),
        "{PP_LIST}": ",".join(map(str, pp)),
        "{TG_LIST}": ",".join(map(str, tg)),
    }


def _apply_substitutions(text: str, subs: dict[str, str]) -> str:
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def build_linux_artifact(
    pkg_dir: Path,
    models: list[dict],
    device: str,
    tmp: Path,
    ctx: list[int],
    pp: list[int],
    tg: list[int],
    accuracy_prompts: list[str] | None = None,
    think: bool = True,
) -> Path:
    """`accuracy_prompts` picks the entry script's mode: None runs the bench
    sweep, a list runs one accuracy pass over exactly those prompts."""
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")

    subs = {
        "{MODELS}": "\n".join(model_rows(models, device)),
        "{CHIPSET}": CHIPSET.get(device, ""),
        "{MODE}": "accuracy" if accuracy_prompts else "bench",
        "{THINK}": "--think" if think else "--no-think",
        **_sweep_placeholders(ctx, pp, tg),
    }
    script = _apply_substitutions((HERE / "linux" / "run_linux.sh").read_text(), subs)
    script_path = stage / "run_linux.sh"
    script_path.write_text(script, newline="\n")
    script_path.chmod(0o755)

    _stage_shared(stage, accuracy_prompts)
    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


def _stage_shared(stage: Path, accuracy_prompts: list[str] | None) -> None:
    """The VLM test image and the prompt set every platform ships. In accuracy
    mode the committed prompt file is replaced by what --prompt-limit kept.

    Prompts stay LF on every platform: geniex-bench tolerates CRLF around the
    `---` separators, but the \\r would otherwise ride into the prompt text."""
    shutil.copy(TEST_IMAGE, stage / "test.png")
    shutil.copytree(PROMPTS, stage / "prompts")
    if accuracy_prompts:
        (stage / "prompts" / ACCURACY_PROMPTS.name).write_text(
            "\n---\n".join(accuracy_prompts) + "\n", newline="\n"
        )


def build_windows_artifact(
    pkg_dir: Path,
    models: list[dict],
    device: str,
    tmp: Path,
    ctx: list[int],
    pp: list[int],
    tg: list[int],
    accuracy_prompts: list[str] | None = None,
    think: bool = True,
) -> Path:
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")

    subs = {
        "{MODELS}": "\n".join(model_rows(models, device)),
        "{CHIPSET}": CHIPSET.get(device, ""),
        "{MODE}": "accuracy" if accuracy_prompts else "bench",
        "{THINK}": "--think" if think else "--no-think",
        **_sweep_placeholders(ctx, pp, tg),
    }
    script = _apply_substitutions(
        (HERE / "windows" / "run_windows.ps1").read_text(), subs
    )
    (stage / "run_windows.ps1").write_text(script, newline="\r\n")

    cert = HERE.parents[2] / ".github" / "certs" / "hexagon" / "ggml-htp-v1.cer"
    shutil.copy(cert, stage / "ggml-htp-v1.cer")

    _stage_shared(stage, accuracy_prompts)
    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


def build_android_artifact(
    pkg_dir: Path,
    models: list[dict],
    device: str,
    tmp: Path,
    ctx: list[int],
    pp: list[int],
    tg: list[int],
    accuracy_prompts: list[str] | None = None,
    think: bool = True,
) -> Path:
    # Phones lack python3/curl, so the appium pytest harness on the QDC host
    # fetches+extracts each model and adb-pushes it, then runs geniex-bench
    # on-device; results land in the device's QDC_logs and are auto-collected.
    # There is no entry script to substitute into, so the mode travels as JSON
    # and the harness skips whichever test the mode did not ask for.
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")
    (stage / "matrix_rows.txt").write_text("\n".join(model_rows(models, device)))
    (stage / "chipset.txt").write_text(CHIPSET.get(device, ""))
    (stage / "params.json").write_text(
        json.dumps(
            {
                "mode": "accuracy" if accuracy_prompts else "bench",
                "think": think,
                "ctx": ctx,
                "tg": tg,
            }
        )
    )
    shutil.copytree(HERE / "tests", stage / "tests")
    shutil.copy(HERE / "tests" / "requirements.txt", stage / "requirements.txt")
    _stage_shared(stage, accuracy_prompts)
    (stage / "pytest.ini").write_text("[pytest]\naddopts = --junitxml=results.xml\n")

    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


ENTRY = {
    "linux": "/bin/bash /data/local/tmp/TestContent/run_linux.sh",
    "windows": "C:\\Temp\\TestContent\\run_windows.ps1",
    "android": None,
}
BUILDERS = {
    "linux": build_linux_artifact,
    "windows": build_windows_artifact,
    "android": build_android_artifact,
}


def download_cells(
    client, job_id: str, tmp: Path, model_names: list[str] | None = None
) -> list[dict]:
    members = _qdc.download_log_members(
        client, job_id, tmp, lambda n: n.endswith(".json")
    )
    return cells_from(members, model_names)


def cells_from(
    members: list[tuple[str, bytes]], model_names: list[str] | None = None
) -> list[dict]:
    """Parse the cell JSONs out of downloaded log members.

    QDC reuses physical hosts across jobs, so the log archive can carry
    stale cell files from earlier sessions in addition to what this job
    actually produced. When ``model_names`` is given, we keep only cells
    whose ``cell_id`` starts with one of those names — cell_id is
    ``{model}-{plugin}-{device}-c{ctx}`` on the device side, so a name
    prefix is enough to disambiguate."""
    cells = [json.loads(data) for name, data in members if name.endswith(".json")]
    if model_names:
        prefixes = tuple(f"{n}-" for n in model_names)
        kept, dropped = [], []
        for c in cells:
            (
                kept if str(c.get("cell_id", "")).startswith(prefixes) else dropped
            ).append(c)
        if dropped:
            log.warning(
                "dropping %d stale cell(s) not in %s: %s",
                len(dropped),
                model_names,
                [c.get("cell_id") for c in dropped],
            )
        cells = kept
    return sorted(cells, key=lambda c: c["cell_id"])


def _fmt_med_sd(agg: dict, key: str) -> str:
    entry = agg.get(key) or {}
    med = entry.get("median")
    sd = entry.get("stdev")
    if med is None:
        return "-"
    if sd is None:
        return f"{med:.1f}"
    return f"{med:.1f} ± {sd:.1f}"


LLAMA_CPP_COMMIT_BASE = "https://github.com/ggml-org/llama.cpp/commit"


_CTX_SUFFIX = re.compile(r"-c(\d+)$")


def _ctx_from_cell(c: dict) -> int:
    """Pull ctx from the `-c{N}` cell_id suffix; fall back to params.n_ctx."""
    m = _CTX_SUFFIX.search(c.get("cell_id") or "")
    if m:
        return int(m.group(1))
    return int((c.get("params") or {}).get("n_ctx") or 0)


def _model_label(c: dict) -> str:
    cid = _CTX_SUFFIX.sub("", c.get("cell_id") or "")
    return cid.removesuffix(f"-{c['plugin']}-{c['device']}")


def detect_geniex_label(sha: str) -> str:
    """Return `<tag> (<sha>)` when the current commit is a release tag, else
    just `<sha>`. Three-layer fallback: explicit env > exact-match git tag >
    bare sha."""
    if env := os.environ.get("GENIEX_RELEASE_TAG"):
        return f"{env} ({sha})"
    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and (tag := r.stdout.strip()):
        return f"{tag} ({sha})"
    return sha or "unknown"


def _pick_field(cells: list[dict], key: str) -> str | None:
    return next((v for c in cells if (v := c.get(key))), None)


def _models_block(models: list[dict], device: str) -> list[str]:
    lines = ["**Models**", ""]
    for m in models:
        url = resolve_model_url(m, device)
        lines.append(f"- {m['name']} ({m['model_id']}): {url or '-'}")
    return lines


def _details_block(
    cells: list[dict], device: str, label: str, models: list[dict] | None
) -> list[str]:
    qairt_v = _pick_field(cells, "qairt_version") or "-"
    llama_v = _pick_field(cells, "llama_cpp_version")
    llama_line = f"[`{llama_v}`]({LLAMA_CPP_COMMIT_BASE}/{llama_v})" if llama_v else "-"
    lines = [
        "<details><summary>Build & models</summary>",
        "",
        "**Versions**",
        "",
        f"- geniex: `{label}`",
        f"- QAIRT: `{qairt_v}`",
        f"- llama.cpp: {llama_line}",
        f"- generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
    ]
    if models:
        lines += _models_block(models, device)
        lines.append("")
    lines += ["</details>", ""]
    return lines


def _is_spec_cell(c: dict) -> bool:
    return bool((c.get("params") or {}).get("spec_type"))


def _render_mtp_table(cells: list[dict], models: list[dict] | None) -> list[str]:
    """Pair each spec cell with its no-spec baseline on (target model_id,
    device, ctx). Emits nothing when no spec cells are present."""
    if not models:
        return []
    spec_entries = [m for m in models if m.get("spec")]
    if not spec_entries:
        return []
    by_name_key: dict[str, dict[tuple[str, int], dict]] = {}
    for c in cells:
        by_name_key.setdefault(_model_label(c), {})[
            (c["device"], _ctx_from_cell(c))
        ] = c
    rows: list[str] = []
    for spec_m in spec_entries:
        baseline = next(
            (
                m
                for m in models
                if m["model_id"] == spec_m["model_id"]
                and not m.get("spec")
                and m.get("devices")
            ),
            None,
        )
        draft_m = next(
            (m for m in models if m["name"] == spec_m["spec"]["draft"]), None
        )
        spec_cells = by_name_key.get(spec_m["name"], {})
        base_cells = by_name_key.get(baseline["name"], {}) if baseline else {}
        for (dev, ctx), sc in sorted(spec_cells.items()):
            agg = sc.get("agg") or {}
            s_dec = (agg.get("decode_tps") or {}).get("median")
            bc = base_cells.get((dev, ctx))
            b_dec = (
                ((bc.get("agg") or {}).get("decode_tps") or {}).get("median")
                if bc
                else None
            )
            uplift = f"{s_dec / b_dec:.2f}x" if s_dec and b_dec else "-"
            p_med = (agg.get("prompt_tokens") or {}).get("median")
            g_med = (agg.get("gen_tokens") or {}).get("median")
            test = (
                f"pp{int(p_med)}+tg{int(g_med)}"
                if p_med is not None and g_med is not None
                else "-"
            )
            rows.append(
                f"| {spec_m['name']} | {draft_m['name'] if draft_m else '-'} | "
                f"{dev} | {ctx} | {test} | "
                f"{_fmt_med_sd(agg, 'decode_tps')} | "
                f"{_fmt_med_sd((bc or {}).get('agg') or {}, 'decode_tps')} | "
                f"{uplift} |"
            )
    if not rows:
        return []
    return [
        "",
        "## MTP (speculative decoding)",
        "",
        "| Target | Draft | Device | Ctx | Test | Decode (mtp) | Decode (baseline) | Uplift |",
        "|--------|-------|--------|----:|------|-------------:|------------------:|-------:|",
        *rows,
    ]


def render(
    cells: list[dict],
    device: str,
    label: str,
    models: list[dict] | None = None,
) -> str:
    lines = [f"## QDC Bench — {device} — {label}", ""]
    lines += _details_block(cells, device, label, models)
    lines += [
        "| Model | Backend | Device | Ctx | ngl | Test | TTFT (ms) | Media enc (ms) | Prefill (tok/s) | Decode (tok/s) |",
        "|-------|---------|--------|----:|----:|------|----------:|---------------:|----------------:|---------------:|",
    ]
    sort_key = lambda c: (_model_label(c), c["plugin"], c["device"], _ctx_from_cell(c))  # noqa: E731
    for c in sorted(cells, key=sort_key):
        if _is_spec_cell(c):
            continue
        agg = c.get("agg") or {}
        params = c.get("params") or {}
        model = _model_label(c)
        ngl_v = params.get("n_gpu_layers")
        ngl = "-" if c["plugin"] == "qairt" or not ngl_v else str(ngl_v)
        ctx = _ctx_from_cell(c)
        ctx_s = str(ctx) if ctx else "-"
        p_med = (agg.get("prompt_tokens") or {}).get("median")
        g_med = (agg.get("gen_tokens") or {}).get("median")
        menc_med = (agg.get("media_ms") or {}).get("median")
        has_media = menc_med is not None and menc_med > 0
        if p_med is not None and g_med is not None:
            test = f"pp{int(p_med)}+tg{int(g_med)}"
        else:
            test = "-"
        media_enc = f"{menc_med:.1f}" if has_media and menc_med is not None else "-"
        lines.append(
            f"| {model} | {c['plugin']} | {c['device']} | {ctx_s} | {ngl} | {test} | "
            f"{_fmt_med_sd(agg, 'ttft_ms')} | {media_enc} | {_fmt_med_sd(agg, 'prefill_tps')} | "
            f"{_fmt_med_sd(agg, 'decode_tps')} |"
        )
    lines += _render_mtp_table(cells, models)
    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as f:
            f.write(text)


def _short_sha() -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )


def _require(value, flag: str):
    if value is None:
        raise SystemExit(f"{flag} is required for this --mode")
    return value


def render_aggregate(cells_dir: Path, device: str) -> int:
    cells = (
        [
            c
            for f in sorted(cells_dir.rglob("*.json"))
            for c in json.loads(f.read_text())
        ]
        if cells_dir.exists()
        else []
    )
    label = detect_geniex_label(_short_sha())
    if not cells:
        write_summary(f"## QDC Bench — {device} — {label}\n\nNo results recovered.\n")
        return 0
    models = json.loads(MODELS_FILE.read_text()) if MODELS_FILE.exists() else None
    write_summary(render(cells, device, label, models))
    return 0


def collect_bench(client, job_id: str, tmp: Path, args, models: list[dict]) -> int:
    """Bench mode: the per-cell timing JSON in the device's results dir."""
    cells = download_cells(client, job_id, tmp, model_names=[m["name"] for m in models])
    if not cells:
        for name, data in _qdc.download_log_members(
            client, job_id, tmp, lambda n: n.endswith((".log", ".stdout", ".txt"))
        ):
            print(f"===== QDC log: {name} =====")
            print(decode_log(data))

    if args.out:
        args.out.write_text(json.dumps(cells))
    if not cells:
        raise SystemExit("no benchmark results recovered from the device")

    # Render this model's own table into its job summary for immediate visibility;
    # the aggregate job later flattens every model's cells into one unified table.
    write_summary(render(cells, args.device, detect_geniex_label(_short_sha()), models))
    return 0


def collect_accuracy(
    client, job_id: str, tmp: Path, args, models: list[dict], prompts: list[str]
) -> int:
    """Accuracy mode: the generated text, which only exists on the device's
    stdout — redirected into the uploaded log. Cells only label the summary.

    One download for both: each download_log_members call refetches the whole
    log archive. QDCDeviceLogs is the phones' logcat/kernel dump, megabytes of
    noise with no bearing on generation."""
    members = _qdc.download_log_members(
        client,
        job_id,
        tmp,
        lambda n: n.endswith((".log", ".json")) and "/QDCDeviceLogs/" not in n,
    )
    text = "\n".join(decode_log(d) for name, d in members if name.endswith(".log"))
    raw = parse_items(text, prompts)
    if not raw:
        print(text)
        # The device leg failed before generating; on Android the reason is in
        # the appium harness's own stdout, which is not one of the logs above.
        for name, data in _qdc.download_log_members(
            client, job_id, tmp, lambda n: n.endswith((".stdout", ".txt", ".xml"))
        ):
            print(f"===== QDC log: {name} =====")
            print(decode_log(data))
        raise SystemExit("no generated text recovered from the device")

    # {model}-{device} disambiguates the same model on two chipsets once cells
    # are merged for grading. A multi-device run (more than one compute unit
    # for this model) also needs the compute unit, or those cells collide too;
    # the common single-device case keeps the plain two-part label.
    multi = len({it["compute"] for it in raw}) > 1
    items = []
    for it in raw:
        cell = f"{args.model_name}-{args.device}"
        if multi and it["compute"]:
            cell += f"-{it['compute']}"
        items.append({"prompt": it["prompt"], "response": it["response"], "cell": cell})
    if args.out:
        args.out.write_text(json.dumps({"items": items}, indent=2))
    print(f"accuracy: {len(items)} items -> {sorted({it['cell'] for it in items})}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "bench",
            "bench_aggregate",
            "accuracy",
            "accuracy_payload",
            "accuracy_report",
        ),
        default="bench",
        help="bench: timing sweep on a device. accuracy: one geniex-bench "
        "--accuracy pass over the committed prompt set. The rest run on the "
        "host over --in-dir artifacts: bench_aggregate renders the bench "
        "table, accuracy_payload builds the grading payload, accuracy_report "
        "renders the grades",
    )
    p.add_argument("--pkg-dir", type=Path)
    p.add_argument("--device", default="QCS9075M")
    p.add_argument("--model-name", help=f"run only this model from {MODELS_FILE.name}")
    p.add_argument(
        "--compute",
        default="",
        help="comma-separated compute filter (cpu/gpu/npu/hybrid); "
        "empty keeps every compute unit declared on the model. In accuracy "
        "mode it overrides the declared units instead of filtering",
    )
    p.add_argument(
        "--ctx",
        default="",
        help="comma-separated ctx sizes (empty = 512,1024,4096)",
    )
    p.add_argument(
        "--pp",
        default="",
        help="comma-separated prefill lengths matching --ctx (empty = ctx-tg per cell)",
    )
    p.add_argument(
        "--tg",
        default="",
        help="comma-separated decode lengths matching --ctx (empty = 128 per cell)",
    )
    p.add_argument(
        "--prompt-limit",
        type=int,
        default=0,
        help="accuracy mode: use only the first N prompts (0 = all)",
    )
    p.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="accuracy mode: keep the model's thinking phase (geniex-bench default)",
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        help="aggregate/payload/report mode: the artifact directory to read",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="where this mode's artifact goes: bench cells, accuracy items, or "
        "the grading payload",
    )
    p.add_argument(
        "--grades-in",
        type=Path,
        help="report mode: the grader's result comment",
    )
    p.add_argument(
        "--expected-cells",
        default="",
        help="report mode: comma-separated `{model}-{device}` base cell names "
        "to show even when the generate job produced no items for them",
    )
    args = p.parse_args()

    # Host-side modes read artifacts the device runs left behind; they need
    # neither QDC nor a device.
    if args.mode == "bench_aggregate":
        return render_aggregate(_require(args.in_dir, "--in-dir"), args.device)
    if args.mode in ("accuracy_payload", "accuracy_report"):
        items = merge_items(_require(args.in_dir, "--in-dir"))
        if not items:
            raise SystemExit(f"no items found under {args.in_dir}")
        if args.mode == "accuracy_payload":
            out = _require(args.out, "--out")
            out.write_text(render_payload(items))
            log.info("payload: %d items -> %s", len(items), out)
            # A fixed sibling name, not a flag -- the workflow's upload step
            # for it (ITEM_MAP_FILE) expects this exact name.
            out.with_name("grade-item-map.md").write_text(render_item_map(items))
        else:
            grades = _require(args.grades_in, "--grades-in")
            expected_cells = [
                c.strip() for c in args.expected_cells.split(",") if c.strip()
            ]
            write_summary(render_grades(items, grades.read_text(), expected_cells))
        return 0

    if _qdc is None:
        raise SystemExit("qualcomm_device_cloud_sdk is required for run mode")
    api_key = os.environ.get("QDC_API_KEY")
    if not api_key:
        raise SystemExit("QDC_API_KEY must be set")
    if not args.pkg_dir:
        raise SystemExit("--pkg-dir is required")

    platform = platform_for(args.device)
    if platform not in BUILDERS:
        raise SystemExit(f"{platform} not implemented yet")

    accuracy = args.mode == "accuracy"
    all_models = json.loads(MODELS_FILE.read_text())
    if args.model_name:
        models = [m for m in all_models if m["name"] == args.model_name]
        if not models:
            raise SystemExit(f"model {args.model_name!r} not in {MODELS_FILE}")
        # Pull in every spec.draft dependency so _resolve_draft_model_id can
        # still find it after --model-name has trimmed the list to one row.
        needed = {m["spec"]["draft"] for m in models if m.get("spec")}
        for name in needed - {m["name"] for m in models}:
            entry = next((m for m in all_models if m["name"] == name), None)
            if entry is None:
                raise SystemExit(f"draft {name!r} not in {MODELS_FILE}")
            models.append(entry)
    else:
        models = all_models

    compute_pick = [c.strip() for c in args.compute.split(",") if c.strip()]
    if accuracy:
        # Accuracy pins the unit outright rather than intersecting: the declared
        # units are the timing matrix, which excludes `hybrid`.
        if compute_pick:
            models = [{**m, "devices": compute_pick} for m in models]
    elif compute_pick:
        kept = []
        for m in models:
            devs = [d for d in m["devices"] if d in compute_pick]
            if not devs:
                log.warning(
                    "%s declares %s, none match --compute=%s, skipping",
                    m["name"],
                    m["devices"],
                    compute_pick,
                )
                continue
            kept.append({**m, "devices": devs})
        models = kept
        if not models:
            raise SystemExit(
                f"no model in {MODELS_FILE} runs any of --compute={compute_pick}"
            )
    if accuracy:
        prompts = load_accuracy_prompts(args.prompt_limit)
        # One pass, not a sweep, and the model's declared sweep is irrelevant:
        # ctx only has to hold one prompt plus the requested generation, and a
        # 512 cell from the timing matrix would truncate a 2048-token answer.
        tg_first = (_parse_int_list(args.tg) or [DEFAULT_TG_PER_CELL])[0]
        ctx_arg = args.ctx.split(",")[0] if args.ctx else str(max(2 * tg_first, 4096))
        tg_arg = str(tg_first)
    else:
        prompts = None
        tg_arg = args.tg
        ctx_arg = args.ctx
        if not ctx_arg:
            # A lone model may carry its own sweep in bench-models.json.
            active = [m for m in models if m.get("devices")]
            if len(active) == 1 and active[0].get("ctx"):
                ctx_arg = ",".join(str(x) for x in active[0]["ctx"])

    ctx_list, pp_list, tg_list = resolve_sweep(ctx_arg, args.pp, tg_arg)
    if accuracy:
        log.info("accuracy: %d prompts, ctx=%s tg=%s", len(prompts), ctx_list, tg_list)
    else:
        log.info("sweep: ctx=%s pp=%s tg=%s", ctx_list, pp_list, tg_list)

    client = _qdc.make_client(api_key)
    target_id = _qdc.resolve_target(client, args.device)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = BUILDERS[platform](
            args.pkg_dir,
            models,
            args.device,
            tmp,
            ctx_list,
            pp_list,
            tg_list,
            accuracy_prompts=prompts,
            think=args.think,
        )
        job_id = _qdc.submit_and_wait(
            client,
            target_id=target_id,
            job_name=f"geniex-{args.mode}-{args.device}",
            platform=platform,
            entry_script=ENTRY[platform],
            zip_path=zip_path,
            timeout=JOB_TIMEOUT,
        )
        if accuracy:
            return collect_accuracy(client, job_id, tmp, args, models, prompts)
        else:
            return collect_bench(client, job_id, tmp, args, models)


if __name__ == "__main__":
    raise SystemExit(main())
