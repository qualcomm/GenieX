"""Run a prompt suite through both `geniex` and `genie-t2t-run` for the same
genie-formatted model, then write the answers to a CSV ready for scoring.

The geniex side runs **in-process** via the `geniex` Python API: the model is
loaded once (`AutoModelForCausalLM.from_pretrained`) and reused across every
prompt — KV cache reset between prompts — which is both faster and closer to
how the runtime is actually embedded than re-spawning the `geniex infer` CLI
per prompt. Each prompt goes through the model's own chat template
(`apply_chat_template`) before `generate`, mirroring exactly what the Go CLI
`infer` command does. The genie side still shells out to `genie-t2t-run`
(which has no Python API and does no chat templating, so it gets a
hand-built formatted prompt).

Usage (minimal — runs the bundled prompt suite, writes results into ./results/):
    python runtime_quality_benchmark.py --geniex-model qualcomm/Qwen3-4B-Instruct-2507

Usage (full):
    python runtime_quality_benchmark.py \
        --prompts testing_prompts.md \
        --geniex-model qualcomm/Qwen3-4B-Instruct-2507 \
        --device-map auto \
        --genie-config-dir <model-dir-with-genie_config.json> \
        --out results/qwen3_4b.csv

If --genie-config-dir is omitted, the script asks `geniex list` /
`geniex model` for the cached path. If --out is omitted, the path is derived
from the geniex model name (slashes → underscores) under ./results/.

The script does NOT score answers — scoring is a separate pass (see
SCORE_WITH_CLAUDE.md). Once the run finishes, the script also writes a
companion <out>.answers.json that the scoring agent consumes.

Requires the `geniex` Python package to be importable (pip install geniex);
the bundled CLI installs it, so anywhere `geniex infer` runs this does too.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

CATEGORY_RE = re.compile(r"^\s*#\s*---\s*(.+?)\s*---\s*$")
PROMPT_RE = re.compile(r"^\s*-\s+(.*?)\s*$")


@dataclass
class Prompt:
    id: int
    category: str
    text: str


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    category = ""
    pid = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        cat_m = CATEGORY_RE.match(raw)
        if cat_m:
            category = cat_m.group(1).strip()
            continue
        if raw.lstrip().startswith("#"):
            continue
        m = PROMPT_RE.match(raw)
        if not m:
            continue
        text = m.group(1).strip()
        # Strip surrounding quotes if present (the source file uses them
        # whenever a bullet contains a comma or apostrophe).
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1]
        pid += 1
        prompts.append(Prompt(id=pid, category=category, text=text))
    return prompts


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

# System prompt shared by both runtimes. Matches the QAIRT plugin's
# kDefaultSystemPrompt (sdk/plugins/qairt/src/llm.cpp) so the geniex side
# (which applies the model's real chat template) and the genie side (which
# uses the hardcoded template below) see the same system message.
SYSTEM_PROMPT = "You are a helpful AI assistant."

# Only genie-t2t-run needs a hand-built prompt: it does NO chat templating and
# takes a fully-formatted string. The geniex side does NOT use these — it
# calls the model's own chat template via apply_chat_template (see
# run_geniex), exactly like the Go CLI `infer` command. We detect the model
# family from genie_config.json's bos-token to pick the genie template.

TEMPLATES = {
    "qwen": (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def detect_template(genie_config: dict) -> str:
    bos = genie_config.get("dialog", {}).get("context", {}).get("bos-token")
    # Qwen3 / Qwen2 uses 151643. Llama 3 uses 128000.
    if bos == 128000:
        return "llama3"
    return "qwen"


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\[\?[0-9;]*[a-z]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_genie(text: str) -> str:
    """Extract the assistant response from genie-t2t-run output.

    genie-t2t-run prints headers, then `[BEGIN]: <answer>[END]`. The answer can
    span many lines (the [BEGIN]/[END] markers literally bracket it).
    """
    text = strip_ansi(text)
    m = re.search(r"\[BEGIN\]:(.*?)\[END\]", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # No [END] (model truncated by max-tokens or context exhaustion).
    m = re.search(r"\[BEGIN\]:(.*)$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# genie-t2t-run writes profiling data to the file given by `--profile FILE` as
# JSON (artifact_type "GENIE_PROFILE"). The per-query metrics live in the
# `GenieDialog_query` event of the `dialog` component:
#
#   "time-to-first-token":  {"value": 155464, "unit": "us"}
#   "token-generation-rate":{"value": 13.07,  "unit": "toks/sec"}
#
# We read those two fields and normalise TTFT to milliseconds.


def _us_value_to_ms(field: dict | None) -> float | None:
    """Convert a {'value': N, 'unit': 'us'|'ms'|'s'} profile field to ms."""
    if not isinstance(field, dict) or "value" not in field:
        return None
    value = float(field["value"])
    unit = str(field.get("unit", "us")).lower()
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value  # already ms (or unknown — leave as-is)


def parse_genie_profile(profile_path: Path) -> tuple[float | None, float | None]:
    """Return ``(ttft_ms, tps)`` from a genie-t2t-run --profile JSON file.

    Scans every component's events for a `GenieDialog_query` and pulls
    `time-to-first-token` (→ ms) and `token-generation-rate` (toks/sec).
    Returns ``None`` for any field the file doesn't carry, so a missing or
    malformed profile degrades gracefully instead of aborting the run."""
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    for component in data.get("components", []):
        for event in component.get("events", []):
            if event.get("type") != "GenieDialog_query":
                continue
            ttft_ms = _us_value_to_ms(event.get("time-to-first-token"))
            rate = event.get("token-generation-rate")
            tps = float(rate["value"]) if isinstance(rate, dict) and "value" in rate else None
            return ttft_ms, tps
    return None, None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    answer: str
    seconds: float
    error: str | None = None
    # Time-to-first-token in milliseconds and decode throughput in tokens/sec.
    # None when the runtime didn't report them (e.g. on error/timeout).
    ttft_ms: float | None = None
    tps: float | None = None


def _csv_num(v: float | None) -> str:
    """Format a metric for the CSV cell ('' when unavailable)."""
    return f"{v:.2f}" if v is not None else ""


def _fmt_ms(v: float | None) -> str:
    return f"{v:6.0f}ms" if v is not None else "    -  "


def _fmt_tps(v: float | None) -> str:
    return f"{v:5.1f}t/s" if v is not None else "    -   "


def load_geniex_model(model: str, device_map: str):
    """Load the geniex model once via the Python API. Returns the model handle
    (``GeniexLLM``) to reuse across prompts. Importing geniex is deferred to
    here so `--help` and prompt-parsing work without the package installed."""
    try:
        from geniex import AutoModelForCausalLM
    except ImportError as e:
        raise SystemExit(
            "ERROR: the `geniex` Python package is not importable "
            f"({e}). Install it (pip install geniex) and retry."
        ) from e
    return AutoModelForCausalLM.from_pretrained(model, device_map=device_map)


def run_geniex(model_handle, prompt_text: str, max_tokens: int) -> RunResult:
    """Generate one answer in-process, mirroring the Go CLI `infer` path:
    build a [system, user] chat history, run it through the model's *own*
    chat template (apply_chat_template), then generate from the formatted
    text. This is important — for QAIRT, apply_chat_template also stages the
    system prompt / first-turn KV prefix in the plugin, so feeding a
    pre-formatted string straight to generate() would bypass that and use a
    template that may not match the model's real one.

    KV cache is reset before each call so prompts don't bleed into one
    another (QAIRT reset() also re-arms first-turn system-prompt staging)."""
    start = time.time()
    try:
        model_handle.reset()
        formatted = model_handle.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            add_generation_prompt=True,
        )
        out = model_handle.generate(formatted, max_new_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001 — record per-prompt failures, keep going
        return RunResult(answer="", seconds=time.time() - start, error=str(e)[-200:])
    # GenerateOutput splits reasoning (<think>…</think>) from the answer and
    # trims stop tokens, so no banner/echo cleaning is needed. genie-t2t-run
    # returns the model's full raw output, so re-attach the thinking block to
    # keep the two answer columns comparable for scoring.
    answer = out.text or ""
    if out.thinking:
        answer = f"<think>{out.thinking}</think>\n{answer}"
    # ProfileData.ttft is microseconds; decode_speed is tokens/sec.
    prof = out.profile
    ttft_ms = prof.ttft / 1000.0 if prof and prof.ttft else None
    tps = prof.decode_speed if prof and prof.decode_speed else None
    return RunResult(
        answer=answer.strip(),
        seconds=time.time() - start,
        ttft_ms=ttft_ms,
        tps=tps,
    )


def run_genie(config_dir: Path, formatted_prompt: str, timeout: int) -> RunResult:
    start = time.time()
    # genie-t2t-run writes its profiling JSON to the path given via --profile,
    # but it *refuses to overwrite* an existing file. So we make a fresh temp
    # DIRECTORY per call and point --profile at a not-yet-created file inside
    # it (absolute path, so it's unaffected by the cwd=config_dir we run from).
    profile_dir = Path(tempfile.mkdtemp(prefix="genie_profile_"))
    profile_path = profile_dir / "profile.log"
    # genie-t2t-run resolves ctx-bins / tokenizer relative to cwd, so we
    # must invoke it from the model directory.
    try:
        cp = subprocess.run(
            [
                "genie-t2t-run",
                "-c",
                "genie_config.json",
                "-p",
                formatted_prompt,
                "--profile",
                str(profile_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(config_dir),
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(profile_dir, ignore_errors=True)
        return RunResult(answer="", seconds=time.time() - start, error="timeout")
    out = (cp.stdout or "") + (cp.stderr or "")
    cleaned = clean_genie(out)
    err = None
    if cp.returncode != 0 and not cleaned:
        err = f"exit {cp.returncode}: {out[-200:].strip()}"
    ttft_ms, tps = parse_genie_profile(profile_path)
    shutil.rmtree(profile_dir, ignore_errors=True)
    return RunResult(
        answer=cleaned,
        seconds=time.time() - start,
        error=err,
        ttft_ms=ttft_ms,
        tps=tps,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = SCRIPT_DIR / "testing_prompts.md"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def slugify_model(name: str) -> str:
    """qualcomm/Qwen3-4B-Instruct-2507 -> qwen3_4b_instruct_2507."""
    last = name.rsplit("/", 1)[-1]
    return last.replace("-", "_").replace(".", "_").lower()


def find_genie_config_dir(geniex_model: str) -> Path | None:
    """Look up where `geniex pull` cached the model so genie-t2t-run can
    read its genie_config.json. Tries the standard cache layout used on
    Snapdragon X Elite Windows hosts and Linux QDC images."""
    candidates: list[Path] = []
    # Windows default cache (matches what `geniex pull` populates).
    home = Path.home()
    candidates.append(home / ".cache" / "geniex" / "models" / geniex_model)
    # GENIEX_DATADIR override.
    env = os.environ.get("GENIEX_DATADIR")
    if env:
        candidates.append(Path(env) / "models" / geniex_model)
    for c in candidates:
        if (c / "genie_config.json").is_file():
            return c
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare geniex vs genie-t2t-run answer quality on a fixed prompt suite.",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help=f"Prompt-suite markdown (default: {DEFAULT_PROMPTS.name} alongside this script)",
    )
    p.add_argument(
        "--geniex-model",
        required=True,
        help="Model name as known to `geniex list` (e.g. qualcomm/Qwen3-4B-Instruct-2507)",
    )
    p.add_argument(
        "--genie-config-dir",
        type=Path,
        default=None,
        help="Directory containing genie_config.json + ctx-bins for genie-t2t-run "
        "(default: auto-discover under ~/.cache/geniex/models/<geniex-model>)",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to geniex's AutoModelForCausalLM "
        "(auto / cpu / gpu / npu / hybrid / <plugin>:<device>; default: auto). "
        "For a cached QAIRT bundle, 'auto' resolves to qairt:NPU via the "
        "manifest's plugin_id; pass 'npu' to be explicit.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path (default: results/<slug>.csv next to this script)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Cap on tokens generated per prompt (geniex side, via max_new_tokens; "
        "genie side honors EOS via chat template)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-prompt timeout in seconds for the genie-t2t-run subprocess "
        "(the in-process geniex side has no subprocess timeout)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only run the first N prompts (useful for smoke tests)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompts whose id already appears in --out",
    )
    args = p.parse_args()

    if args.genie_config_dir is None:
        found = find_genie_config_dir(args.geniex_model)
        if found is None:
            print(
                f"ERROR: could not locate genie_config.json for {args.geniex_model}. "
                "Pass --genie-config-dir explicitly, or run `geniex pull` first.",
                file=sys.stderr,
            )
            return 2
        args.genie_config_dir = found
        print(f"Auto-discovered genie config dir: {args.genie_config_dir}", flush=True)

    if args.out is None:
        slug = slugify_model(args.geniex_model)
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        args.out = DEFAULT_RESULTS_DIR / f"{slug}.csv"
        print(f"Auto-derived output path: {args.out}", flush=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}", flush=True)

    cfg_path = args.genie_config_dir / "genie_config.json"
    if not cfg_path.is_file():
        print(f"ERROR: missing {cfg_path}", file=sys.stderr)
        return 2
    genie_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    template_key = detect_template(genie_cfg)
    template = TEMPLATES[template_key]
    print(f"Using chat template: {template_key}", flush=True)

    done_ids: set[int] = set()
    rows: list[dict] = []
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                try:
                    done_ids.add(int(row["id"]))
                except (KeyError, ValueError):
                    pass
        print(f"Resume: {len(done_ids)} prompts already in {args.out}", flush=True)

    fieldnames = [
        "id",
        "category",
        "prompt",
        "genie_answer",
        "geniex_answer",
        "genie_score",
        "geniex_score",
        "note",
        "genie_ttft_ms",
        "geniex_ttft_ms",
        "genie_tps",
        "geniex_tps",
        "genie_error",
        "geniex_error",
    ]

    def write_all() -> None:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    # Load the geniex model once (skip if every remaining prompt is already
    # done on a --resume run).
    pending = [pr for pr in prompts if pr.id not in done_ids]
    model_handle = None
    if pending:
        print(f"Loading geniex model {args.geniex_model} (device_map={args.device_map})...", flush=True)
        model_handle = load_geniex_model(args.geniex_model, args.device_map)

    try:
        for prompt in prompts:
            if prompt.id in done_ids:
                continue
            # genie-t2t-run needs the hand-built formatted string; geniex
            # applies the model's own template internally from the raw text.
            genie_formatted = template.format(prompt=prompt.text)
            print(f"\n[{prompt.id:03d}] ({prompt.category}) {prompt.text[:80]}", flush=True)

            gx = run_geniex(model_handle, prompt.text, args.max_tokens)
            print(
                f"  geniex: {gx.seconds:5.1f}s  ttft={_fmt_ms(gx.ttft_ms)}  "
                f"tps={_fmt_tps(gx.tps)}  err={gx.error or '-'}",
                flush=True,
            )
            gn = run_genie(args.genie_config_dir, genie_formatted, args.timeout)
            print(
                f"  genie : {gn.seconds:5.1f}s  ttft={_fmt_ms(gn.ttft_ms)}  "
                f"tps={_fmt_tps(gn.tps)}  err={gn.error or '-'}",
                flush=True,
            )

            rows.append(
                {
                    "id": prompt.id,
                    "category": prompt.category,
                    "prompt": prompt.text,
                    "genie_answer": gn.answer,
                    "geniex_answer": gx.answer,
                    "genie_score": "",
                    "geniex_score": "",
                    "note": "",
                    "genie_ttft_ms": _csv_num(gn.ttft_ms),
                    "geniex_ttft_ms": _csv_num(gx.ttft_ms),
                    "genie_tps": _csv_num(gn.tps),
                    "geniex_tps": _csv_num(gx.tps),
                    "genie_error": gn.error or "",
                    "geniex_error": gx.error or "",
                }
            )
            # Persist after every prompt — long runs are expensive to lose.
            rows.sort(key=lambda r: int(r["id"]))
            write_all()
    finally:
        if model_handle is not None:
            model_handle.close()

    # Companion JSON: minimal payload the scoring agent reads. Same basename
    # as the CSV, with `.answers.json` appended so `<slug>.csv` /
    # `<slug>.answers.json` stay paired.
    answers_path = args.out.with_suffix("").with_suffix(".answers.json")
    if str(answers_path) == str(args.out):
        answers_path = args.out.parent / (args.out.stem + ".answers.json")
    answers = [
        {
            "id": int(r["id"]),
            "category": r.get("category", ""),
            "prompt": r.get("prompt", ""),
            "genie_answer": r.get("genie_answer", ""),
            "geniex_answer": r.get("geniex_answer", ""),
        }
        for r in rows
    ]
    answers_path.write_text(
        json.dumps(answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {args.out} ({len(rows)} rows)", flush=True)
    print(f"Wrote {answers_path} (for scoring agent)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
