"""Run a prompt suite through both `geniex` and `genie-t2t-run` for the same
genie-formatted model, then write the answers to a CSV ready for scoring.

Usage (minimal — runs the bundled prompt suite, writes results into ./results/):
    python runtime_quality_benchmark.py --geniex-model qualcomm/Qwen3-4B-Instruct-2507

Usage (full):
    python runtime_quality_benchmark.py \
        --prompts testing_prompts.md \
        --geniex-model qualcomm/Qwen3-4B-Instruct-2507 \
        --genie-config-dir <model-dir-with-genie_config.json> \
        --out results/qwen3_4b.csv

If --genie-config-dir is omitted, the script asks `geniex list` /
`geniex model` for the cached path. If --out is omitted, the path is derived
from the geniex model name (slashes → underscores) under ./results/.

The script does NOT score answers — scoring is a separate pass (see
SCORE_WITH_CLAUDE.md). Once the run finishes, the script also writes a
companion <out>.answers.json that the scoring agent consumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
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

# Both runtimes need a fully-formatted prompt (genie-t2t-run does no chat
# templating; geniex's `infer` doesn't wrap the prompt either when given
# `-p` directly). We detect the model family from genie_config.json's
# bos-token to pick a template.

TEMPLATES = {
    "qwen": (
        "<|im_start|>system\nYou are a helpful AI assistant<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant<|eot_id|>"
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


def clean_geniex(text: str) -> str:
    text = strip_ansi(text)
    # Drop spinner-like "loading model..." / "encoding..." status lines and
    # the trailing "— X tok/s — N tok — Y s first token —" timing line.
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if "loading model..." in s or "encoding..." in s:
            continue
        if "tok/s" in s and "first token" in s:
            continue
        # Drop the "> <prompt-echo>" line geniex prints before its output.
        if s.startswith(">"):
            continue
        kept.append(line)
    body = "\n".join(kept).strip()

    # geniex `infer -p <formatted_prompt>` echoes the *whole* chat-templated
    # prompt back into the output (the model sees it, generates "assistant"
    # content, but the harness prints prompt+completion as one stream). Peel
    # off everything through the last assistant header so we keep just the
    # generated answer.
    markers = [
        "<|start_header_id|>assistant<|end_header_id|>",  # llama 3
        "<|im_start|>assistant",                          # qwen
    ]
    for m in markers:
        idx = body.rfind(m)
        if idx != -1:
            body = body[idx + len(m):]
            break
    # Strip stop-token leftovers and leading whitespace/newlines.
    body = body.replace("<|eot_id|>", "").replace("<|im_end|>", "")
    return body.strip()


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


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    answer: str
    seconds: float
    error: str | None = None


def run_geniex(model: str, formatted_prompt: str, max_tokens: int, timeout: int) -> RunResult:
    start = time.time()
    try:
        cp = subprocess.run(
            [
                "geniex",
                "infer",
                model,
                "-p",
                formatted_prompt,
                "--max-tokens",
                str(max_tokens),
                "--skip-update",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return RunResult(answer="", seconds=time.time() - start, error="timeout")
    out = (cp.stdout or "") + (cp.stderr or "")
    cleaned = clean_geniex(out)
    err = None
    if cp.returncode != 0 and not cleaned:
        err = f"exit {cp.returncode}: {out[-200:].strip()}"
    return RunResult(answer=cleaned, seconds=time.time() - start, error=err)


def run_genie(config_dir: Path, formatted_prompt: str, timeout: int) -> RunResult:
    start = time.time()
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
            ],
            capture_output=True,
            text=True,
            cwd=str(config_dir),
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return RunResult(answer="", seconds=time.time() - start, error="timeout")
    out = (cp.stdout or "") + (cp.stderr or "")
    cleaned = clean_genie(out)
    err = None
    if cp.returncode != 0 and not cleaned:
        err = f"exit {cp.returncode}: {out[-200:].strip()}"
    return RunResult(answer=cleaned, seconds=time.time() - start, error=err)


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
    import os

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
        "--out",
        type=Path,
        default=None,
        help="CSV output path (default: results/<slug>.csv next to this script)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Cap on tokens generated per prompt (geniex side; genie side honors EOS via chat template)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-prompt timeout in seconds for each runtime",
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
        "genie_seconds",
        "geniex_seconds",
        "genie_error",
        "geniex_error",
    ]

    def write_all() -> None:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    for prompt in prompts:
        if prompt.id in done_ids:
            continue
        formatted = template.format(prompt=prompt.text)
        print(f"\n[{prompt.id:03d}] ({prompt.category}) {prompt.text[:80]}", flush=True)

        gx = run_geniex(args.geniex_model, formatted, args.max_tokens, args.timeout)
        print(f"  geniex: {gx.seconds:5.1f}s  err={gx.error or '-'}", flush=True)
        gn = run_genie(args.genie_config_dir, formatted, args.timeout)
        print(f"  genie : {gn.seconds:5.1f}s  err={gn.error or '-'}", flush=True)

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
                "genie_seconds": f"{gn.seconds:.2f}",
                "geniex_seconds": f"{gx.seconds:.2f}",
                "genie_error": gn.error or "",
                "geniex_error": gx.error or "",
            }
        )
        # Persist after every prompt — long runs are expensive to lose.
        rows.sort(key=lambda r: int(r["id"]))
        write_all()

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
