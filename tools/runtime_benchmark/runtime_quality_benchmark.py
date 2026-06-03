"""Run a prompt suite through both `geniex` and `genie-t2t-run` for the same
genie-formatted model, then write the answers to a CSV ready for scoring.

The geniex side now shells out to the `geniex infer` CLI **once per prompt**
(via `--test-mode`, which emits a JSON record on stderr with Output + Profile).
This pays a per-prompt CLI startup cost but tests exactly the surface that
ships to users — same chat template, same BOS handling, same default flags —
without depending on the Python bindings. The genie side still shells out to
`genie-t2t-run` (which has no Python API and does no chat templating, so it
gets a hand-built formatted prompt).

Usage (minimal — runs the bundled prompt suite, writes results into ./results/):
    python runtime_quality_benchmark.py --geniex-model qualcomm/Qwen3-4B-Instruct-2507

Usage (full):
    python runtime_quality_benchmark.py \
        --prompts testing_prompts.md \
        --geniex-model qualcomm/Qwen3-4B-Instruct-2507 \
        --quant Q4_0 \
        --device npu \
        --genie-config-dir <model-dir-with-genie_config.json> \
        --out results/qwen3_4b.csv

Usage (geniex-only re-run, when only the geniex code changed):
    python runtime_quality_benchmark.py \
        --geniex-model qualcomm/Qwen3-4B-Instruct-2507 \
        --skip-genie

`--skip-genie` skips the genie-t2t-run pass entirely — useful when geniex
has changed but the model under test hasn't, since genie's answers for the
same model are deterministic across runs. The geniex-only output is written
to results/<slug>_geniex_only.csv by default to avoid clobbering a prior
full run; pair the two CSVs at scoring time (see SCORE_WITH_CLAUDE.md).

If --genie-config-dir is omitted, the script asks `geniex list` /
`geniex model` for the cached path. If --out is omitted, the path is derived
from the geniex model name (slashes → underscores) under ./results/.

The script does NOT score answers — scoring is a separate pass (see
SCORE_WITH_CLAUDE.md). Once the run finishes, the script also writes a
companion <out>.answers.json that the scoring agent consumes.

Requires the `geniex` CLI on PATH (`geniex --help` should work). The Python
package is no longer needed.
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
#   "num-tokens-generated": {"value": 96}             (when present)
#
# We read those, normalise TTFT to milliseconds, and (when the token count is
# available) derive a decode-only TPS that can be compared apples-to-apples
# with the geniex side, which reports decode-only TPS by construction.


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


def _value_to_int(field: dict | None) -> int | None:
    if not isinstance(field, dict) or "value" not in field:
        return None
    try:
        return int(field["value"])
    except (TypeError, ValueError):
        return None


@dataclass
class GenieProfile:
    ttft_ms: float | None = None
    # Genie's reported rate. Empirically this is decode/decode-time on a
    # warm path but can include some constant cost on short answers — we
    # surface it as-is and compute a derived decode-only TPS alongside.
    tps_reported: float | None = None
    generated_tokens: int | None = None
    # Wall-clock total time for the GenieDialog_query event, when the
    # profile carries it. Used to back out decode time = total - ttft.
    total_ms: float | None = None


def parse_genie_profile(profile_path: Path) -> GenieProfile:
    """Read the GenieDialog_query event from a genie-t2t-run --profile JSON.

    Pulls TTFT (→ ms), token-generation-rate (as Genie reports it),
    num-tokens-generated, and (if the event carries a duration field) total
    wall time. Missing fields stay ``None`` so a partial profile degrades
    gracefully instead of aborting the run."""
    out = GenieProfile()
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for component in data.get("components", []):
        for event in component.get("events", []):
            if event.get("type") != "GenieDialog_query":
                continue
            out.ttft_ms = _us_value_to_ms(event.get("time-to-first-token"))
            rate = event.get("token-generation-rate")
            if isinstance(rate, dict) and "value" in rate:
                out.tps_reported = float(rate["value"])
            out.generated_tokens = _value_to_int(event.get("num-tokens-generated"))
            # The event itself usually carries a duration via either
            # "duration"/"total-time"/"value" — try a few common names.
            for key in ("duration", "total-time", "value"):
                ms = _us_value_to_ms(event.get(key)) if isinstance(event.get(key), dict) else None
                if ms is not None:
                    out.total_ms = ms
                    break
            return out
    return out


def derive_tps(generated_tokens: int | None, ttft_ms: float | None,
               total_ms: float | None, decode_ms: float | None) -> tuple[float | None, float | None]:
    """Return (tps_decode_only, tps_with_ttft).

    tps_decode_only = (N - 1) / decode_ms — matches geniex's native metric.
    tps_with_ttft   = N / (ttft + decode_ms) — matches what genie-t2t-run's
    `token-generation-rate` reports on short answers.

    Either is None when the inputs needed to compute it aren't available."""
    n = generated_tokens or 0
    if decode_ms is None and total_ms is not None and ttft_ms is not None:
        decode_ms = max(total_ms - ttft_ms, 0.0)
    decode_only = None
    if n > 1 and decode_ms and decode_ms > 0:
        decode_only = (n - 1) / (decode_ms / 1000.0)
    with_ttft = None
    if n > 0:
        denom_ms = None
        if ttft_ms is not None and decode_ms is not None:
            denom_ms = ttft_ms + decode_ms
        elif total_ms is not None:
            denom_ms = total_ms
        if denom_ms and denom_ms > 0:
            with_ttft = n / (denom_ms / 1000.0)
    return decode_only, with_ttft


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
    # tps == tps_decode (kept for backward compat with existing CSVs).
    tps: float | None = None
    # Decode-only throughput: (N - 1) / decode_time. Matches what the
    # geniex pipeline reports natively.
    tps_decode: float | None = None
    # Throughput with TTFT included: N / (ttft + decode_time). Matches what
    # `genie-t2t-run` surfaces in its profile JSON.
    tps_with_ttft: float | None = None
    generated_tokens: int | None = None
    decode_ms: float | None = None


def _csv_num(v: float | None) -> str:
    """Format a metric for the CSV cell ('' when unavailable)."""
    return f"{v:.2f}" if v is not None else ""


def _csv_int(v: int | None) -> str:
    return str(v) if v is not None else ""


def _fmt_ms(v: float | None) -> str:
    return f"{v:6.0f}ms" if v is not None else "    -  "


def _fmt_tps(v: float | None) -> str:
    return f"{v:5.1f}t/s" if v is not None else "    -   "


def run_geniex(
    geniex_model: str,
    quant: str | None,
    device: str | None,
    prompt_text: str,
    max_tokens: int,
    timeout: int,
) -> RunResult:
    """Generate one answer by shelling out to the `geniex infer` CLI.

    The CLI loads the model fresh on every call (process startup), so this is
    slower than the in-process Python API but it's exactly what `geniex infer`
    does — which is the surface developers actually ship. The chat-template +
    BOS handling all live in the CLI, so we just pass the raw user prompt.

    We use --test-mode, which emits a single JSON line to stderr with
    {Input, Output, Profile, Error} — Profile carries TTFT (us), PrefillSpeed,
    DecodingSpeed (toks/sec). We parse that for both the answer and the perf
    metrics. The system prompt is forwarded with -s to match the genie side
    (which builds its hand-built template around the same SYSTEM_PROMPT).
    """
    start = time.time()
    target = geniex_model + (f":{quant}" if quant else "")
    cmd = [
        "geniex",
        "--test-mode",
        "infer",
        target,
        "-s",
        SYSTEM_PROMPT,
        "-p",
        prompt_text,
        "--max-tokens",
        str(max_tokens),
    ]
    if device:
        cmd.extend(["-d", device])
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return RunResult(answer="", seconds=time.time() - start, error="timeout")
    except FileNotFoundError as e:
        raise SystemExit(
            f"ERROR: `geniex` CLI not found on PATH ({e}). Install the geniex "
            "release and ensure `geniex --help` works before running this script."
        ) from e

    payload = _parse_geniex_test_mode(cp.stderr or "")
    if payload is None:
        # No JSON record — propagate stderr/stdout as the error so it lands in
        # the CSV's geniex_error column for triage.
        tail = (cp.stderr or cp.stdout or "")[-200:].strip()
        return RunResult(
            answer="",
            seconds=time.time() - start,
            error=f"exit {cp.returncode}: {tail}" if cp.returncode else "no test-mode payload",
        )

    answer = (payload.get("Output") or "").strip()
    err = payload.get("Error") or None
    if isinstance(err, dict):
        err = err.get("message") or json.dumps(err)
    if cp.returncode != 0 and not answer and not err:
        err = f"exit {cp.returncode}: {(cp.stderr or '')[-200:].strip()}"

    prof = payload.get("Profile") or {}
    ttft_us = prof.get("TTFT") or prof.get("ttft")
    ttft_ms = float(ttft_us) / 1000.0 if ttft_us else None
    tps_decode = prof.get("DecodingSpeed") or prof.get("decoding_speed") or prof.get("decode_speed")
    tps_decode = float(tps_decode) if tps_decode else None
    decode_us = prof.get("DecodeTime") or prof.get("decode_time")
    decode_ms = float(decode_us) / 1000.0 if decode_us else None
    gen_tok = prof.get("GeneratedTokens") or prof.get("generated_tokens")
    gen_tok = int(gen_tok) if gen_tok else None

    # Derive a TTFT-included TPS to match Genie's `token-generation-rate`.
    _, tps_with_ttft = derive_tps(gen_tok, ttft_ms, None, decode_ms)

    return RunResult(
        answer=answer,
        seconds=time.time() - start,
        error=err,
        ttft_ms=ttft_ms,
        tps=tps_decode,
        tps_decode=tps_decode,
        tps_with_ttft=tps_with_ttft,
        generated_tokens=gen_tok,
        decode_ms=decode_ms,
    )


def _parse_geniex_test_mode(stderr: str) -> dict | None:
    """Find the JSON record `geniex --test-mode infer` writes to stderr.

    The CLI prints other log lines too, so we scan from the bottom for the
    last line that parses as a dict containing "Output" / "Profile".
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and ("Output" in obj or "Profile" in obj):
            return obj
    return None


# Cached per-config-dir tokenizer for backfilling Genie's generated-token
# count when the profile JSON omits it. Loaded lazily so the benchmark
# still runs (with `genie_tps_decode` empty, just like before) on hosts
# without the `tokenizers` package installed.
_TOKENIZER_CACHE: dict[Path, object | None] = {}


def _load_tokenizer(config_dir: Path) -> object | None:
    if config_dir in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[config_dir]
    tok_path = config_dir / "tokenizer.json"
    tok: object | None = None
    if tok_path.is_file():
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
            tok = Tokenizer.from_file(str(tok_path))
        except Exception:
            tok = None
    _TOKENIZER_CACHE[config_dir] = tok
    return tok


def _count_tokens(tokenizer: object | None, text: str) -> int | None:
    if tokenizer is None or not text:
        return None
    try:
        return len(tokenizer.encode(text).ids)  # type: ignore[attr-defined]
    except Exception:
        return None


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
    gp = parse_genie_profile(profile_path)
    shutil.rmtree(profile_dir, ignore_errors=True)

    # If genie-t2t-run's profile didn't carry `num-tokens-generated`
    # (depends on QAIRT version), fall back to tokenising the recorded
    # answer with the model's own tokenizer.json — same approach as
    # derive_matched_tps.py, so live and retroactive numbers agree.
    gen_tok = gp.generated_tokens
    if gen_tok is None and cleaned:
        gen_tok = _count_tokens(_load_tokenizer(config_dir), cleaned)

    # Genie's `token-generation-rate` is what we'll treat as `tps_with_ttft`
    # (it's the per-event total-time / token count). We then derive a
    # decode-only number when num-tokens-generated + (total_ms or
    # tps_reported) give us enough to back out decode time.
    decode_ms = None
    if gp.total_ms is not None and gp.ttft_ms is not None:
        decode_ms = max(gp.total_ms - gp.ttft_ms, 0.0)
    elif gp.tps_reported and gen_tok and gp.ttft_ms is not None:
        # Genie's rate ≈ tokens / total_ms. Solve for decode_ms so we can
        # report decode-only TPS.
        total_ms = (gen_tok / gp.tps_reported) * 1000.0
        decode_ms = max(total_ms - gp.ttft_ms, 0.0)
    tps_decode, tps_with_ttft = derive_tps(gen_tok, gp.ttft_ms, gp.total_ms, decode_ms)
    if tps_with_ttft is None:
        tps_with_ttft = gp.tps_reported  # fall back to whatever Genie said
    return RunResult(
        answer=cleaned,
        seconds=time.time() - start,
        error=err,
        ttft_ms=gp.ttft_ms,
        tps=tps_decode if tps_decode is not None else gp.tps_reported,
        tps_decode=tps_decode,
        tps_with_ttft=tps_with_ttft,
        generated_tokens=gen_tok,
        decode_ms=decode_ms,
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
        "--device",
        default=None,
        help="--device passed through to `geniex infer` "
        "(cpu / gpu / npu / hybrid; default: let the CLI pick — npu for QAIRT, "
        "hybrid for llama_cpp). The script does NOT pass anything when this is "
        "unset, so the CLI's own per-plugin default applies.",
    )
    p.add_argument(
        "--quant",
        default=None,
        help="Quantization variant suffix appended to the model name as "
        "<model>:<quant> (e.g. Q4_0). When unset, `geniex infer` picks the "
        "only cached precision or prompts interactively — pass --quant to "
        "make a non-interactive run deterministic.",
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
        default=2048,
        help="Cap on tokens generated per prompt (geniex side, via max_new_tokens; "
        "genie side has no equivalent flag and stops on EOS / context exhaustion). "
        "Set high enough that the geniex side won't truncate before EOS — "
        "otherwise the two sides aren't comparable on length.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-prompt subprocess timeout in seconds. Applies to both the "
        "`geniex infer` and `genie-t2t-run` calls.",
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
    p.add_argument(
        "--skip-genie",
        action="store_true",
        help="Skip the genie-t2t-run pass entirely (only run geniex). Use this "
        "when only geniex code changed — genie's answers for the same model "
        "are deterministic and were captured in a previous full run. The "
        "genie_* columns in the output CSV are left blank, and the default "
        "output path becomes <slug>_geniex_only.csv so it doesn't clobber "
        "the prior full run. Pair the new file with the prior full run when "
        "scoring (see SCORE_WITH_CLAUDE.md).",
    )
    args = p.parse_args()

    # genie_config.json is still needed for the geniex side (we read its bos
    # token to pick the chat template label, even though geniex itself uses
    # the model's own tokenizer template). Skipping genie does not skip this.
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
        suffix = "_geniex_only.csv" if args.skip_genie else ".csv"
        args.out = DEFAULT_RESULTS_DIR / f"{slug}{suffix}"
        print(f"Auto-derived output path: {args.out}", flush=True)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_genie:
        print("--skip-genie: genie-t2t-run pass disabled; only geniex will run.", flush=True)

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
        # `*_tps` is the legacy column (decode-only on the geniex side, the
        # genie-reported rate on the genie side). Kept so existing scoring
        # tooling and historical CSVs still load. The two columns below are
        # the apples-to-apples pair: decode-only on both sides, and total
        # (TTFT + decode) on both sides.
        "genie_tps",
        "geniex_tps",
        "genie_tps_decode",
        "geniex_tps_decode",
        "genie_tps_with_ttft",
        "geniex_tps_with_ttft",
        "genie_generated_tokens",
        "geniex_generated_tokens",
        "genie_decode_ms",
        "geniex_decode_ms",
        "genie_error",
        "geniex_error",
    ]

    def write_all() -> None:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    # The geniex side now shells out to the `geniex infer` CLI per prompt —
    # no in-process model handle to set up or tear down. That makes resume +
    # SIGINT handling trivial (just stop the loop) at the cost of paying the
    # CLI's startup cost on every prompt.
    pending = [pr for pr in prompts if pr.id not in done_ids]
    if pending:
        target = args.geniex_model + (f":{args.quant}" if args.quant else "")
        device_msg = f", device={args.device}" if args.device else ""
        print(f"Driving `geniex infer {target}` (per-prompt subprocess{device_msg}).", flush=True)

    for prompt in prompts:
        if prompt.id in done_ids:
            continue
        # genie-t2t-run needs the hand-built formatted string; the geniex CLI
        # applies the model's own template internally from the raw text.
        genie_formatted = template.format(prompt=prompt.text)
        print(f"\n[{prompt.id:03d}] ({prompt.category}) {prompt.text[:80]}", flush=True)

        gx = run_geniex(
            args.geniex_model,
            args.quant,
            args.device,
            prompt.text,
            args.max_tokens,
            args.timeout,
        )
        print(
            f"  geniex: {gx.seconds:5.1f}s  ttft={_fmt_ms(gx.ttft_ms)}  "
            f"tps={_fmt_tps(gx.tps_decode)} (decode) "
            f"{_fmt_tps(gx.tps_with_ttft)} (w/ttft)  "
            f"err={gx.error or '-'}",
            flush=True,
        )
        if args.skip_genie:
            gn = RunResult(answer="", seconds=0.0)
            print("  genie : skipped (--skip-genie)", flush=True)
        else:
            gn = run_genie(args.genie_config_dir, genie_formatted, args.timeout)
            print(
                f"  genie : {gn.seconds:5.1f}s  ttft={_fmt_ms(gn.ttft_ms)}  "
                f"tps={_fmt_tps(gn.tps_decode)} (decode) "
                f"{_fmt_tps(gn.tps_with_ttft)} (w/ttft)  "
                f"err={gn.error or '-'}",
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
                "genie_tps_decode": _csv_num(gn.tps_decode),
                "geniex_tps_decode": _csv_num(gx.tps_decode),
                "genie_tps_with_ttft": _csv_num(gn.tps_with_ttft),
                "geniex_tps_with_ttft": _csv_num(gx.tps_with_ttft),
                "genie_generated_tokens": _csv_int(gn.generated_tokens),
                "geniex_generated_tokens": _csv_int(gx.generated_tokens),
                "genie_decode_ms": _csv_num(gn.decode_ms),
                "geniex_decode_ms": _csv_num(gx.decode_ms),
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
    answers_path = args.out.parent / (args.out.stem + ".answers.json")
    rows_payload = [
        {
            "id": int(r["id"]),
            "category": r.get("category", ""),
            "prompt": r.get("prompt", ""),
            "genie_answer": r.get("genie_answer", ""),
            "geniex_answer": r.get("geniex_answer", ""),
        }
        for r in rows
    ]
    # Wrap with a meta block so downstream tooling (SCORE_WITH_CLAUDE.md) can
    # tell a geniex-only run apart from a full run without diffing every row.
    # Keep backward compat: a top-level list (the legacy shape) is still
    # readable by the scoring agent — the meta block is purely additive.
    answers_payload: list[dict] | dict
    if args.skip_genie:
        answers_payload = {
            "meta": {
                "model": args.geniex_model,
                "runtimes": ["geniex"],
                "skip_genie": True,
            },
            "rows": rows_payload,
        }
    else:
        answers_payload = rows_payload
    answers_path.write_text(
        json.dumps(answers_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {args.out} ({len(rows)} rows)", flush=True)
    print(f"Wrote {answers_path} (for scoring agent)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
