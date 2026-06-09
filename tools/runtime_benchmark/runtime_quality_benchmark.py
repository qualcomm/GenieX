"""Run a prompt suite through both `geniex` and `genie-t2t-run` for the same
genie-formatted model, then write the answers to a CSV ready for scoring.

The geniex side runs **in-process via the geniex Python bindings** (pybind):
the model is loaded once and reused across every prompt, with the KV cache
reset between prompts so each answer is independent. This is much faster than
shelling out per prompt and exercises the same chat-template + generation path
the bindings ship to users. For each prompt we build a [system, user] message
list, call the model's own `apply_chat_template` (so geniex applies the real
chat template, BOS handling and default flags), then call `generate`. The
genie side still shells out to `genie-t2t-run` (which has no Python API and
does no chat templating, so it gets a hand-built formatted prompt).

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

Requires the `geniex` Python package importable (`import geniex` should work)
for the geniex side, and the `genie-t2t-run` CLI on PATH for the genie side
(skip it with --skip-genie).
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
# GeniexRunner.run), exactly like the Go CLI `infer` command. We detect the
# model family from genie_config.json's bos-token to pick the genie template.

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


# The geniex side has no native "unlimited" max_tokens: the SDK treats
# max_tokens <= 0 as "use a small default" (128/512), not "no cap". To match
# the genie side — which has no token cap and stops only on EOS / context
# exhaustion — we pass a sentinel larger than any model's context window when
# the user asks for no limit (--max-tokens 0). The plugin's decode loop only
# uses this as an upper bound (no preallocation), and stops earlier on EOS or
# context_length, so an oversized value is safe.
UNLIMITED_MAX_TOKENS = 1_000_000


class GeniexRunner:
    """In-process geniex runner: loads the model once, reuses it per prompt.

    Drives the geniex Python bindings (pybind) directly instead of shelling
    out to the CLI. The model is loaded a single time in :meth:`__init__` and
    every :meth:`run` call resets the KV cache, applies the model's own chat
    template to a ``[system, user]`` message list, optionally prepends a BOS
    token (only when one is passed via ``--bos-token``), then generates. This
    mirrors the surface the bindings ship (same template + default flags) while
    being far faster than a per-prompt subprocess.
    """

    def __init__(
        self,
        geniex_model: str,
        quant: str | None,
        device: str | None,
        bos_token: str | None,
    ) -> None:
        try:
            import geniex  # noqa: PLC0415 — imported lazily so --skip side still runs
        except ImportError as e:
            raise SystemExit(
                f"ERROR: could not import the `geniex` Python package ({e}). "
                "Install the geniex wheel (pip install geniex) before running "
                "this script — the geniex side now runs in-process via pybind."
            ) from e
        self._geniex = geniex
        self.model_name = geniex_model
        self.bos_token = bos_token
        # device_map='auto' lets the SDK pick its per-plugin default (npu for
        # QAIRT, hybrid for llama_cpp) — same default the CLI applies when no
        # -d is passed. A non-None device overrides that.
        device_map = device or "auto"
        try:
            self.model = geniex.AutoModelForCausalLM.from_pretrained(
                geniex_model,
                quant=quant,
                device_map=device_map,
            )
        except Exception as e:  # noqa: BLE001 — surface load failures verbatim
            raise SystemExit(f"ERROR: failed to load {geniex_model} via geniex: {e}") from e
        meta = getattr(self.model, "_meta", None) or {}
        where = meta.get("backend") or device_map
        if meta.get("device"):
            where = f"{where}:{meta['device']}"
        self.where = where

    def last_raw_prompt(self) -> str | None:
        """The exact prompt string fed to ``generate`` for the most recent
        :meth:`run` (chat-templated, with the BOS prefix already applied).
        Useful for logging / spot-checking the template + BOS handling."""
        return getattr(self, "_last_raw_prompt", None)

    def run(self, prompt_text: str, max_tokens: int, timeout: int) -> RunResult:
        start = time.time()
        # max_tokens <= 0 means "no cap": let geniex run to its natural EOS /
        # context-exhaustion stop, matching the genie side. The SDK has no
        # unlimited sentinel of its own (it would otherwise fall back to a
        # small default), so we substitute a value past any context window.
        effective_max_tokens = max_tokens if max_tokens > 0 else UNLIMITED_MAX_TOKENS
        # Reset KV cache + sampler so each prompt is independent (the model is
        # reused across prompts, unlike a fresh-process CLI call).
        try:
            self.model.reset()
        except Exception:  # noqa: BLE001 — non-fatal; first prompt has clean state anyway
            pass

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        try:
            # geniex applies the model's OWN chat template here — same BOS
            # handling and default flags as `geniex infer`. We then optionally
            # prepend a BOS token, but only when one was passed via --bos-token.
            raw_prompt = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:  # noqa: BLE001
            return RunResult(answer="", seconds=time.time() - start, error=f"chat_template: {e}")
        if self.bos_token and not raw_prompt.startswith(self.bos_token):
            raw_prompt = self.bos_token + raw_prompt
        self._last_raw_prompt = raw_prompt

        # Stream so we can enforce the per-prompt timeout: the streamer drives
        # generation on a worker thread and exposes cancel(), which asks the C
        # decode loop to stop at the next token boundary (same mechanism the
        # CLI uses on Ctrl-C). We drain chunks with a deadline and cancel if we
        # blow past it, so a wedged prompt can't stall the whole suite.
        try:
            streamer = self.model.generate(raw_prompt, max_new_tokens=effective_max_tokens, stream=True)
        except Exception as e:  # noqa: BLE001
            return RunResult(answer="", seconds=time.time() - start, error=str(e))

        chunks: list[str] = []
        timed_out = False
        deadline = start + timeout
        gen_thread = getattr(streamer, "_thread", None)
        try:
            for chunk in streamer:
                chunks.append(chunk)
                if time.time() > deadline:
                    timed_out = True
                    streamer.cancel()
                    break
        except Exception as e:  # noqa: BLE001 — generation raised on the worker thread
            return RunResult(answer="".join(chunks).strip(), seconds=time.time() - start, error=str(e))

        if timed_out:
            # Give the cancelled worker a moment to publish its final profile,
            # then report a timeout (with whatever partial text we collected).
            if gen_thread is not None:
                gen_thread.join(timeout=30)
            answer = "".join(chunks).strip()
            return RunResult(answer=answer, seconds=time.time() - start, error="timeout")

        out = streamer.output
        if out is None:
            return RunResult(
                answer="".join(chunks).strip(),
                seconds=time.time() - start,
                error="no output from streamer",
            )
        prof = out.profile
        # geniex splits the model's output: `out.text` is the answer with the
        # reasoning removed, and `out.thinking` holds the `<think>` block's
        # contents (see GenerateOutput.from_raw in the bindings). Reconstruct
        # the full emitted string by wrapping the reasoning back in <think>…
        # </think> and prepending it, so geniex_answer matches what the model
        # actually generated — and stays apples-to-apples with the genie side,
        # which records the raw output (it does no such stripping).
        answer = (out.text or "").strip()
        thinking = (getattr(out, "thinking", None) or "").strip()
        if thinking:
            answer = f"<think>\n{thinking}\n</think>\n\n{answer}"

        # ProfileData reports microseconds; convert to the ms the CSV expects.
        ttft_ms = prof.ttft / 1000.0 if prof.ttft else None
        decode_ms = prof.decode_time / 1000.0 if prof.decode_time else None
        tps_decode = prof.decode_speed or None
        gen_tok = prof.generated_tokens or None
        # Derive a TTFT-included TPS to match Genie's `token-generation-rate`.
        _, tps_with_ttft = derive_tps(gen_tok, ttft_ms, None, decode_ms)

        return RunResult(
            answer=answer,
            seconds=time.time() - start,
            error=None,
            ttft_ms=ttft_ms,
            tps=tps_decode,
            tps_decode=tps_decode,
            tps_with_ttft=tps_with_ttft,
            generated_tokens=gen_tok,
            decode_ms=decode_ms,
        )

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            try:
                model.close()
            except Exception:  # noqa: BLE001
                pass


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
        help="device_map passed to AutoModelForCausalLM.from_pretrained "
        "(cpu / gpu / npu / hybrid / <plugin>[:<device>]; default: 'auto', "
        "which lets the SDK pick its per-plugin default — npu for QAIRT, "
        "hybrid for llama_cpp).",
    )
    p.add_argument(
        "--quant",
        default=None,
        help="Quantization variant passed to from_pretrained(quant=...) "
        "(e.g. Q4_0). When unset, geniex resolves the only cached precision; "
        "pass --quant to make the choice deterministic when several are cached.",
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
        default=0,
        help="Cap on tokens generated per prompt on the geniex side. Default 0 "
        "= no cap: geniex runs to its natural EOS / context-exhaustion stop, "
        "matching the genie side (which has no token cap). Pass a positive "
        "value to truncate (e.g. for quick smoke tests). The genie side has no "
        "equivalent flag and always stops on EOS / context exhaustion.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-prompt timeout in seconds. Applies to both the in-process "
        "geniex `generate` call and the `genie-t2t-run` subprocess.",
    )
    p.add_argument(
        "--bos-token",
        default=None,
        help="BOS token to prepend to the geniex chat-templated prompt. Off by "
        "default — the geniex side relies on the model's own chat template. "
        "Pass a string (e.g. '<|endoftext|>') to force a leading BOS for a "
        "QAIRT bundle whose context-binary expects one but whose chat template "
        "doesn't emit it.",
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

    # genie_config.json locates the cached model and supplies the genie-side
    # hand-built template (its bos-token field picks llama3 vs qwen). The
    # geniex side no longer reads it — it applies the model's own template via
    # the bindings — but we still require it so a missing/un-pulled model fails
    # fast here rather than deep in genie-t2t-run. Skipping genie keeps this.
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

    # BOS token to prepend on the geniex side. Off unless the user passes
    # --bos-token; otherwise the geniex side relies on the model's own chat
    # template (the same path the bindings ship to users).
    bos_token = args.bos_token
    if bos_token:
        print(f"geniex: prepending BOS token {bos_token!r} to the templated prompt", flush=True)

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

    # The geniex side now runs in-process via the Python bindings: load the
    # model once here and reuse it for every prompt (KV cache reset between
    # prompts). Only construct the runner when there's pending work, so a
    # fully-resumed run doesn't pay the load cost.
    pending = [pr for pr in prompts if pr.id not in done_ids]
    runner: GeniexRunner | None = None
    if pending:
        target = args.geniex_model + (f":{args.quant}" if args.quant else "")
        device_msg = f", device={args.device}" if args.device else ""
        print(f"Loading geniex model `{target}` in-process (pybind{device_msg}) ...", flush=True)
        runner = GeniexRunner(args.geniex_model, args.quant, args.device, bos_token)
        print(f"  loaded ({runner.where})", flush=True)

    try:
        for prompt in prompts:
            if prompt.id in done_ids:
                continue
            # genie-t2t-run needs the hand-built formatted string; the geniex
            # side applies the model's own template internally (see GeniexRunner).
            genie_formatted = template.format(prompt=prompt.text)
            print(f"\n[{prompt.id:03d}] ({prompt.category}) {prompt.text[:80]}", flush=True)

            assert runner is not None  # pending is non-empty here
            gx = runner.run(prompt.text, args.max_tokens, args.timeout)
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
    finally:
        if runner is not None:
            runner.close()

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
