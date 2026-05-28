# Scoring runtime-quality benchmark answers (instructions for Claude Code)

You — the Claude Code agent reading this — are the **scoring** half of
the workflow. The collection half (`RUN_ON_QDC.md`) ran the prompt
suite on a QDC device and produced two paired files:

```
results/<slug>.csv               # 12-column raw results
results/<slug>.answers.json      # [{id, category, prompt, genie_answer, geniex_answer}, ...]
```

Your job is to read the answers, score every (genie_answer, geniex_answer)
pair against the rubric, and emit the final 7-column scored CSV the user
will hand off downstream.

## Inputs

1. `results/<slug>.answers.json` — the only file you need to read for
   scoring decisions. It contains all 100 prompts and both runtimes'
   answers.
2. `scoring_rubric.txt` — the rubric (also reproduced below).
3. `results/<slug>.csv` — the full row data. You will overwrite the
   `genie_score`, `geniex_score`, and `note` columns when you're done,
   but everything else (timing, errors) stays intact.

## Rubric

Each prompt yields a score per runtime, drawn from `{0, 2, 7, 10}`:

- **0** — complete jibberish (output is unreadable / not in the right
  language at all / nothing approximating a response).
- **2** — catastrophic failure: repeated words, jibberish words, the
  answer is grammatically broken, or it confidently states a flatly
  wrong fact that ruins the response.
- **7** — sound grammar, but strange or noticeably wrong content.
  Examples: a working code snippet with a logic bug, a coherent
  paragraph that contains a fabricated fact, a translation that's
  understandable but mistranslates a word.
- **10** — no issues. The answer is correct, well-formed, and on-topic.

There are no other allowed scores. If something feels like a 5, push
yourself to decide whether it's closer to "strange content" (7) or
"catastrophic" (2).

### Special cases that come up

- **Output cut off mid-sentence at the token cap.** If the body up to
  the cut-off was correct, score 7 (not 10). If the model ran on past
  EOS and the visible content was already wrong, score 2.
- **Trick / classic puzzles** (`missing dollar`, `all but 9 die`,
  `5 machines / 5 widgets`, the burning ropes, the three labelled
  boxes). These are intentionally counter-intuitive; getting the
  textbook-wrong answer is a 2 because the model confidently asserts
  a wrong number.
- **Constraint-following prompts** (e.g., "exactly 50 words", "use
  only consonant-starting words", "exactly 5 numbered points"). Score
  on a sliding scale: meet the constraint and the content → 10;
  meet the content but miss the constraint → 7; miss both or produce
  garbled output → 2.
- **Disagreement between the two runtimes is normal**, and is the
  signal you're trying to surface. Score each runtime independently —
  don't try to keep totals balanced.

## Workflow

1. **Load the answers JSON.** Use the Read tool. The JSON may be too
   big for a single read; if so, split into chunks of ~5 prompts each
   (write helper files under `%TEMP%`) and read each chunk in turn.
2. **Score every prompt.** Hold the rubric in mind and write a
   `<slug>_scoring.json` file under `results/` of the form
   ```json
   [
     {"id": 1, "genie_score": 7, "geniex_score": 7,
      "note": "Both: sound grammar but factual error in X."},
     ...
   ]
   ```
   Notes are optional but encouraged when the score isn't 10/10 — they
   make divergences understandable later.
3. **Merge into the final CSV.** Read `results/<slug>.csv`, fill in
   the `genie_score`, `geniex_score`, and `note` columns from your
   scoring JSON, and write back the same 12-column CSV. Then derive
   the **7-column scored CSV** the user actually wants:
   ```
   id, prompt, genie_answer, geniex_answer, genie_score, geniex_score, note
   ```
   Save it as `results/<slug>_scored.csv`. That's the deliverable.
4. **Report totals.** Print the per-runtime totals (out of 1000),
   averages, score distribution (`{10: N, 7: N, 2: N, 0: N}`), and
   how many of the 100 prompts had divergent scores. Highlight any
   `0` or `2` rows for quick scan.

A working example of the JSON-merge step (use as a template):

```python
PYTHONIOENCODING=utf-8 python -c "
import csv, json, sys
sys.stdout.reconfigure(encoding='utf-8')

slug = '<slug>'
csv_path = f'results/{slug}.csv'
score_path = f'results/{slug}_scoring.json'
out_path = f'results/{slug}_scored.csv'

rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
scoring = {s['id']: s for s in json.load(open(score_path, encoding='utf-8'))}

g, x = 0, 0
for r in rows:
    s = scoring[int(r['id'])]
    r['genie_score'] = s['genie_score']
    r['geniex_score'] = s['geniex_score']
    r['note'] = s['note']
    g += s['genie_score']; x += s['geniex_score']

# Rewrite full 12-col with scores filled in.
full_cols = ['id','category','prompt','genie_answer','geniex_answer',
             'genie_score','geniex_score','note',
             'genie_seconds','geniex_seconds','genie_error','geniex_error']
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=full_cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, '') for k in full_cols})

# 7-col deliverable.
short_cols = ['id','prompt','genie_answer','geniex_answer',
              'genie_score','geniex_score','note']
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=short_cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, '') for k in short_cols})

print(f'genie  {g}/1000 avg {g/100:.2f}')
print(f'geniex {x}/1000 avg {x/100:.2f}')
print(f'wrote {out_path}')
"
```

## Where prior runs live

The `results/` directory already contains scored runs you can use as
calibration references when scoring a new model:

- `qwen3_4b_instruct_2507_scored.csv` — Qwen3-4B-Instruct-2507 on
  Snapdragon X Elite. genie 9.54 avg, geniex 9.13 avg.
- `llama_v3_2_1b_instruct_scored.csv` — Llama-3.2-1B-Instruct on
  Snapdragon X Elite. genie 7.08 avg, geniex 7.06 avg.

The matching `_scoring.json` files alongside each CSV record the
per-row notes from those scoring passes — useful if a new run happens
to repeat the same prompt+answer and you want to keep notes consistent.

## What NOT to do

- **Don't re-run the model.** You are scoring pre-collected answers;
  do not invoke `geniex infer` or `genie-t2t-run` on the local machine.
- **Don't invent scores other than 0/2/7/10.**
- **Don't edit the answer text** in the CSV. Score it as-is, even if
  the chat-template strip left a trailing `<|im_end|>` or similar —
  that's a runtime artifact, not a model failure.
- **Don't delete or rewrite the `.answers.json`** after scoring. Keep
  it as the immutable input record.
