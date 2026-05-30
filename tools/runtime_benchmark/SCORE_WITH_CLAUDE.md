# Scoring runtime-quality benchmark answers (instructions for Claude Code)

You — the Claude Code agent reading this — are the **scoring** half of
the workflow. The collection half (`RUN_ON_QDC.md`) ran the prompt
suite on a QDC device and produced two paired files:

```
results/<slug>.csv               # raw results (answers, perf, errors)
results/<slug>.answers.json      # [{id, category, prompt, genie_answer, geniex_answer}, ...]
```

Your job is to read the answers, score every (genie_answer, geniex_answer)
pair against the rubric, and emit the final 7-column scored CSV the user
will hand off downstream.

## Inputs

There are two shapes of input the user may hand you. The first is the
default; the second comes up after a geniex-only re-run.

### Shape A — single full run (the common case)

1. `results/<slug>.answers.json` — the only file you need to read for
   scoring decisions. It contains all 100 prompts and both runtimes'
   answers.
2. `scoring_rubric.txt` — the rubric (also reproduced below).
3. `results/<slug>.csv` — the full row data. You will overwrite the
   `genie_score`, `geniex_score`, and `note` columns when you're done,
   but everything else (timing, errors) stays intact.

### Shape B — geniex-only re-run paired with a prior full run

When the user changed only geniex code and re-ran with `--skip-genie`,
they will hand you **two** answers files instead of one:

- `results/<slug>_geniex_only.answers.json` — the **new** run. Has the
  fresh `geniex_answer` values; `genie_answer` is empty on every row
  (skipped). The file's top-level shape is `{"meta": {...}, "rows": [...]}`
  with `meta.skip_genie == true` — read `meta.rows` for the answers.
- `results/<slug>.answers.json` — the **prior** full run. Has the
  authoritative `genie_answer` values for the same model. (genie's
  output is deterministic for a given model + prompt + template, so
  re-running it would not change the answer — that's why `--skip-genie`
  exists.)

The user will tell you which file is which when they ask for scoring;
if they hand you a `_geniex_only.answers.json` without a partner, ask
them for the prior full run before scoring.

For Shape B you score each runtime from a different file:

| column to score   | source file                                  | source field     |
| ----------------- | -------------------------------------------- | ---------------- |
| `geniex_score`    | `<slug>_geniex_only.answers.json` (new)      | `geniex_answer`  |
| `genie_score`     | `<slug>.answers.json` (prior full run)       | `genie_answer`   |

The prompts and ids in both files come from the same `testing_prompts.md`,
so they line up by `id`. If a row exists in one file but not the other,
flag it and skip — don't guess.

The scored CSV you emit for Shape B is `results/<slug>_geniex_only_scored.csv`
(7 columns, same shape as Shape A's deliverable). Do **not** touch
`<slug>_scored.csv` from the prior run — leave that file alone as the
scoring record for the prior geniex build.

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

## Workflow (Shape A — single full run)

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
   scoring JSON, and write back the full CSV (same columns it came with).
   Then derive the **7-column scored CSV** the user actually wants:
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

# Rewrite the full CSV with scores filled in (same columns the collection
# script emits — perf metrics preserved, no timing columns).
full_cols = ['id','category','prompt','genie_answer','geniex_answer',
             'genie_score','geniex_score','note',
             'genie_ttft_ms','geniex_ttft_ms','genie_tps','geniex_tps',
             'genie_error','geniex_error']
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

## Workflow (Shape B — geniex-only re-run + prior full run)

The flow is the same as Shape A, except step 1 reads two answers
files and merges them by `id` into one in-memory list of rows; step 3
writes the geniex-only CSV (`<slug>_geniex_only.csv`) and the
geniex-only scored deliverable (`<slug>_geniex_only_scored.csv`),
leaving the prior full-run files untouched.

1. **Load both answers files** and merge by `id`.

   ```python
   PYTHONIOENCODING=utf-8 python -c "
   import json
   new = json.load(open('results/<slug>_geniex_only.answers.json', encoding='utf-8'))
   prior = json.load(open('results/<slug>.answers.json', encoding='utf-8'))
   # New file uses the {meta, rows} shape; prior is the legacy top-level list.
   new_rows = new['rows'] if isinstance(new, dict) else new
   prior_rows = prior['rows'] if isinstance(prior, dict) else prior
   prior_by_id = {r['id']: r for r in prior_rows}
   merged = []
   for r in new_rows:
       p = prior_by_id.get(r['id'])
       if p is None:
           print(f'skip id={r[\"id\"]}: no prior row'); continue
       merged.append({
           'id': r['id'],
           'category': r.get('category', p.get('category', '')),
           'prompt': r['prompt'],
           'genie_answer': p['genie_answer'],
           'geniex_answer': r['geniex_answer'],
       })
   json.dump(merged, open('results/<slug>_geniex_only.merged.json', 'w', encoding='utf-8'),
             ensure_ascii=False, indent=2)
   print(f'merged {len(merged)} rows')
   "
   ```

   Read `<slug>_geniex_only.merged.json` for scoring decisions. (Or
   merge in-memory — the helper file is just here for transparency.)

2. **Score every prompt.** Same rubric, same `_scoring.json` shape as
   Shape A. Write to `results/<slug>_geniex_only_scoring.json`.

3. **Merge into the geniex-only CSV.** The collection script wrote
   `results/<slug>_geniex_only.csv` with empty `genie_*` columns. Pull
   the genie answer + genie perf metrics from the prior `<slug>.csv`
   (same id) so the geniex-only CSV is self-contained for scoring,
   then fill in `genie_score`, `geniex_score`, and `note`. Finally
   emit the 7-column deliverable
   `results/<slug>_geniex_only_scored.csv`.

   ```python
   PYTHONIOENCODING=utf-8 python -c "
   import csv, json, sys
   sys.stdout.reconfigure(encoding='utf-8')

   slug = '<slug>'
   new_csv  = f'results/{slug}_geniex_only.csv'
   prior_csv = f'results/{slug}.csv'
   score_path = f'results/{slug}_geniex_only_scoring.json'
   out_path = f'results/{slug}_geniex_only_scored.csv'

   rows = list(csv.DictReader(open(new_csv, encoding='utf-8')))
   prior_by_id = {r['id']: r for r in csv.DictReader(open(prior_csv, encoding='utf-8'))}
   scoring = {s['id']: s for s in json.load(open(score_path, encoding='utf-8'))}

   g, x = 0, 0
   for r in rows:
       p = prior_by_id.get(r['id'], {})
       # Backfill genie_* columns from the prior full run so the file is
       # self-contained for downstream review.
       for col in ('genie_answer','genie_ttft_ms','genie_tps','genie_error'):
           if not r.get(col): r[col] = p.get(col, '')
       s = scoring[int(r['id'])]
       r['genie_score'] = s['genie_score']
       r['geniex_score'] = s['geniex_score']
       r['note'] = s['note']
       g += s['genie_score']; x += s['geniex_score']

   full_cols = ['id','category','prompt','genie_answer','geniex_answer',
                'genie_score','geniex_score','note',
                'genie_ttft_ms','geniex_ttft_ms','genie_tps','geniex_tps',
                'genie_error','geniex_error']
   with open(new_csv, 'w', encoding='utf-8', newline='') as f:
       w = csv.DictWriter(f, fieldnames=full_cols); w.writeheader()
       for r in rows: w.writerow({k: r.get(k, '') for k in full_cols})

   short_cols = ['id','prompt','genie_answer','geniex_answer',
                 'genie_score','geniex_score','note']
   with open(out_path, 'w', encoding='utf-8', newline='') as f:
       w = csv.DictWriter(f, fieldnames=short_cols); w.writeheader()
       for r in rows: w.writerow({k: r.get(k, '') for k in short_cols})

   print(f'genie  {g}/1000 avg {g/100:.2f} (from prior run)')
   print(f'geniex {x}/1000 avg {x/100:.2f} (this run)')
   print(f'wrote {out_path}')
   "
   ```

4. **Report totals.** Same as Shape A, plus call out the comparison:
   how `geniex_score` shifted versus the prior `<slug>_scored.csv`
   (per-prompt deltas of interest, total delta). The genie totals
   should match the prior run's genie totals exactly (same answers,
   same rubric); if they don't, you scored something differently from
   last time — note the disagreements.

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
- **Don't overwrite a prior scored CSV.** Shape B writes to
  `<slug>_geniex_only_scored.csv` so the prior `<slug>_scored.csv`
  stays as the historical record for the previous geniex build.
