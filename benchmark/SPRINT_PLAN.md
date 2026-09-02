# Sprint Plan — Multi-Method Chunking Implementation
### For: Codex (or any coding agent) working on `research-benchmark`
### Scope discipline: touch ONLY the files listed per sprint. Do not refactor, rename, or "improve" anything not explicitly listed.

---

## Ground Rules (read first, apply to every sprint)

1. **Do not modify** the existing Method 1 baseline behavior. `chunk_paper.py --method boundary` (the default) must produce byte-identical output to what it produces today. Method 1 is the control group — if it changes, every comparison becomes invalid.
2. **Do not touch** `models/qwen/*`, `models/deepseek/*`, `benchmark/answer_f1.py`, `benchmark/evaluation_result.py`, `benchmark/reference_answers/*.json`, or anything under `papers/qasper/`. Out of scope for this sprint set.
3. **Every new file's output must be tagged with its chunking method** — filenames, JSON fields, and folder paths must all make it unambiguous which method produced which answer. This is the top requirement — see "Output Separation Contract" below before writing any code.
4. **No new top-level dependencies.** Everything here is stdlib `re` + existing `ollama` calls already in the codebase.
5. After each sprint: run the verification command listed, paste the output, and stop. Do not start the next sprint in the same pass.

---

## Output Separation Contract (applies to Sprints 2–4)

Every artifact produced by chunking/summarization/evaluation must carry `method` in three places:

| Artifact | Naming rule |
|---|---|
| Chunk files | `outputs/chunked/{paper_name}__{method}__chunk_{n}.txt` |
| Summary files | `outputs/chunked/{paper_name}__{method}__summary_{n}.txt` |
| Evaluation JSON | `benchmark/evaluations/granite/chunk/{method}/{paper_id}.json` (new `{method}` subfolder) |
| Evaluation JSON field | every result object gets `"chunking_method": "boundary" \| "structure_aware" \| "adaptive_verified"` |
| Reference-eval report | `benchmark/reference_reports/{paper}__{model}__{mode}__{method}__reference.json` |

This means: **never overwrite one method's files with another's.** Today's code uses `{paper_name}_chunk_{n}.txt` with no method tag — that must change everywhere chunk/summary files are written, or Method 1 and Method 3 outputs will silently collide in the same folder.

---

## Sprint 0 — Rename existing methods to match agreed terminology (small, mechanical)

**Why:** current code calls the equation/table-protection method `"structure_aware"`. Going forward, `"structure_aware"` means *section/paragraph-boundary splitting* (the other AI's Method 2), and equation/table protection + verification becomes part of `"adaptive_verified"` (Method 3). This sprint just renames, no new logic.

**Files to touch:**
- `benchmark/text_chunker.py`

**Tasks:**
1. Rename the current `method="structure_aware"` (equation/table protection) to `method="equation_table_safe"` internally — keep the function, just rename the string literal and update the docstring at the top of the file to describe 3 target methods: `boundary`, `structure_aware`, `adaptive_verified`.
2. Do not remove `equation_table_safe` — Sprint 2 will fold it into `adaptive_verified`.
3. Update `models/granite/chunk_paper.py`'s `--method` choices to `["boundary", "equation_table_safe"]` for now (temporary; Sprint 2 replaces this).

**Verification:**
```
python3 -c "from benchmark.text_chunker import split_into_chunks; print(split_into_chunks('a. b. c.', 3, method='equation_table_safe'))"
```
Must run without error and produce the same chunks as the old `structure_aware` did.

---

## Sprint 1 — Method 2: Structure-Aware (section/paragraph) Splitting

**New method name:** `method="structure_aware"`

**Files to create:**
- None — add to `benchmark/text_chunker.py`

**Files to touch:**
- `benchmark/text_chunker.py`

**Tasks:**
1. Add a header-detection function:
   ```python
   def _find_section_headers(text: str) -> list[int]:
       """
       Returns character offsets of lines that look like real section
       headers (e.g. '3.2 Residual Learning', '4. Experiments'), NOT
       numbered captions or list items.

       A line qualifies as a header if ALL of:
         - starts with 1-2 dot-separated numbers OR is short Title Case text
         - line length <= 80 chars
         - does not end in a period, comma, or colon
         - the following line does not start with the same number pattern
           (rules out numbered lists)
       Reject lines containing "Figure", "Table", "Algorithm" immediately
       followed by a number (those are captions, not headers).
       """
   ```
   This must handle the real false-positive case already found in `papers/resnet_32k.txt` line 638 (`"34. Right: a "bottleneck" building block..."`) — that must NOT be detected as a header.
2. Add `_find_paragraph_breaks(text) -> list[int]`: offsets of blank lines (`\n\n` or `\r\n\r\n`).
3. Wire both into `split_into_chunks(..., method="structure_aware")`: when searching for a cut point, section headers and paragraph breaks are the *first* preference (stronger than `_STRONG_BOUNDARIES`), sentence punctuation is the fallback if no header/paragraph break exists within the search window.
4. Update `--method` choices in `models/granite/chunk_paper.py` to `["boundary", "structure_aware", "equation_table_safe"]`.

**Verification (must pass before moving on):**
```python
from benchmark.text_chunker import _find_section_headers
text = open("papers/resnet_32k.txt").read()
headers = _find_section_headers(text)
# Must include the offsets of "1. Introduction", "2. Related Work",
# "3. Deep Residual Learning", "4. Experiments"
# Must NOT include the offset of "34. Right: a "bottleneck"..." (line 638)
```
Print the matched header text at each detected offset and confirm by eye against `grep -n "^[0-9]" papers/resnet_32k.txt`.

---

## Sprint 2 — Method 3: Adaptive Verified Chunking

**New method name:** `method="adaptive_verified"`

**Files to create:**
- `models/granite/summarize_chunk_verified.py` already exists from a prior sprint — **extend it, do not recreate it.**

**Files to touch:**
- `benchmark/text_chunker.py` — fold `equation_table_safe`'s protected-span logic into `adaptive_verified`'s chunking step (i.e. `adaptive_verified` chunking = `structure_aware` splitting + equation/table protected spans, combined)
- `models/granite/summarize_chunk_verified.py` — upgrade the verification step

**Tasks:**
1. In `text_chunker.py`, add `method="adaptive_verified"` to `split_into_chunks()`: uses the same header/paragraph-aware cut preference as `structure_aware`, AND applies the protected-span logic from `equation_table_safe` (equations, table rows) on top. One combined method, not two separate calls.
2. In `summarize_chunk_verified.py`, upgrade `build_summary_prompt` usage (do not edit the shared `build_summary_prompt` in `summarize_chunk.py` — copy/extend it locally in this file) so the model is asked, in the SAME generation call, to output:
   ```
   SUMMARY:
   <summary text>

   COVERAGE:
   Problem: Present/Missing
   Method: Present/Missing
   Evidence: Present/Missing
   Numbers: Present/Missing
   Conclusion: Present/Missing
   ```
   Parse this deterministically (split on the `COVERAGE:` marker). This replaces a *second* model call with structured output in the *same* call — do not add an extra `ollama.chat` call for the rubric.
3. Combine this rubric coverage with the existing `verify_chunk_summary()` regex fact-check (both checks run, no extra cost — regex is free, rubric came from the same call). If EITHER signals a problem (rubric has any "Missing" OR regex coverage < 0.8), trigger exactly one retry — reuse the existing `MAX_RETRIES = 1` cap, do not change it.
4. Every chunk's verification result (rubric fields + regex coverage + retry count) gets written to `outputs/chunked/{paper_name}__adaptive_verified__verification_{n}.json` alongside the summary file.

**Verification:**
```
python3 models/granite/summarize_chunk_verified.py --paper resnet_8k
```
Then confirm:
- Exactly one `_verification_{n}.json` file per chunk exists.
- No chunk has `retries_used > 1`.
- Total ollama calls logged does not exceed `2 * num_chunks`.

---

## Sprint 3 — Wire Methods 2 & 3 into the Real Evaluation Pipeline

**Files to touch:**
- `benchmark/granite_validation.py` — the `_run_chunk_question()` function (currently calls `chunk_text_to_files(artifact_name, paper_text)` with no method argument)
- `benchmark/evaluation.py` — `_split_into_chunks()` caller path, if it's used by anything still active

**Tasks:**
1. Add a `method` parameter to `_run_chunk_question(paper_name, question_payload, paper_text, method="boundary")`.
2. Thread `method` through to `chunk_text_to_files(artifact_name, paper_text, method=method)` and to `summarize_chunk_files(...)` — for `method="adaptive_verified"`, call `summarize_chunk_verified()` instead of the plain `summarize_chunk_files()`; for `"boundary"` and `"structure_aware"`, keep using the existing plain summarizer.
3. Output path for the saved evaluation JSON must become `benchmark/evaluations/granite/chunk/{method}/{paper_name}.json` — create the `{method}` subfolder, do not write directly into `chunk/`.
4. Add `"chunking_method": method` as a field on every `EvaluationResult` produced in chunk mode. Direct mode results get `"chunking_method": null` (direct mode has no chunking).
5. Do not change how direct mode works at all.

**Verification:**
```
python3 -m benchmark.granite_validation --paper paper_0001 --mode chunk --method boundary
python3 -m benchmark.granite_validation --paper paper_0001 --mode chunk --method structure_aware
python3 -m benchmark.granite_validation --paper paper_0001 --mode chunk --method adaptive_verified
```
Confirm three separate, non-overwriting JSON files exist:
```
benchmark/evaluations/granite/chunk/boundary/paper_0001.json
benchmark/evaluations/granite/chunk/structure_aware/paper_0001.json
benchmark/evaluations/granite/chunk/adaptive_verified/paper_0001.json
```
Each must contain `"chunking_method"` matching its folder name.

---

## Sprint 4 — Comparison Report Across All 3 Methods

**Files to create:**
- `benchmark/compare_chunking_methods.py`

**Files to touch:**
- None else.

**Tasks:**
1. New script, CLI: `python -m benchmark.compare_chunking_methods --paper paper_0001 --model granite`
2. Runs (or reads, if already run) all three chunk-mode evaluations for that paper (`boundary`, `structure_aware`, `adaptive_verified`) plus the existing direct-mode result.
3. Produces one CSV: `results/{paper}_method_comparison.csv` with columns:
   ```
   method, mode, questions_scored, average_answer_f1, total_time,
   avg_retries_per_chunk, avg_coverage_score, num_chunks
   ```
   (`avg_retries_per_chunk` and `avg_coverage_score` are `null`/blank for `boundary`, `structure_aware`, and `direct` — those fields only apply to `adaptive_verified`.)
4. Also print a plain-text side-by-side table to stdout so it's readable without opening the CSV.

**Verification:**
```
python -m benchmark.compare_chunking_methods --paper paper_0001 --model granite
```
Confirm the CSV has exactly 4 rows (`direct`, `boundary`, `structure_aware`, `adaptive_verified`) and every F1 value is a real float, not `None`, for any row that ran successfully.

---

## Definition of Done (all sprints)

- [ ] Method 1 (`boundary`) output is unchanged from before this sprint plan started (regression check).
- [ ] Method 2 (`structure_aware`) correctly identifies real section headers and skips numbered captions (tested against the known false-positive at `resnet_32k.txt` line 638).
- [ ] Method 3 (`adaptive_verified`) never exceeds 1 retry per chunk, never makes more than 2 model calls per chunk.
- [ ] Every chunk/summary/evaluation file on disk is unambiguously traceable to its method via filename AND a `chunking_method` JSON field — no folder or filename collisions between methods.
- [ ] `compare_chunking_methods.py` produces one CSV per paper with all 4 rows (direct, boundary, structure_aware, adaptive_verified) so F1/time/coverage can be read side by side.
- [ ] No file outside this plan's explicit list was modified.
