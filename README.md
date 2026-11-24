# Small Language Models on Deep Paper Understanding

**A controlled evaluation of DeepSeek-R1 1.5B and Granite 3.3 2B on long, technical research-paper question answering.**

> **Six-second overview** — Built a 55-question benchmark from the BERT research paper and evaluated two locally run small language models at five context lengths (8K–44K characters), producing **550 scored model answers** plus a **275-row static Granite 7B comparator**. On the archived 18 November 2026 evaluation, **Granite 3.3 2B achieved 27.1% pooled token-level Answer F1 versus 24.7% for DeepSeek-R1 1.5B**. Granite's F1 increased from **23.8% at 8K to 30.7% at 44K**; DeepSeek showed no statistically detectable context-length effect. A Streamlit dashboard supports model/context comparison, question-level evidence inspection, diagnostic analysis, human-review capture and result export.

**The key distinction:** these are *reference-alignment results*, not proof that a model understands the paper. The archived evaluation export contains zero submitted human reviews, and token-level F1 can reward plausible wording even when a factual detail is wrong.

---

## Project at a glance

| | Details |
|---|---|
| Research area | Small language models, long-document QA, scientific-paper understanding, LLM evaluation |
| Source document | *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* |
| Benchmark | 55 fixed questions across 9 question categories |
| Context stages | 8K, 16K, 24K, 32K and 44K characters |
| Main models | DeepSeek-R1 1.5B and IBM Granite 3.3 2B |
| Static comparator | Granite 7B answer set; not ground truth and not a context-scaling run |
| Processing method in the analysed export | Fixed 7,000-character chunking |
| Scored outputs | 550 from the two actively evaluated models; 275 static Granite 7B comparator rows; 825 rows total |
| Primary metric | Token-level Answer F1 against a common Claude-generated reference answer key |
| Application | Streamlit research dashboard with question-level inspection and human-review capture |
| Evidence snapshot | Retention Lab export generated 18 November 2026 |

## Demo and research report

- **[Watch the platform demo](https://drive.google.com/file/d/12T9uPdO7QcpBxOo8M_RqbmAy37hZcroG/view?usp=sharing)**
- **[Read the technical evaluation report](https://lnkd.in/dc4wGCit)**
- **[GitHub profile](https://github.com/pranay-3108)**
- **[LinkedIn](https://linkedin.com/in/pranayugale)**

---

## 1. Research question

Small language models can run locally and require fewer resources than much larger models, but answering questions from long technical documents requires retrieving evidence, retaining details across a context, and producing complete answers without adding unsupported information.

This project investigates three questions:

1. **Reference alignment:** How closely do lightweight local models' answers match a common reference answer key on detailed questions about a research paper?
2. **Context length:** How do answer scores change across five document-context settings, from 8K to 44K characters?
3. **Processing methods:** Can alternative document-processing methods improve long-document QA?

The archived results provide evidence for the first question and partial evidence for the second. They do **not** establish the third: only the fixed 7K chunking baseline has completed runs in the analysed export. A symbolic processing method is present in the project but has no completed results in this archive.

## 2. Benchmark design

### Source document and questions

The benchmark uses the BERT research paper, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. It contains **55 fixed questions** designed to test paper-specific reading rather than general familiarity with machine learning.

Questions cover factual retrieval, evidence-based reasoning, methods and architecture, numerical/table interpretation, motivation, experiments, hallucination checks, training details and multi-hop synthesis. Some questions refer to specific tables and claims in the paper, including Tables 1, 3, 4, 5 and 8.

### Question categories

| Category | Questions | Rows across three model sets |
|---|---:|---:|
| Retrieval / factual | 10 | 150 |
| Evidence / reasoning | 8 | 120 |
| Method / architecture | 7 | 105 |
| Numerical / tables | 7 | 105 |
| Motivation / concept | 5 | 75 |
| Experiment / ablation | 5 | 75 |
| Hallucination check | 5 | 75 |
| Training / implementation | 4 | 60 |
| Multi-hop / synthesis | 4 | 60 |
| **Total** | **55** | **825** |

### Evaluation matrix

Each actively evaluated model answers the same 55 questions at all five context settings:

\[
55\ \text{questions} \times 5\ \text{context settings} = 275\ \text{answers per model}
\]

\[
2\ \text{active models} \times 275 = 550\ \text{scored answers}
\]

The supplied Granite 7B comparator contributes another 275 rows. Those rows represent a **static answer set repeated across the five context labels**, not 275 newly generated answers and not a Granite 7B context-scaling experiment.

### Models

| Model | Role in the experiment | Execution / data status |
|---|---|---|
| DeepSeek-R1 1.5B | Actively evaluated small model | Local model tag: `deepseek-r1:1.5b` |
| Granite 3.3 2B | Actively evaluated small model | Local model; five context settings |
| Granite 7B | Static comparison reference | Supplied answer set; unchanged across context labels |
| Claude | Reference-answer authoring | Common reference answer key; not treated as a model under test |

**Important:** the Claude answer key is a reference, not independently verified ground truth. Granite 7B is a model comparator, not ground truth. Agreement with either reference cannot by itself establish factual correctness.

### Document processing

The analysed experiment uses a single method: **fixed 7,000-character chunking**. The symbolic method (terminology map followed by 7,000-character chunks) is installed but has no completed runs in the archived results. Therefore, the current report cannot support a claim that symbolic chunking improves performance.

The export also does not fully specify the mapping between each context-window setting and the exact portion of the source paper visible to the model. The context-length results should therefore be interpreted as an association with the configured context setting, not definitive causal proof that the model saw or retained more of the paper.

---

## 3. Evaluation metrics

### 3.1 Token-level Answer F1

The primary metric compares a generated answer with the common reference answer using token overlap. The archived scorer is `answer_f1.py`; the report describes its normalization as removing articles and punctuation before scoring.

Let:

- \(A\) be the generated answer's normalized token sequence;
- \(R\) be the reference answer's normalized token sequence;
- \(TP\) be the number of overlapping tokens;
- \(FP\) be generated-answer tokens that do not overlap the reference;
- \(FN\) be reference tokens not found in the generated answer.

Then:

\[
P = \frac{TP}{TP+FP}
\]

\[
R_c = \frac{TP}{TP+FN}
\]

\[
F1 = \frac{2 \times P \times R_c}{P+R_c}
\]

Here \(P\) is lexical precision and \(R_c\) is lexical recall. The subscript distinguishes recall from the reference token sequence. The implementation's exact token-overlap and edge-case handling should be treated as authoritative when reproducing a score.

**Interpretation:** Answer F1 measures lexical overlap with one reference answer. It is not a semantic similarity metric, factuality score, completeness judgment or direct measurement of understanding. Correct paraphrases can score poorly, while an answer containing a factual error can still obtain a substantial score.

### 3.2 Additional metrics and diagnostics

The question-level export includes additional fields to help inspect answers rather than relying on one aggregate score.

| Metric / diagnostic | What it is used to inspect | Interpretation caution |
|---|---|---|
| Precision | How much of the generated answer overlaps the reference | Can reward short answers that omit important information |
| Recall | How much of the reference content appears in the answer | Can reward verbosity without guaranteeing correctness |
| Answer and reference token counts | Response length and coverage context | Length is not answer quality |
| Bigram overlap | Local phrase overlap between answer and reference | Low overlap does not necessarily mean an answer is wrong |
| Repetition ratio | Repeated content within a generated answer | A diagnostic of generation behaviour, not factual correctness |
| Missing numbers | Reference numbers not detected in the answer | Numeric parsing is a diagnostic; it does not judge the whole answer |
| Extra numbers | Answer numbers not detected in the reference | An extra number may be unsupported, but should be checked in context |
| F1 against Granite 7B | Agreement with the static comparator | Comparator agreement is not truth or correctness |
| Negation / polarity flag | Potential changes in positive/negative meaning | Heuristic flag; not a verified error label |
| Repetition / abnormal-length flags | Potentially degraded or runaway generation | Requires question-level inspection |
| Free-text flags | Additional diagnostic signals | Flags are heuristics, not human correctness labels |

The missing-number and extra-number diagnostics can be understood conceptually as:

\[
\text{Missing numbers} = N_R \setminus N_A
\]

\[
\text{Extra numbers} = N_A \setminus N_R
\]

where \(N_R\) and \(N_A\) are the numbers detected in the reference and answer. The exact extraction and matching rules are implementation-specific; these diagnostics should not be interpreted as a complete mathematical or factual audit.

### 3.3 Why multiple signals are necessary

A response can have a reasonable F1 while containing a decisive factual error. Conversely, a correct abstention can be penalized because its wording differs from a longer reference. For this reason, the dashboard exposes the question, reference, model answer, metrics and diagnostic evidence together. Aggregate metrics should be read alongside individual examples and, where available, human judgments.

---

## 4. Statistical analysis

The evaluation report uses question-level statistical procedures because the five context observations for the same question are related and should not be treated as five independent questions.

### Pooled mean

For an actively evaluated model, the pooled mean averages its 275 answer-level F1 values:

\[
\overline{F1}_{m} = \frac{1}{55 \times 5}
\sum_{q=1}^{55}\sum_{c=1}^{5} F1_{m,q,c}
\]

where \(m\) is the model, \(q\) is the question and \(c\) is the context setting.

### Cluster bootstrap confidence interval

The 95% confidence intervals use **3,000 bootstrap resamples clustered by question**. Each resample draws from the 55 questions and retains the five context observations belonging to each sampled question. This preserves the within-question grouping better than resampling all 275 rows independently.

### Paired model comparison

The main Granite 3.3 2B versus DeepSeek-R1 1.5B comparison is paired by question and context. The primary inferential test collapses the five context scores to a mean for each question, giving **55 paired question-level observations**, and applies the Wilcoxon signed-rank test.

### Context-length comparison

The Friedman test compares the five context settings across the same 55 questions. The report also uses a Wilcoxon signed-rank test for the 44K-versus-8K comparison and Spearman correlation as a trend check.

### Category-level analysis

Category comparisons are exploratory. They pool context observations, have small question counts (4–10 per category), and are not independent in the same way as a new set of questions. The report notes that nine category comparisons were not originally corrected; a Bonferroni-style reference threshold is:

\[
\alpha_{\text{Bonferroni}} = \frac{0.05}{9} \approx 0.0056
\]

Category p-values should therefore be interpreted cautiously, especially those close to the threshold.

---

## 5. Results

**Evidence snapshot:** Retention Lab export generated **18 November 2026**. Scores below describe that archive and should not be silently mixed with later runs or changed benchmark configurations.

### 5.1 Overall model performance

| Model | Pooled Answer F1 | 95% CI | Precision | Recall | Mean answer length |
|---|---:|---:|---:|---:|---:|
| **Granite 3.3 2B** | **27.1%** | **[25.2%, 29.2%]** | 29.5% | 28.9% | 82 tokens |
| DeepSeek-R1 1.5B | 24.7% | [23.0%, 26.4%] | 24.6% | 28.4% | Typically 79–98 tokens; 44K mean affected by an outlier |
| Granite 7B (static) | 24.1% | Not estimated | 42.1% | 18.3% | 30 tokens |

Granite 3.3 2B leads DeepSeek-R1 1.5B by **2.5 percentage points** in the report's paired analysis (95% CI for the paired difference: **[+1.1, +3.8]**; question-level Wilcoxon **p = 0.0014**). The paired answer comparisons favour Granite 3.3 2B in 58.5% of cases, DeepSeek in 40.7%, with 0.7% ties.

The absolute lexical F1 values are modest. The result supports a limited statement: Granite 3.3 2B produced greater lexical agreement with the Claude-written reference key in this benchmark snapshot. It does **not** establish that Granite is more factually correct or generally understands technical papers better.

The static Granite 7B score is shown for transparency, not as a valid model-size comparison. Its answer set is unchanged across the five context labels, and its shorter answer style produces higher precision but lower recall. It should not be interpreted as a controlled 7B-versus-2B experiment.

### 5.2 Performance across context lengths

**Mean token-level Answer F1 by context length**

| Context | DeepSeek-R1 1.5B | Granite 3.3 2B | Granite 7B (static) |
|---|---:|---:|---:|
| 8K | 24.8% | 23.8% | 24.1% |
| 16K | 24.3% | 26.5% | 24.1% |
| 24K | 24.5% | 27.6% | 24.1% |
| 32K | 25.7% | 27.0% | 24.1% |
| 44K | 24.1% | 30.7% | 24.1% |

**Granite 3.3 2B:** F1 rises by **7.0 percentage points** between 8K and 44K (23.8% to 30.7%). The Friedman test reports **p < 0.0001**; the 44K-versus-8K Wilcoxon test reports **p < 0.0001**; Spearman's \(\rho = 0.21\), **p = 0.0005**. The trend is not perfectly monotonic: the score falls from 27.6% at 24K to 27.0% at 32K.

**DeepSeek-R1 1.5B:** Scores range from 24.1% to 25.7%, with no statistically detectable context effect in this export (Friedman **p = 0.46**). The 44K-versus-8K difference is −0.7 percentage points (Wilcoxon **p = 0.56**).

**Granite 7B:** The score remains 24.1% at every context label because this is a static answer set, not a newly generated context-length experiment.

The context trend is a result associated with the configured context settings. Because the archived configuration does not fully document the source-paper token count and the mapping from each window setting to the exact text visible to the model, it cannot establish that greater document exposure alone caused the observed change.

### 5.3 Results by question category

Mean F1 is pooled across the five context settings. The difference is Granite 3.3 2B minus DeepSeek-R1 1.5B.

| Question category | DeepSeek 1.5B | Granite 3.3 2B | Granite 7B | Difference (pp) | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| Retrieval / factual | 30.4% | 34.4% | 40.0% | +4.0 | 0.0960 |
| Evidence / reasoning | 25.6% | 29.1% | 11.8% | +3.6 | 0.0008 |
| Motivation / concept | 25.0% | 28.2% | 27.0% | +3.2 | 0.0046 |
| Multi-hop / synthesis | 25.8% | 26.9% | 13.5% | +1.1 | 0.1800 |
| Experiment / ablation | 25.5% | 26.9% | 21.3% | +1.4 | 0.4600 |
| Method / architecture | 22.6% | 25.5% | 29.4% | +2.8 | 0.1800 |
| Hallucination check | 19.2% | 25.2% | 9.7% | +6.0 | 0.0056 |
| Numerical / tables | 23.4% | 20.3% | 23.1% | −3.1 | 0.0135 |
| Training / implementation | 18.2% | 21.1% | 30.6% | +3.0 | 0.6000 |

At the Bonferroni-style threshold of approximately 0.0056, the Evidence / reasoning and Motivation / concept comparisons fall below the threshold. Hallucination check is reported at the rounded threshold and should be treated as borderline. These category results are exploratory, not definitive evidence of category-wide superiority.

A notable reversal occurs for Numerical / tables: DeepSeek scores 3.1 points higher than Granite 3.3 2B (p = 0.0135), although this does not meet the nine-comparison Bonferroni threshold.

### 5.4 Precision, recall, length and repetition

The three answer sets have different lexical profiles:

- **Granite 7B** is brief (about 30 tokens on average), with higher precision (42.1%) and lower recall (18.3%).
- **Granite 3.3 2B** averages 82 tokens, with 29.5% precision and 28.9% recall.
- **DeepSeek-R1 1.5B** has typical answer lengths around 79–98 tokens, with 24.6% precision and 28.4% recall. Its 44K mean length is distorted by one runaway generation; the median is more representative there.

For Granite 3.3 2B, answer length has a modest positive association with F1 (Spearman \(\rho = 0.18\), **p = 0.003**). For DeepSeek, the association is not statistically significant (\(\rho = 0.06\), **p = 0.34**). This is another reason not to interpret F1 without checking answer length and content.

Repetition diagnostics distinguish the two active models:

| Diagnostic | DeepSeek-R1 1.5B | Granite 3.3 2B | Granite 7B |
|---|---:|---:|---:|
| Mean repetition ratio | 0.39–0.43 | 0.25–0.32 | 0.17 |
| Answers with repetition flag | 237 / 275 (86.2%) | 130 / 275 (47.3%) | Not stated in the report |

Bigram overlap with the reference is low across the answer sets (approximately 4–6%). This indicates limited phrase-level overlap, but does not by itself establish whether an answer is correct or incorrect.

### 5.5 Important observed failure modes

**Runaway generation — DeepSeek at 44K**

One DeepSeek answer (Q20, SQuAD v2.0 adaptation) generated **24,196 tokens** / **207,370 characters** and took **554.7 seconds**. Its repetition ratio was **0.997** and F1 was **0.002**. Excluding this answer changes DeepSeek's 44K mean F1 only from approximately 24.1% to 24.5%, so the pooled F1 conclusion is not driven by this single case. However, the 44K mean answer length and mean latency are distorted: report the median length (**98 tokens**) and median time (**10.7 seconds**) alongside the affected means.

**Numerical and table questions**

Every one of the **35 answers per active model** in the Numerical / tables category is flagged as missing at least one reference number. Across all categories, **72.7%** of answers from each active model miss at least one reference number. Extra numbers are detected in **38.2%** of DeepSeek answers and **35.6%** of Granite 3.3 2B answers.

For example, on Q2 (BERT_BASE specification), DeepSeek returned 6 attention heads and 342 million parameters, while the reference states 12 attention heads and 110 million parameters. The answer still received F1 = 0.27 because other terms, including the layer count and hidden size, overlapped.

**Lexical overlap can hide a factual error**

On Q1, DeepSeek expanded BERT as “Bidirectional Language Model” rather than “Bidirectional Encoder Representations from Transformers.” Despite this acronym error, it received F1 = 0.67. The example shows why a high lexical score is not equivalent to factual correctness.

**A correct abstention can be penalized**

On a carbon-emissions hallucination-check question, both active models correctly stated that the paper did not provide the requested figure. Their scores were nevertheless only 0.22 (DeepSeek) and 0.30 (Granite 3.3 2B), because the longer Claude reference included additional TPU-related details. This score reflects reference wording and coverage, not simply whether the model hallucinated.

**Negation / polarity heuristic**

The negation or polarity flag fires on 110 DeepSeek answers and 123 Granite answers (approximately 40–45%). The report considers this heuristic likely oversensitive. These counts must not be reported as verified negation-error rates without manual inspection.

---

## 6. Research dashboard

The Streamlit dashboard is designed to make the evaluation inspectable beyond a single leaderboard.

### Main capabilities

- Select models and context lengths for comparison.
- Compare model answers against the common reference answer key.
- Compare model outputs directly, while keeping comparator agreement separate from correctness.
- Inspect individual questions, reference answers and generated responses.
- Review Answer F1, precision, recall and diagnostic fields.
- Examine missing / extra numbers, repetition, abnormal-length and other heuristic flags.
- View performance trends across context lengths and question categories.
- Record human research judgments.
- Export research data for further analysis.

### Question-level inspection workflow

\[
\text{Question}
\rightarrow \text{Reference answer}
\rightarrow \text{Model response}
\rightarrow \text{Metrics}
\rightarrow \text{Diagnostic evidence}
\rightarrow \text{Human review}
\]

### Human-review status and evidence boundary

The **platform supports recording human review judgments**. However, the specific Retention Lab export analyzed in the technical report contains an empty human-review list (**0 submitted reviews**). Therefore, the results reported in Sections 5.1–5.5 are not human-validated correctness results. Any reviews submitted in a later platform session or later export should be reported as a separate, dated evidence snapshot rather than retroactively attributed to the 18 November archive.

---

## 7. Reproducibility and repository layout

The project is organized around the dashboard, benchmark logic, experiment scripts, model configuration, stored results, datasets and source papers.

```text
research-benchmark-bert-research-dashboard-v6/
│
├── app.py
├── benchmark/
├── scripts/
├── models/
├── results/
├── datasets/
├── papers/
└── requirements.txt
```

The archived analysis package described in the report includes:

- Research brief (HTML) with summary tables and evaluation safeguards.
- Question-level metrics in CSV and JSON (825 rows, 23 columns in each representation).
- Human-review JSON (empty list in the 18 November 2026 export).
- Experiment configuration JSON with model, context, scorer and comparator notes.

### Reproducing the dashboard

The repository's `requirements.txt` defines the Python dependencies. From the repository root, a typical setup is:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

To reproduce the archived numerical results, use the matching question-level CSV/JSON export and the same scorer/configuration recorded with that export. Re-running local models may produce different outputs unless model versions, prompts, inference parameters, hardware, context construction and generation settings are held constant. The 18 November report does not document every one of those details, so the aggregate figures should be treated as results from that archived export, not a guarantee that a fresh run will reproduce every cell exactly.

### Reproducibility checklist

For a fully repeatable future release, publish or record:

- Exact model identifiers, model-file versions or hashes, and inference runtime versions.
- Prompt templates and generation parameters, including temperature, top-p, maximum output tokens and stop conditions.
- The exact paper text, character/token counts, preprocessing and chunk-to-context mapping.
- Chunk overlap and ordering rules, if any.
- Scorer version, tokenization rules and normalization details.
- Hardware, runtime and timing-measurement procedure.
- Run identifiers, timestamps, seeds and repeated-run policy.
- Human-review rubric, reviewer counts and adjudication procedure.

---

## 8. What the evidence supports — and what it does not

### Supported by the 18 November 2026 export

- Granite 3.3 2B has higher pooled lexical Answer F1 than DeepSeek-R1 1.5B on this 55-question BERT benchmark: 27.1% versus 24.7%.
- The paired question-level difference favours Granite 3.3 2B by approximately 2.5 percentage points, with a 95% CI of [+1.1, +3.8] and Wilcoxon p = 0.0014.
- Granite 3.3 2B's reference-alignment score rises from 23.8% at 8K to 30.7% at 44K in this configuration.
- DeepSeek-R1 1.5B has no detectable context-length effect in this export.
- Both active models have substantial difficulty with numerical and table-based questions.
- DeepSeek has a higher repetition rate and one extreme runaway-generation case.
- Token-level F1 alone can misrepresent answer quality, as shown by the acronym and correct-abstention examples.

### Not established by this export

- That either model truly understands the BERT paper or is more factually correct; the archived export contains no completed human reviews.
- That increasing context length *causes* better understanding; the exact visible-text mapping is insufficiently documented.
- That symbolic chunking or another processing method improves performance; the archive contains no completed symbolic-method runs.
- A causal model-size effect from Granite 7B; its answer set is static and not a controlled context-matched run.
- A speed advantage for either active model; timing is available only for DeepSeek in this export.
- Generalization to other research papers, domains, languages or question sets; the benchmark contains one paper and 55 questions.
- Stable run-to-run performance; there is one answer per model, question and context, without repeated seeds.

---

## 9. Limitations and threats to validity

1. **No human reviews in the archived export.** The platform has human-review functionality, but the 18 November 2026 export contains zero submitted reviews. Findings concern lexical agreement and diagnostics, not human-verified correctness.
2. **Model-generated reference.** The Claude answer key is generated text, not an independently adjudicated gold standard. Its wording and length (about 82 tokens on average) influence lexical scores.
3. **Single processing method in completed runs.** Only fixed 7K chunking has results in this archive; no method comparison can be claimed.
4. **Unclear context-to-document mapping.** The exact amount of source text exposed at each context setting is not fully specified.
5. **Static Granite 7B comparator.** Its answers are repeated unchanged across context labels and have a different length/style profile. This cannot establish a model-size effect.
6. **Single-document benchmark.** Fifty-five questions about one research paper provide limited evidence for generalization.
7. **No repeated runs.** There is one answer per model, question and context; run-to-run variability is unmeasured.
8. **Small category samples.** Each category contains only 4–10 questions. Category-level results are exploratory.
9. **Multiple category comparisons.** Category p-values require correction and cautious interpretation.
10. **Lexical metric limitations.** Token F1 can penalize paraphrases and correct abstentions while rewarding answers that repeat many reference terms despite a factual error.
11. **Timing asymmetry.** Generation times are available only for DeepSeek, so the archive does not support a speed comparison between the two active models.
12. **Runaway-output sensitivity.** One DeepSeek generation substantially distorts mean response length and mean latency at 44K; medians are more representative for those measurements.

---

## 10. Next research steps

1. Complete a stratified human review, prioritizing all numerical and hallucination-check questions plus a random sample. Score correctness, completeness, additions and unsupported claims using a documented rubric.
2. Run the symbolic 7K chunking method with the same models, questions and context settings to enable a controlled processing-method comparison.
3. Document the source-paper token count and exactly how each context setting maps to visible text and chunks.
4. Run at least three independent repetitions per model/question/context cell, with recorded seeds and inference settings, to estimate run-to-run variability.
5. Add numeric-match scoring and an abstention-aware metric for unanswerable questions.
6. Validate any LLM-judge or rubric-based metric against human-reviewed examples.
7. Run Granite 7B under the same question and context protocol if a model-size comparison is intended.
8. Add output-token caps, stop-reason logging and runaway-generation safeguards; report medians alongside means for skewed time and length distributions.
9. Expand the benchmark to multiple research papers and technical domains.

---

## 11. Selected numbers

| Evidence item | Result |
|---|---:|
| Fixed questions | 55 |
| Question categories | 9 |
| Context settings | 5 (8K–44K characters) |
| Active-model answers | 550 |
| Static Granite 7B comparator rows | 275 |
| Total archived question-level rows | 825 |
| Granite 3.3 2B pooled Answer F1 | 27.1% |
| DeepSeek-R1 1.5B pooled Answer F1 | 24.7% |
| Paired difference (Granite minus DeepSeek) | +2.5 percentage points |
| Paired difference 95% CI | [+1.1, +3.8] percentage points |
| Paired Wilcoxon p-value | 0.0014 |
| Granite 3.3 2B F1 at 8K → 44K | 23.8% → 30.7% |
| DeepSeek context-effect Friedman p-value | 0.46 |
| Human reviews in archived 18 Nov export | 0 |
| DeepSeek runaway output | 24,196 tokens; F1 = 0.002 |

---

## Author

**Pranay Ugale**  
B.Tech, Computer Science and Engineering (Data Science)  
GitHub: [pranay-3108](https://github.com/pranay-3108)  
LinkedIn: [pranayugale](https://linkedin.com/in/pranayugale)

---

## Research interpretation

This project is an ongoing experimental study of long-document question answering in lightweight language models. Its central contribution is not a claim that one small model has solved long-document understanding. It is a transparent evaluation workflow that combines controlled questions, context-length comparisons, reference-based metrics, question-level diagnostics and a dashboard for inspecting where aggregate scores succeed or fail.

**The results should be interpreted as benchmark-specific experimental evidence, not universal claims about model capability.**
