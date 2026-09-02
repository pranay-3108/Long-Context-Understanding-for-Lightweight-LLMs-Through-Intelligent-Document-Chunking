# Enhancing Long-Context Information Retention in Lightweight LLMs

A research benchmark for studying whether lightweight local LLMs can retain and understand long research papers when the document is too large for their practical context window.

The central idea is:

> **Can better document processing and context-preserving chunking help a small local LLM understand long research papers well enough that we do not need a heavy model?**

## Research Goal

This project focuses on improving long-context information retention in lightweight LLMs through document chunking and context-preserving processing.

Instead of evaluating a model only by whether its response sounds fluent, the project compares model answers with a fixed reference answer key and examines:

- what the model got correct
- what information it missed
- what it added
- what it got wrong
- whether additions are valid or unsupported
- how answer quality changes as more of the paper becomes available

The main processing comparison is:

```text
Direct full-context processing
                vs
Chunk -> summarize -> aggregate
                vs
Improved chunking strategies
```

## Benchmark Tracks

### Track 1 — Long-document / ResNet benchmark

The earlier benchmark uses character-length versions of:

**Deep Residual Learning for Image Recognition**

Paper stages:

```text
8k
16k
24k
32k
44k
```

These values are character counts, not token counts.

The original baseline uses approximately **7,000 characters per chunk**:

```text
Paper
 ↓
Fixed-size chunks
 ↓
Summarize each chunk
 ↓
Aggregate summaries
 ↓
Final analysis
```

Earlier experiments showed that chunking can lose important research information such as equations, numerical evidence, citations, and relationships between sections.

A documented Granite 44k run produced:

```text
Direct: 211.73 s
Fixed 7k chunking: 228.28 s
```

This is historical timing evidence only. It does not establish a quality winner.

### Track 2 — BERT deep-understanding benchmark

The newer benchmark uses:

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

It contains **55 research questions** designed to test more than direct retrieval.

The benchmark covers:

- factual understanding
- conceptual understanding
- motivation
- methodology and architecture
- training details
- numerical and table reasoning
- experiment and ablation analysis
- cross-section / multi-hop reasoning
- evidence-bound reasoning
- critical/adversarial questions
- deep synthesis
- insufficient-information / hallucination checks

The same questions are used across comparable runs.

## Reference and Model Comparison

The BERT benchmark has three answer sources.

### Primary reference — Claude

The Claude-generated answer key is the **primary ground-truth reference**.

### Secondary comparator — Granite 7B

The supplied **Granite4:tiny-h 7B** answer set is preserved as a model comparator.

It is **not** treated as ground truth because the 7B output itself contains incorrect or incomplete answers.

### Lightweight evaluated models

The project includes:

- DeepSeek R1 1.5B
- Granite 3.3 2B
- Qwen 2.5 3B

Additional models can be tested using the same benchmark.

## BERT Stage Experiment

The BERT paper is evaluated at:

```text
8k
16k
24k
32k
44k
```

The current baseline chunking method is:

```text
Fixed 7,000-character chunks
```

For each stage:

```text
BERT paper
 ↓
Selected paper-length stage
 ↓
Processing method
 ↓
LLM
 ↓
55 questions
 ↓
Model answers
 ↓
Compare with Claude reference
```

This lets us study how answer quality changes as more of the same paper becomes available.

## Current Granite 2B + Fixed 7k Result

The current BERT fixed-chunk Granite 3.3 2B results are:

| Paper length | Avg Answer F1 |
|---:|---:|
| 8k | 0.2378 |
| 16k | 0.2649 |
| 24k | 0.2761 |
| 32k | 0.2696 |
| 44k | 0.3073 |

The current highest stage is **44k**.

Answer F1 is a **lexical-overlap baseline**, not proof of semantic understanding.

## What Is Compared for Each Question

For a selected run, the dashboard compares:

```text
Claude ground truth
        ↓
Granite 7B comparator
        ↓
Current lightweight model
```

The question-level analysis records:

- reference answer
- model answer
- correct / partially correct / incorrect / unsupported
- what the model got right
- what the model missed
- what the model added
- whether added information is valid/supportable
- differences from the Granite 7B comparator
- research notes

The goal is to explain **why** a model succeeds or fails, rather than relying on one score.

## Example of the Intended Error Analysis

For each answer:

```text
Reference answer
       vs
Model answer
```

we want to identify:

```text
Correct content
Missing content
Added content
Incorrect content
Unsupported claims
```

For example:

> The model may correctly identify the main concept but miss a numerical result.

or:

> The model may add a plausible statement that is not supported by the paper.

or:

> The model may agree with Granite 7B but both models may still disagree with the Claude reference.

This is why the 7B output is a comparator rather than the truth.

## Model-Size Question

One major research question is:

> **Does a larger parameter count always produce better long research-paper understanding?**

We want to compare:

```text
1.5B
2B
3B
7B
```

under the same question set and, where possible, the same processing conditions.

A deeper question is:

> **Can better document processing allow a smaller model to match or outperform a larger model?**

This is more useful than assuming that parameter count alone determines long-document performance.

## Direct vs Chunking

For the same model and paper stage, the controlled comparison is:

```text
Direct
    vs
Fixed 7k chunking
```

A direct-vs-chunking winner is declared only when both actual runs exist.

Missing results must never be treated as zero.

The same Claude reference is used for both.

Example:

| Model | Stage | Direct | Fixed 7k chunking | Winner |
|---|---:|---:|---:|---|
| Granite 2B | 8k | — | 0.2378 | Awaiting direct run |
| Granite 2B | 16k | — | 0.2649 | Awaiting direct run |
| Granite 2B | 24k | — | 0.2761 | Awaiting direct run |
| Granite 2B | 32k | — | 0.2696 | Awaiting direct run |
| Granite 2B | 44k | — | 0.3073 | Awaiting direct run |

## Chunking Methods

The intended long-term comparison has three methods.

### Method 1 — Fixed / baseline chunking

Simple fixed-size chunks.

Example:

```text
7,000 characters
```

### Method 2 — Structure-aware chunking

Improve chunk boundaries to avoid splitting meaningful units such as:

- equations
- paragraphs
- section headings
- figure captions
- tables
- algorithm blocks
- experimental evidence
- important numerical results

The objective is better information preservation without adding unnecessary model calls.

### Method 3 — Verified / adaptive chunking

```text
Chunk
 ↓
Initial summary
 ↓
Verification
 ↓
Accept OR improve once
 ↓
Aggregate
```

Verification checks whether important content was preserved, such as:

```text
Problem
Method
Evidence
Experiments
Important numbers
Equations
Conclusion
```

The method should have a hard retry limit and no endless self-reflection loop.

## Why Chunking Matters

Long research papers contain information whose meaning depends on nearby or distant context.

Potential failure modes include:

```text
equation separated from explanation
number separated from experiment
figure caption separated from figure
claim separated from evidence
section relationship lost
summary becomes generic
model fills missing context with unsupported information
```

The benchmark is specifically designed to detect these failures.

## Question Difficulty

The 55-question benchmark contains multiple difficulty levels:

**Easy**
- direct facts
- simple retrieval

**Medium**
- interpretation
- comparison
- concept understanding

**Hard**
- multi-step reasoning
- experiment interpretation
- multi-section connections

**Very Hard**
- cross-section synthesis
- numerical reasoning
- evidence-bound conclusions
- counterfactual reasoning
- consistency checks

Questions are intentionally designed so that fluent generation alone is not enough.

## Dashboard

`app.py` provides a research-oriented Streamlit dashboard with:

- stage-level results
- model comparison
- method comparison
- Claude vs Granite 7B vs lightweight model comparison
- question-level visualization
- Answer F1 charts
- model win counts
- missed/added information
- research review fields
- direct-vs-chunk tables
- stage-wise trend views

For a selected question, the dashboard can show a visual comparison of model Answer F1 and identify which model has the stronger lexical agreement with the reference.

## Data Organization

Results should remain separated by:

```text
Model
  ↓
Method
  ↓
Paper stage
```

Example:

```text
results/
└── bert/
    └── models/
        └── granite3_3_2b/
            └── fixed_chunk_7k/
                ├── 8k/
                ├── 16k/
                ├── 24k/
                ├── 32k/
                └── 44k/
```

This prevents results from different models or processing methods from being mixed.

## Current Status

### Completed / available

- BERT research paper added
- BERT 8k / 16k / 24k / 32k / 44k stages created
- 55-question BERT deep-understanding benchmark
- Claude ground-truth answer key
- supplied Granite 7B answer set
- Granite 3.3 2B fixed 7k chunk results for all five stages
- question-level Answer F1 comparisons
- three-way dashboard comparison
- missed/added information analysis interface
- research review storage
- visualization for per-question model comparison

### Still to complete for the full study

- clean Direct BERT runs under the same 55-question benchmark
- structure-aware chunking runs
- verified/adaptive chunking runs
- equivalent benchmark runs for Qwen 3B and DeepSeek 1.5B
- final semantic review across methods
- final model-size comparison
- final research conclusions

## Main Research Questions

1. Does long-context answer quality improve as more of the research paper becomes available?
2. Does chunking preserve research-paper understanding better than direct processing?
3. Which chunking method preserves the most important information?
4. Can structure-aware processing reduce context and evidence loss?
5. Can verified/adaptive processing improve reliability without excessive retries?
6. Can a smaller LLM with better document processing outperform or approach a larger LLM?
7. Which failures are caused by missing context and which are caused by model misunderstanding?
8. Does improved context preservation reduce unsupported or hallucinated claims?

## Practical Objective

The broader objective is to make capable AI systems possible on resource-constrained hardware.

Instead of requiring a heavy LLM with a very large context window, the project investigates whether:

> **Smarter document processing can give lightweight local models better access to the information they need, allowing useful long-document understanding on ordinary laptops and edge systems.**

## IBM-Relevant Direction

The project is designed around a practical AI engineering problem:

> How can capable language-model functionality be brought to lower-resource environments without automatically requiring a much larger model?

This makes the work relevant to efficient AI, edge AI, local inference, model optimization, and practical deployment constraints.

## Important Interpretation Rules

- Claude is the primary reference for correctness.
- Granite 7B is a comparator, not ground truth.
- Answer F1 is a baseline, not proof of understanding.
- Model fluency is not treated as correctness.
- Missing information and unsupported additions should be described explicitly.
- Direct-vs-chunking conclusions require actual results for both methods.
- Old historical results must not be mixed with the new BERT benchmark.
- Character-length stages are not token-length claims.
- Parameter count alone is not treated as a sufficient explanation for performance.

## Running the Dashboard

From the project root:

```powershell
python -m streamlit run app.py
```

The dashboard is intended primarily for analyzing already-generated benchmark results.
