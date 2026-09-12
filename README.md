Evidence-Based Long-Document Understanding Research

An experimental research project focused on improving how lightweight language models understand and retain information from long research documents.

Overview

This project evaluates lightweight local LLMs on challenging research-paper question answering tasks across increasing document lengths.

The benchmark is designed to measure not only whether a model produces an answer, but also whether the answer remains accurate, complete, consistent, and evidence-aligned as more document content becomes available.

The work combines:

controlled long-document evaluation

fixed question sets

common reference answers

model-to-model comparison

question-level verification

detailed error analysis

a research dashboard for transparent inspection

What I am studying

The main research focus is long-document understanding and context retention in smaller language models.

The experiment investigates how model performance changes when the same research document is evaluated at multiple context lengths, and whether improvements in document understanding can make a lightweight model competitive with a substantially larger model.

Benchmark

The current research benchmark uses the BERT research paper:

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

The benchmark contains:

55 fixed research questions

multiple question types covering factual understanding, reasoning, methodology, numerical evidence, experiments, synthesis, and hallucination checks

5 document-length stages: 8K, 16K, 24K, 32K, and 44K characters

The same questions are used across comparable evaluations to keep the comparison consistent.

Models Evaluated

The project evaluates lightweight local models alongside a larger model baseline.

Current models include:

DeepSeek 1.5B

Granite 3.3 2B

Granite 7B baseline

Additional models can be evaluated using the same benchmark.

Reference-Based Evaluation

A common Claude Extra reference answer key is used as the primary evaluation reference.

The evaluation compares each model answer against the same reference and also supports direct comparison between model outputs.

The Granite 7B answer set is treated as a model comparator, not as ground truth.

This allows the project to distinguish:

reference agreement from model-to-model agreement.

What is measured

The benchmark goes beyond a single aggregate score.

Primary answer-quality measurements include:

Answer F1

Precision

Recall

Additional analysis includes:

answer length

generation time

question-level performance

missing information

added information

numerical inconsistencies

unsupported claims

negation or polarity differences

repetitive generation behaviour

response similarity and agreement

The purpose is to understand not only which model scores higher, but also where and why the answers differ.

Question-Level Verification

The dashboard makes the experiment inspectable at the individual-question level.

For a selected question, a reviewer can compare:

Question

→ Claude reference answer

→ Model response

→ Answer metrics

→ Missing information

→ Added information

→ Diagnostic evidence

This makes important results easier to verify without relying only on an overall leaderboard.

Long-Context Evaluation

The same research paper is evaluated at:

8K → 16K → 24K → 32K → 44K characters

This enables analysis of how answer quality changes as the amount of available document information increases.

The experiment is particularly focused on whether lightweight models can preserve useful information when working with documents that become increasingly long.

Research Finding

A key finding from the current experiment is that parameter count alone does not fully explain long-document answer quality.

After improving the long-document understanding and information-retention behaviour of DeepSeek 1.5B, the model was able to outperform the Granite 7B baseline in the tested benchmark and show strong performance across multiple context levels and question-level comparisons.

The project therefore studies the broader relationship between:

model size

context length

information retention

answer reliability

and long-document understanding.

Transparent Error Analysis

The benchmark is designed to surface failure cases that can be hidden by a single score.

Examples include:

missing important numerical information

incomplete answers

unsupported additions

hallucinated information

changes in negation or meaning

repeated or degraded responses

differences between model answers

This is useful for evaluating practical reliability rather than measuring fluency alone.

Research Dashboard

A Streamlit dashboard provides an interactive view of the experiment.

It supports:

selecting models for comparison

selecting context lengths

comparing models against the common reference

comparing models directly

inspecting individual questions

reviewing answer differences

viewing performance trends

examining diagnostic signals

recording human research judgments

exporting research data

The dashboard is designed so that both technical and non-technical readers can inspect the evidence behind a result.

Current Results

The current repository contains BERT benchmark results for the Granite 3.3 2B model using the current evaluation setup.

Context

Average Answer F1

8K

23.78%

16K

26.49%

24K

27.61%

32K

26.96%

44K

30.73%

These values are based on the project's lexical Answer F1 evaluation against the common reference answer key.

Additional model results are stored separately and can be loaded by the dashboard when available.

Research Value

The project is aimed at a practical AI engineering problem:

Can smaller language models achieve stronger long-document performance through better information handling, without depending only on parameter scaling?

This has potential relevance to:

efficient AI

local inference

resource-constrained environments

edge AI

long-context applications

document intelligence

research-paper analysis

trustworthy LLM evaluation

Reproducibility

The project uses a controlled benchmark with fixed research questions, fixed document stages, common reference answers, and local model inference.

The dashboard reads stored benchmark results so that experiments can be inspected after execution.

The repository contains the benchmark data, evaluation components, result storage, and dashboard needed to reproduce the documented workflow.

Project Structure

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

Detailed implementation and experimental-processing logic are intentionally not documented here.

Research Status

This is an ongoing research project.

Current work includes:

long-document evaluation

lightweight-model comparison

question-level answer verification

context-length analysis

reliability-oriented error analysis

Future expansion will include additional research papers, broader model coverage, repeated experiments, stronger semantic evaluation, and deeper statistical analysis.

Important Interpretation

The results in this repository should be interpreted as experimental benchmark findings, not as universal claims about model capability.

The project distinguishes:

lexical evaluation from semantic correctness

model agreement from factual correctness

benchmark evidence from broad generalization

model size from document-processing capability

Author

Pranay Ugale

B.Tech Computer Science and Engineering (Data Science)

GitHub: https://github.com/pranay-3108

LinkedIn: https://linkedin.com/in/pranayugale
report -https://lnkd.in/dc4wGCit

