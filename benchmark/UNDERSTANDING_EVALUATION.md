# BERT understanding evaluation

This benchmark compares a 7B Granite answer set against a Claude-provided ground-truth answer key for the same 55 BERT questions.

## Important interpretation

Answer F1 is only a lexical-overlap baseline. It must NOT be treated as a proof of understanding. A model can receive a high F1 score while copying phrases, and a good paraphrase can receive a lower score.

The app therefore shows:
- question-level side-by-side answers
- answer F1 as a baseline metric
- category-level F1
- manual review fields for correctness, conceptual understanding, evidence grounding, and hallucination
- a separate deep-understanding view

## Experiment principle

The main research question is whether the model can maintain an evidence-grounded, coherent understanding of a paper rather than merely retrieving or generating plausible text.

Use the same questions for every model and every reading mode. Do not give the evaluated model the ground-truth answers.

For stronger claims, compare direct vs chunked answers and use the same rubric for both.
