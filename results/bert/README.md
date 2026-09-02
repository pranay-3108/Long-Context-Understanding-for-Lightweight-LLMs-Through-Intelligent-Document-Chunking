# BERT staged evaluation outputs

Every run is isolated by model, processing mode, and paper length.

Structure:

`models/<model_label>/<direct|fixed_chunk_7k>/<8k|16k|24k|32k|44k>/answers.json`

Use the SAME 55-question set and the SAME Claude ground-truth key for every condition.

`direct`: the selected paper prefix is supplied directly to the model.

`fixed_chunk_7k`: the selected paper prefix is split into 7000-character chunks; each chunk is summarized, then the summaries are aggregated and used for the question answers.

Do not mix these results with the older QASPER/ResNet outputs.
