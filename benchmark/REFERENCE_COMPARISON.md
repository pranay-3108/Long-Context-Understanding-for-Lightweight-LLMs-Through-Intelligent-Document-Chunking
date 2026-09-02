# BERT reference comparison

The benchmark keeps two answer sources for every question:

- **Claude ground truth**: primary reference for correctness.
- **Granite4:tiny-h 7B answer set**: secondary, uncorrected model comparator.

The dashboard reports F1 against both. Agreement with the 7B answer is not treated as proof of correctness; the Claude key remains the primary reference.

For each current model/stage/method, the dashboard also shows the raw answer beside both references and lexical clues for missed/added terms. Human diagnosis records whether additions are valid, unsupported, or both.
