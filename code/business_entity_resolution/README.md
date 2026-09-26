# Business Entity Resolution Pipeline

This folder contains the self-contained runnable pipeline required for the challenge submission package.

## Pipeline

1. Load tab-separated source files.
2. Normalize business names, addresses, and country labels.
3. Generate a small candidate set with blocking.
4. Create pairwise similarity features.
5. Train a binary match classifier on training ground truth.
6. Tune the decision threshold on a validation split before final test inference.
7. Write `candidate_pairs.tsv` and `matching_results.tsv`.

## Team ownership

- Member 1: data loading and preprocessing
- Member 2: blocking / candidate generation
- Member 3: features and ML model
- Member 4: evaluation, integration, submission

## Important challenge constraints

- Input/output is TSV, not CSV.
- Every Source 1 test entity must appear exactly once in `matching_results.tsv`.
- Final matches must be a subset of `candidate_pairs.tsv`.
- Do not force a match for every Source 1 entity.
- No external entity lookup or data enrichment.
- Final model must satisfy the challenge's licensing and parameter constraints.

## Run

From the repository root:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python -m src.main
```

The baseline currently generates valid candidate output and a deliberately conservative empty-match baseline. The team should complete validation, model scoring, threshold tuning, and final prediction before submission.
