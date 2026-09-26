# System Architecture

```text
                 TRAIN DATA
                     │
                     ▼
             ┌──────────────┐
             │ Load + Clean │
             └──────┬───────┘
                    ▼
             ┌──────────────┐
             │   Blocking   │
             └──────┬───────┘
                    ▼
          candidate_pairs.tsv
                    │
                    ▼
             ┌──────────────┐
             │   Features   │
             └──────┬───────┘
                    ▼
             ┌──────────────┐
             │   ML Model   │
             └──────┬───────┘
                    ▼
             ┌──────────────┐
             │ F0.5/Thresh. │
             └──────┬───────┘
                    ▼
          matching_results.tsv
```

### Key contract

`matching_results.tsv` must only contain candidate IDs already present in `candidate_pairs.tsv` for the same Source 1 entity.
