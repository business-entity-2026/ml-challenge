# Business Entity Resolution Challenge

Professional 4-member team repository for the Business Entity Resolution Challenge.

## Goal

For every Source 1 business, identify all matching records from Source 2 and Source 3 while keeping candidate sets small and final precision high.

## Repository architecture

```text
business-entity-resolution/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── blocking.py
│   │   ├── config.py
│   │   ├── evaluation.py
│   │   ├── features.py
│   │   ├── io_utils.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── pipeline.py
│   │   └── preprocessing.py
│   ├── README.md
│   └── requirements.txt
├── business_entity_resolution_tests/
├── data/
├── docs/
├── notebooks/
├── output/
├── .github/workflows/
├── Documentation_template.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Team split

| Member | Ownership | Main files |
|---|---|---|
| 1 | Data + preprocessing | `io_utils.py`, `preprocessing.py` |
| 2 | Blocking / candidate generation | `blocking.py` |
| 3 | Features + ML | `features.py`, `model.py` |
| 4 | Evaluation + integration | `evaluation.py`, `pipeline.py`, `main.py` |

## Git workflow

Use one shared GitHub repository with four feature branches:

```text
main
├── member1-preprocessing
├── member2-blocking
├── member3-model
└── member4-evaluation
```

Do not commit challenge dataset files. They are ignored by `.gitignore`.

## Important

This repository is a professional starter scaffold. The dataset-dependent thresholds and final matching strategy must be learned from the provided training data and validated before submission.
