# Remote 4-Member Team Workflow

## Ownership

**Member 1: Data + preprocessing**
- EDA
- missing-value analysis
- conservative normalization
- reusable loading/normalization utilities

**Member 2: Blocking**
- name and address candidate generation
- multiple blocking strategies
- candidate recall
- average candidates per Source 1
- final candidate set written to `candidate_pairs.tsv`

**Member 3: Features + ML**
- pairwise similarity features
- positive/negative training construction
- classifier experiments
- probability calibration if needed

**Member 4: Evaluation + integration**
- entity-level F0.5
- validation split
- threshold tuning
- end-to-end runner
- output validation and packaging

## Collaboration rule

Every module must have a stable input/output contract. Do not edit another member's module directly unless agreed in a pull request.

## Branches

Create one branch per member and merge through pull requests into `main`.
