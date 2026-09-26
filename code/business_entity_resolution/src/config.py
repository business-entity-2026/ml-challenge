from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    train_dir: Path
    test_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class BlockingConfig:
    name_top_k: int = 25
    address_top_k: int = 25
    max_candidates_per_source1: int = 100


@dataclass(frozen=True)
class ModelConfig:
    threshold: float = 0.80
    random_state: int = 42
