from pathlib import Path

from .load_data import load_sources
from .pipeline import run_train_baseline, write_placeholder_test_outputs
from .preprocessing import normalize_source
from .blocking import Blocker


ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "dataset" / "train" if (ROOT / "dataset" / "train").exists() else ROOT / "data" / "train"
TEST_DIR = ROOT / "dataset" / "test" if (ROOT / "dataset" / "test").exists() else ROOT / "data" / "test"
OUTPUT_DIR = ROOT / "output"


def main() -> None:
    # Baseline smoke run. Replace the placeholder prediction stage once validation
    # and threshold tuning are implemented.
    model, _, _ = run_train_baseline(TRAIN_DIR)
    del model

    test_s1, test_s2, test_s3 = load_sources(TEST_DIR, "test")
    test_s1, test_s2, test_s3 = map(normalize_source, (test_s1, test_s2, test_s3))
    candidates = Blocker().generate(test_s1, test_s2, test_s3)
    write_placeholder_test_outputs(test_s1, candidates, OUTPUT_DIR)
    print(f"Generated baseline outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
