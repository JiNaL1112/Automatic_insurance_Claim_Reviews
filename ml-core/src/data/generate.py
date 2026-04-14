import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


logging.basicConfig(
    stream=sys.stdout,
    format='{"level":"%(levelname)s","msg":"%(message)s","time":"%(asctime)s"}',
    level=logging.INFO,
)
log = logging.getLogger(__name__)


FEATURES = [
    "claim_amount",
    "num_services",
    "patient_age",
    "provider_id",
    "days_since_last_claim",
]


def load_params(path: str | os.PathLike = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_normal_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_amount": rng.uniform(100, 5_000, n),
            "num_services": rng.integers(1, 10, n),
            "patient_age": rng.integers(18, 80, n),
            "provider_id": rng.integers(1, 50, n),
            "days_since_last_claim": rng.integers(1, 365, n),
        }
    )


def _make_anomaly_df(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Intentionally extreme / unlikely combinations to give the model a signal.
    return pd.DataFrame(
        {
            "claim_amount": rng.uniform(500_000, 999_999, n),
            "num_services": rng.integers(90, 100, n),
            "patient_age": rng.integers(0, 5, n),
            "provider_id": rng.integers(1, 5, n),
            "days_since_last_claim": rng.integers(0, 1, n),
        }
    )


def generate_dataset(
    *,
    num_normal_samples: int,
    num_anomalies: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    normal = _make_normal_df(rng, int(num_normal_samples))
    anomalies = _make_anomaly_df(rng, int(num_anomalies))

    df = pd.concat([normal, anomalies], ignore_index=True)

    # Optional but useful for debugging/training:
    df.insert(0, "claim_id", np.arange(1, len(df) + 1, dtype=np.int64))
    df["label"] = 0
    if num_anomalies:
        df.loc[len(normal) :, "label"] = 1

    # Ensure deterministic row order but avoid leaking "all anomalies at end" as an easy cue.
    df = df.sample(frac=1.0, random_state=int(random_seed)).reset_index(drop=True)
    return df


def main() -> None:
    params = load_params()
    dp = params["data"]

    out_path = Path(dp["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        num_normal_samples=dp["num_normal_samples"],
        num_anomalies=dp["num_anomalies"],
        random_seed=dp["random_seed"],
    )

    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(df), out_path.as_posix())


if __name__ == "__main__":
    main()
