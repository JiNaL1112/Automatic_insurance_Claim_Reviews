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

import pandas as pd
import numpy as np
import yaml
import os

def load_params(path="params.yaml"):
    # Try ml-core/params.yaml if run from root
    for candidate in [path, "ml-core/params.yaml"]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("params.yaml not found")

def generate_data():
    params = load_params()
    dp = params["data"]

    rng = np.random.default_rng(dp["random_seed"])
    n_normal = dp["num_normal_samples"]
    n_anomalies = dp["num_anomalies"]
    total = n_normal + n_anomalies

    # Normal claims
    normal = pd.DataFrame({
        "claim_id":              [f"CLM{i:05d}" for i in range(1, n_normal + 1)],
        "claim_amount":          rng.normal(1000, 300, n_normal).clip(50, 5000),
        "num_services":          rng.integers(1, 9, n_normal),
        "patient_age":           rng.integers(18, 90, n_normal),
        "provider_id":           rng.integers(1, 50, n_normal),
        "days_since_last_claim": rng.integers(0, 365, n_normal),
    })

    # Anomalous claims — high amount + high service count
    anomalies = pd.DataFrame({
        "claim_id":              [f"CLM{i:05d}" for i in range(n_normal + 1, total + 1)],
        "claim_amount":          rng.normal(10000, 2000, n_anomalies).clip(6000, 50000),
        "num_services":          rng.integers(10, 20, n_anomalies),
        "patient_age":           rng.integers(18, 90, n_anomalies),
        "provider_id":           rng.integers(1, 50, n_anomalies),
        "days_since_last_claim": rng.integers(0, 365, n_anomalies),
    })

    df = pd.concat([normal, anomalies], ignore_index=True).sample(
        frac=1, random_state=dp["random_seed"]
    ).reset_index(drop=True)

    output_path = dp["output_path"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records ({n_normal} normal, {n_anomalies} anomalies) → {output_path}")

if __name__ == "__main__":
    generate_data()

