import numpy as np
import pandas as pd

from graphene_defect_detector.modeling import evaluate, train_regressor


def _synthetic_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area_px = rng.uniform(1, 100, n)
    num_pixels = area_px * rng.uniform(2, 5, n)
    circularity = rng.uniform(0, 1, n)
    solidity = rng.uniform(0, 10, n)
    compactness = rng.uniform(0, 1, n)
    feret_diameter = rng.uniform(0, 1, n)
    eccentricity = rng.uniform(0, 1, n)

    # A simple, mostly-linear relationship the regressor should pick up on.
    total_energy = -3000 - 5 * area_px + 200 * circularity + rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "area_px": area_px,
            "num_pixels": num_pixels,
            "circularity": circularity,
            "solidity": solidity,
            "compactness": compactness,
            "feret_diameter": feret_diameter,
            "eccentricity": eccentricity,
            "total_energy": total_energy,
        }
    )


def test_train_regressor_fits_a_predictive_model() -> None:
    dataset = _synthetic_dataset()

    trained = train_regressor(
        dataset,
        test_size=0.2,
        random_state=0,
        regressor_params={"n_estimators": 200, "max_depth": 3},
    )
    metrics = evaluate(trained)

    # A model that actually learned the (near-linear) relationship should
    # explain most of the test-set variance; a broken pipeline would score
    # near zero or negative. This is a coarse sanity bound, not a precision
    # check -- see docs/known_quirks.md for the real numeric-parity check
    # against the original thesis dataset.
    assert metrics["r2"] > 0.5
    assert metrics["mse"] >= 0


def test_train_regressor_respects_custom_feature_columns() -> None:
    dataset = _synthetic_dataset()
    trained = train_regressor(
        dataset,
        feature_columns=("area_px", "circularity"),
        regressor_params={"n_estimators": 20},
    )
    assert trained.X_train.shape[1] == 2
