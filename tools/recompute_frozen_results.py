"""Audit deterministic, frozen-expert combinations from exported predictions.

This computes predictions from supplied weights; it does not fit a gate, train
a regressor or infer how an archived checkpoint was selected. Use separate MC
draw replay for stochastic predictions, rather than averaging component means.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY = ["record_id", "observed_cycle"]
EXPERTS = [f"expert{i}" for i in range(3)]
WEIGHTS = [f"weight{i}" for i in range(3)]
KEYS = ["model", "gate", "seed", *IDENTITY]


def recompute(frame):
    """Return recomputed rows and seed-level/group-level MAE/RMSE in cycles."""
    required = set(KEYS + ["target"] + EXPERTS + WEIGHTS)
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Nonempty rows with complete identity and values required.")
    if any(frame[c].astype(str).str.strip().eq("").any() for c in ["model", "gate", "record_id"]):
        raise ValueError("Model, gate and record IDs cannot be blank.")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate model/gate/seed/observation identity.")
    if "prediction_type" in frame and not frame.prediction_type.eq("deterministic").all():
        raise ValueError("Only deterministic component predictions are supported.")
    numeric = frame[["seed", "observed_cycle", "target", *EXPERTS, *WEIGHTS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite numeric values.")
    if (frame.target < 0).any() or (frame.observed_cycle < 0).any():
        raise ValueError("Targets and observed cycles must be nonnegative.")
    if not np.equal(numeric[:, 0], np.floor(numeric[:, 0])).all():
        raise ValueError("Seeds must be integers.")
    weights = frame[WEIGHTS].to_numpy(dtype=float)
    if (weights < 0).any() or not np.allclose(weights.sum(1), 1, rtol=0, atol=1e-8):
        raise ValueError("Weights must be nonnegative and sum to one; no silent normalization.")
    for _, group in frame.groupby("model"):
        seed_sets = [set(part.seed) for _, part in group.groupby("gate")]
        if any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError("Gate variants must have matching seed sets.")

    # Every gate must act on the same observations and frozen component outputs.
    # Compare by explicit keys, never by incoming row order.
    for (_, _), group in frame.groupby(["model", "seed"]):
        baseline = None
        for _, values in group.groupby("gate", sort=True):
            indexed = values.set_index(IDENTITY).sort_index()
            if baseline is not None:
                if not indexed.index.equals(baseline.index):
                    raise ValueError("Gate variants have different observation identities.")
                if not np.allclose(indexed[["target", *EXPERTS]], baseline[["target", *EXPERTS]], rtol=0, atol=1e-7):
                    raise ValueError("Gate variants must preserve targets and frozen expert outputs.")
            baseline = indexed

    rows = frame.copy()
    predicted = (weights * frame[EXPERTS].to_numpy(dtype=float)).sum(1)
    if "prediction" in rows:
        stored = rows.prediction.to_numpy(dtype=float)
        # Archived float32 component exports may differ by a few thousandths of
        # a cycle at ~100000 cycles. Preserve and report that numerical delta.
        if not np.isfinite(stored).all() or not np.allclose(stored, predicted, rtol=1e-7, atol=1e-6):
            raise ValueError("Stored predictions do not match weighted components; check protocol/MC/clipping.")
        rows["stored_prediction"] = stored
        rows["stored_prediction_difference"] = predicted - stored
    rows["prediction"] = predicted
    rows["error"] = rows.prediction - rows.target
    metrics = []
    for (model, gate, seed), part in rows.groupby(["model", "gate", "seed"], sort=True):
        error = part.error.to_numpy(dtype=float)
        metrics.append(dict(model=model, gate=gate, seed=int(seed), n=len(part),
                            mae=float(abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2)))))
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby(["model", "gate"]).agg(
        seeds=("seed", "nunique"), n_per_seed_min=("n", "min"), n_per_seed_max=("n", "max"),
        mae=("mae", "mean"), mae_sd=("mae", "std"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
    ).reset_index()
    return rows, metrics, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Explicit deterministic component/weight CSV.")
    parser.add_argument("--output", required=True, help="New output directory.")
    args = parser.parse_args()
    source = Path(args.predictions)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows, metrics, summary = recompute(pd.read_csv(source, dtype={"record_id": str, "model": str, "gate": str}))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Input changed during audit.")
    protocol = dict(input_name=source.name, input_sha256=digest, rows=len(rows),
                    metric_units="cycles", prediction_type="deterministic_weighted_sum",
                    seed_summary="arithmetic mean of seed MAE/RMSE; sample SD",
                    trained_models=False, fitted_gate=False, modified_sensor_csv=False,
                    stored_prediction_tolerance=dict(rtol=1e-7, atol_cycles=1e-6),
                    maximum_stored_prediction_difference_cycles=(float(abs(rows.stored_prediction_difference).max())
                        if "stored_prediction_difference" in rows else None),
                    selection_history="must be supplied with the archived run; not inferred by this tool")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
