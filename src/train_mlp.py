"""Train the reference attacker: which service classes are present in a tunnel?

The attacker is a multi-label MLP with one presence output per class. It is fitted
on identity-shim training tunnels, its model and decision thresholds are chosen on
identity validation tunnels, and it is then frozen for every policy comparison.
Run ``python3 src/train_mlp.py`` before ``python3 src/evaluate.py``.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from metrics import classification, thresholds
from features import feature_contract, schema
from simulation import identity
from workload import build, load, observe, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def check_compatibility(attacker, config):
    """Validate numerical feature semantics and workload, allowing metadata cleanup."""
    saved = attacker["config"]
    old = dict(feature_schema=attacker.get("feature_schema"),
               feature_extraction=attacker.get("feature_contract") or
               feature_contract(saved["feature_extraction"]),
               shim=saved["shim"], workload=saved["workload"], classes=list(attacker["classes"]))
    current = dict(feature_schema=schema(config["feature_extraction"]),
                   feature_extraction=feature_contract(config["feature_extraction"]),
                   shim=config["shim"], workload=config["workload"],
                   classes=config["data"]["categories"])
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                compare(left.get(key), right.get(key), f"{path}.{key}" if path else key)
        elif left != right:
            differences.append(f"{path}: saved={left!r}, current={right!r}")

    compare(old, current, "")
    if differences:
        raise ValueError("Reference attacker is incompatible; retrain with the current config.\n"
                         + "\n".join(differences))


def transform(features, mean, scale, clip):
    logged = np.sign(features) * np.log1p(np.abs(features))
    return np.clip((logged - mean) / scale, *clip).astype(np.float32)


def tunnels(config, folder, split):
    """Flows of one split with its tunnel groups and class-presence targets."""
    flows, labels, captures = load(folder, split)
    workload = config["workload"]
    seed = [workload["seed"], ("train", "val", "test").index(split)]
    groups, presence = build(labels, captures, len(config["data"]["categories"]),
                             workload["rounds"][split], workload,
                             config["shim"]["streams_per_tunnel"], seed)
    return flows, groups, presence


def fit_mlp(features, presence, validation_features, validation_presence, model, classes):
    """Fit candidates and thresholds on validation data; no test observations are used."""
    settings, prep = model["training"], model["preprocessing"]
    logged = np.sign(features) * np.log1p(np.abs(features))
    mean, scale = logged.mean(axis=0), logged.std(axis=0)
    scale[scale < prep["minimum_scale"]] = 1
    x = transform(features, mean, scale, prep["clip"])
    xv = transform(validation_features, mean, scale, prep["clip"])
    y = presence.astype(int)
    best_score, chosen, history = -1, None, []
    with threadpool_limits(limits=settings["threads"]):
        for hidden in model["hidden_layer_candidates"]:
            estimator = MLPClassifier(hidden_layer_sizes=tuple(hidden), **model["estimator"],
                                      shuffle=False, early_stopping=False, tol=0, max_iter=1)
            rng = np.random.default_rng(model["estimator"]["random_state"])
            best, stale = -1, 0
            curve = []
            for epoch in range(1, settings["max_epochs"] + 1):
                order = rng.permutation(len(y))
                estimator.partial_fit(x[order], y[order], classes=np.arange(len(classes)))
                if not np.isfinite(estimator.loss_):
                    raise ValueError("Non-finite training loss")
                probabilities = estimator.predict_proba(xv)
                cuts = thresholds(probabilities, validation_presence)
                metrics = classification(probabilities, validation_presence, cuts, classes)
                score = metrics[settings["selection_metric"]]
                curve.append(dict(epoch=epoch, loss=float(estimator.loss_),
                                  **{key: float(metrics[key]) for key in
                                     ("macro_average_precision",
                                      "macro_f1", "exact_match")}))
                if score > best + settings["min_delta"]:
                    best, stale, best_epoch = float(score), 0, epoch
                    checkpoint = deepcopy(estimator)
                else:
                    stale += 1
                if stale >= settings["patience"]:
                    break
            if best > best_score:
                best_score, chosen, selected_epoch = best, checkpoint, best_epoch
                selected_hidden = hidden
            history.append(dict(hidden=hidden, best_epoch=best_epoch,
                                best_validation_score=best, epochs=curve))
        cuts = thresholds(chosen.predict_proba(xv), validation_presence)
    return dict(classifier=chosen, mean=mean, scale=scale, preprocessing=prep, classes=classes,
                thresholds=cuts, selected_epoch=selected_epoch,
                selected_hidden=selected_hidden, selection_metric=settings["selection_metric"],
                validation_selection_score=best_score, training_history=history,
                feature_count=features.shape[1],
                sklearn_version=sklearn.__version__)


def identity_features(config, source, split, flows, groups, cache):
    """Cache deterministic simulations, invalidating when inputs or feature schema change."""
    path = source / f"{split}.npz"
    descriptor = dict(shim=config["shim"], features=feature_contract(config["feature_extraction"]),
                      workload=config["workload"], source=str(path.resolve()),
                      size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                      schema=schema(config["feature_extraction"]))
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    target = cache / f"identity_{split}_{fingerprint[:16]}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            return stored["features"]
    values = observe(flows, groups, identity, config, 0)[0]
    np.savez_compressed(target, features=values)
    return values


def train(config_path, environments=None):
    config = yaml.safe_load(config_path.read_text())
    data, model = config["data"], config["model"]
    folder = (config_path.parent / data["root"]).resolve()
    output = (config_path.parent / model["output_root"]).resolve()
    for environment in environments or data["environments"]:
        source = folder / environment
        support = validate_splits(source)
        destination = output / folder.name / environment.replace(" ", "_")
        destination.mkdir(parents=True, exist_ok=True)
        observed = {}
        for split in ("train", "val"):
            flows, groups, presence = tunnels(config, folder / environment, split)
            observed[split] = identity_features(config, source, split, flows, groups,
                                                destination), presence
        fitted = fit_mlp(*observed["train"], *observed["val"], model, data["categories"])
        fitted["feature_schema"] = schema(config["feature_extraction"])
        fitted["feature_contract"] = feature_contract(config["feature_extraction"])
        joblib.dump(dict(fitted, config=config), destination / "model.joblib", compress=3)
        with threadpool_limits(limits=model["training"]["threads"]):
            chances = fitted["classifier"].predict_proba(transform(
                observed["val"][0], fitted["mean"], fitted["scale"], model["preprocessing"]["clip"]))
        scores = classification(chances, observed["val"][1], fitted["thresholds"],
                                data["categories"])
        (destination / "training.json").write_text(json.dumps(dict(
            feature_schema=fitted["feature_schema"], feature_count=fitted["feature_count"],
            selected_hidden=fitted["selected_hidden"], selected_epoch=fitted["selected_epoch"],
            selection_metric=fitted["selection_metric"], thresholds=fitted["thresholds"].tolist(),
            validation=scores, source_support=support,
            history=fitted["training_history"]), indent=2, default=float) + "\n")
        print(f"{folder.name}/{environment}: {len(observed['train'][0])} training and "
              f"{len(chances)} validation tunnels; validation macro AP="
              f"{scores['macro_average_precision']:.4f}, macro F1={scores['macro_f1']:.4f}, "
              f"exact match={scores['exact_match']:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--environment", action="append")
    args = parser.parse_args()
    for path in args.config or [ROOT / "configs" / f"{name}.yaml" for name in ("iscx", "vnat")]:
        train(path.resolve(), args.environment)
