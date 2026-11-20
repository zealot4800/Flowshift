"""Compare shaping policies on one task: which service classes share a tunnel?

Identity, Minos, Ditto, Securitas, Blanket and FlowShift receive the same multi-stream tunnels
and face the same frozen attacker, trained on identity tunnels by
``python3 src/train_mlp.py``. The score is exact match: the share of tunnels whose
full set of present classes the attacker gets right, and its drop relative to the
identity shim. For each policy every candidate setting runs on the validation
tunnels. Among those that deliver all payload within the budget, the cheapest whose
exact match is within selection_margin of the lowest then runs on the test tunnels
over several seeds. If none qualifies, the cheapest candidate that delivers
everything is tested and marked as outside the budget.

Run ``python3 src/evaluate.py`` for everything, or select with
``--policy ditto --config configs/vnat.yaml``.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from features import extract_features
from metrics import costs
from obfuscators import blanket, ditto, flowshift, minos, securitas
from simulation import identity
from train_mlp import ROOT, check_compatibility, identity_features, transform, tunnels
from workload import observe, validate_splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = dict(minos=minos, ditto=ditto, securitas=securitas, blanket=blanket,
                flowshift=flowshift)


def select(rows, budget, margin):
    """Index of the candidate to test, after marking which rows are within budget.

    Within budget, the cheapest candidate whose exact match is within `margin` of
    the lowest wins, so small gains do not buy large overheads. Otherwise the
    cheapest candidate that still delivers everything is taken, and failing that
    the one leaving the fewest bytes unsent.
    """
    def rank(index):
        row = rows[index]
        if not row["feasible"]:
            return 2, row["outstanding_bytes"]
        row["within_budget"] = bool(row["overhead_percent"] <= budget["overhead_percent"] and
                                    row["added_p99_delay_ms"] <= budget["added_p99_delay_ms"])
        if row["within_budget"]:
            return 0, row["exact_match"]
        return 1, row["overhead_percent"]

    for row in rows:
        row["within_budget"] = False
    best = min(range(len(rows)), key=rank)
    if rows[best]["within_budget"]:
        best = min((i for i, row in enumerate(rows) if row["within_budget"]
                    and row["exact_match"] <= rows[best]["exact_match"] + margin),
                   key=lambda i: rows[i]["overhead_percent"])
    return best


def average(rows):
    """Mean and standard deviation over seeds of every score that all rows report."""
    keys = [key for key, value in rows[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and all(key in row for row in rows)]
    return dict(feasible=all(row["feasible"] for row in rows),
                mean={key: np.mean([row[key] for row in rows]) for key in keys},
                std={key: np.std([row[key] for row in rows]) for key in keys}, seeds=rows)


def evaluate(config_path, names):
    config = yaml.safe_load(config_path.read_text())
    data, classes = config["data"], config["data"]["categories"]
    settings = config["evaluation"]
    folder = (config_path.parent / data["root"]).resolve()
    for environment in data["environments"]:
        name = environment.replace(" ", "_")
        attacker = joblib.load((config_path.parent / config["model"]["output_root"]).resolve()
                               / folder.name / name / "model.joblib")
        check_compatibility(attacker, config)
        support = validate_splits(folder / environment)
        split = {key: tunnels(config, folder / environment, key)
                 for key in ("train", "val", "test")}
        # Each stream's service, known only to the trusted sender (used by FlowShift).
        archives = {key: np.load(folder / environment / f"{key}.npz") for key in split}
        labels = {key: archive["labels"] for key, archive in archives.items()}

        def chances(features):
            """The frozen attacker's probability that each class is present."""
            with threadpool_limits(limits=config["model"]["training"]["threads"]):
                return attacker["classifier"].predict_proba(transform(
                    features, attacker["mean"], attacker["scale"], attacker["preprocessing"]["clip"]))

        @lru_cache(maxsize=None)
        def identity_outputs(key):
            return observe(split[key][0], split[key][1], identity, config, 0)

        def measure(policy, key, seed, reference=None):
            flows, groups, presence = split[key]
            features, results = (identity_outputs(key) if policy is identity
                                 else observe(flows, groups, policy, config, seed, labels[key]))
            predicted = chances(features) >= attacker["thresholds"]
            row = dict(exact_match=(predicted == presence).all(axis=1).mean(), **costs(results))
            # Scores and costs are measured against the identity shim on the same tunnels.
            # Unfinished transfers send less; that is not a saving, so get no overhead figure.
            reference = reference or row
            row["exact_match_drop_percent"] = 100 * (1 - row["exact_match"]
                                                     / reference["exact_match"])
            plain_results = identity_outputs(key)[1]
            delays = np.concatenate([r["delay"] for r in results])
            plain_delays = np.concatenate([r["delay"] for r in plain_results])
            done = (delays >= 0) & (plain_delays >= 0)
            row["added_p99_delay_ms"] = (np.percentile(delays[done] - plain_delays[done], 99)
                                         / 1e6 if done.any() else np.nan)
            if row["feasible"]:
                row["overhead_percent"] = 100 * (row["wire_bytes"] / reference["wire_bytes"] - 1)
            return row

        @lru_cache(maxsize=None)
        def training_features():
            cache = ((config_path.parent / config["model"]["output_root"]).resolve()
                     / folder.name / name)
            return identity_features(config, folder / environment, "train",
                                     split["train"][0], split["train"][1], cache)

        context = SimpleNamespace(
            flows=split["train"][0], groups=split["train"][1], presence=split["train"][2],
            labels=labels["train"], captures=archives["train"]["captures"],
            workload=config["workload"],
            budget=settings["budget"],
            features=training_features, attacker=chances, workers=config["workers"],
            featurize=lambda trace: extract_features(trace, config["feature_extraction"]))
        reference = {key: measure(identity, key, 0) for key in ("val", "test")}
        report = dict(identity=dict(candidate={}, within_budget=True,
                                    validation=[reference["val"]],
                                    test=average([reference["test"]])))
        for policy_name in names:
            base = config["obfuscators"][policy_name]
            fitted = []
            for candidate in base["candidates"]:
                print(f"Fitting {folder.name}/{environment} {policy_name}: {candidate}", flush=True)
                fitted.append(POLICIES[policy_name].fit({**base, **candidate}, config["shim"], context))
            rows = [dict(measure(policy, "val", 0, reference["val"]), candidate=candidate)
                    for policy, candidate in zip(fitted, base["candidates"])]
            chosen = select(rows, settings["budget"], settings["selection_margin"])
            tested = [measure(fitted[chosen], "test", seed, reference["test"])
                      for seed in settings["seeds"]]
            for row in tested:
                row["within_budget"] = bool(row["feasible"] and
                    row.get("overhead_percent", np.inf) <= settings["budget"]["overhead_percent"]
                    and row["added_p99_delay_ms"] <= settings["budget"]["added_p99_delay_ms"])
            report[policy_name] = dict(
                candidate=base["candidates"][chosen], within_budget=rows[chosen]["within_budget"],
                test_within_budget=all(row["within_budget"] for row in tested),
                calibration=[getattr(policy, "training_metadata", {}) for policy in fitted],
                validation=rows, test=average(tested))

        print(f"{folder.name}/{environment}: {len(split['test'][1])} test tunnels")
        for policy_name, entry in report.items():
            mean, std = entry["test"]["mean"], entry["test"]["std"]
            cost = (f"overhead {mean['overhead_percent']:+.1f}%" if entry["test"]["feasible"]
                    else f"UNFINISHED: {mean['chunks_completed_percent']:.1f}% of chunks done")
            print(f"  {policy_name:9s} exact match {mean['exact_match']:.1%}"
                  f"±{std['exact_match']:.1%} "
                  f"(drop {mean['exact_match_drop_percent']:.1f}%), {cost}, "
                  f"added p99 delay {mean['added_p99_delay_ms']:.2f} ms, "
                  f"{'within' if entry.get('test_within_budget', entry['within_budget']) else 'OUTSIDE'} "
                  f"test budget, "
                  f"settings {entry['candidate']}", flush=True)

        destination = (config_path.parent / settings["output_root"]).resolve() / folder.name / name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(json.dumps(dict(
            dataset=folder.name, environment=environment, classes=classes,
            attacker="frozen multi-label MLP trained on identity tunnels",
            feature_schema=attacker["feature_schema"], feature_count=attacker["feature_count"],
            source_support=support, shim=config["shim"], budget=settings["budget"],
            workload={key: dict(tunnels=len(groups), prevalence=presence.mean(axis=0).tolist())
                      for key, (_, groups, presence) in split.items()},
            policies=report), indent=2, default=float) + "\n")
        plot(report, f"{folder.name} / {environment}", destination / "comparison.pdf")


def plot(report, title, path):
    names = list(report)
    ticks = np.arange(len(names))
    fig, (scores, cost) = plt.subplots(1, 2, figsize=(11, 4))
    scores.errorbar(ticks, [report[name]["test"]["mean"]["exact_match"] for name in names],
                    yerr=[report[name]["test"]["std"]["exact_match"] for name in names],
                    color="#2a78d6", linestyle="none", marker="o", markersize=7, capsize=3)
    for tick, name in zip(ticks[1:], names[1:]):
        mean = report[name]["test"]["mean"]
        scores.annotate(f"−{mean['exact_match_drop_percent']:.0f}%", (tick, mean["exact_match"]),
                        textcoords="offset points", xytext=(10, 0), va="center")
    scores.set(ylabel="Frozen attacker exact match", ylim=(0, 1.05),
               title="Exact match (drop vs identity)")
    shaped = [name for name in names[1:] if report[name]["test"]["feasible"]]
    bars = cost.bar([names.index(name) for name in shaped],
                    [report[name]["test"]["mean"]["overhead_percent"] for name in shaped],
                    width=0.6, color="#2a78d6")
    cost.bar_label(bars, fmt="%.0f%%", padding=2)
    cost.set(ylabel="Extra wire bytes over identity (%)", yscale="symlog", title="Bandwidth overhead")
    cost.set_yscale("symlog", linthresh=1)
    cost.margins(y=0.2)
    for ax in (scores, cost):
        ax.set_xticks(ticks, names)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Three-stream tunnels: {title}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", action="append", choices=list(POLICIES))
    parser.add_argument("--config", action="append", type=Path)
    arguments = parser.parse_args()
    for path in arguments.config or [ROOT / "configs" / f"{name}.yaml"
                                     for name in ("iscx", "vnat")]:
        evaluate(path.resolve(), arguments.policy or list(POLICIES))
