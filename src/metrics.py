"""Tunnel-level service-presence metrics and resource costs."""
import numpy as np
from sklearn.metrics import average_precision_score


def rates(truth, predicted):
    """Precision, recall, F1 and accuracy of one class; undefined ratios count as 0."""
    hit, miss = (truth & predicted).sum(), (truth & ~predicted).sum()
    alarm, quiet = (~truth & predicted).sum(), (~truth & ~predicted).sum()
    precision, recall = hit / max(hit + alarm, 1), hit / max(hit + miss, 1)
    return dict(precision=precision, recall=recall,
                f1=2 * hit / max(2 * hit + alarm + miss, 1),
                accuracy=(hit + quiet) / max(len(truth), 1))


def thresholds(probabilities, presence, metric="accuracy"):
    """Per-class threshold selected on validation tunnels. Maximizing each class's
    accuracy is the per-class route to exact match: a tunnel is exactly right only
    when every class is."""
    grid = np.linspace(0.05, 0.95, 19)
    scores = [[rates(truth, chance >= cut)[metric] for cut in grid]
              for truth, chance in zip(presence.T, probabilities.T)]
    scores = np.asarray(scores)
    # Flat validation regions do not justify moving the threshold to an extreme.
    scores -= 1e-12 * np.abs(grid - 0.5)
    return grid[np.argmax(scores, axis=1)]


def classification(probabilities, presence, cuts, classes):
    """Scores of class-presence predictions. Macro values average over classes;
    average precision skips a class that is always or never present."""
    predicted = probabilities >= cuts
    per_class = {}
    for name, truth, chance, guess in zip(classes, presence.T, probabilities.T, predicted.T):
        mixed = truth.any() and not truth.all()
        per_class[name] = dict(
            rates(truth, guess), prevalence=truth.mean(),
            average_precision=average_precision_score(truth, chance) if mixed else np.nan)
    macro = {key: np.nanmean([scores[key] for scores in per_class.values()])
             for key in ("average_precision", "f1")}
    return dict(label_accuracy=(predicted == presence).mean(),
                macro_average_precision=macro["average_precision"], macro_f1=macro["f1"],
                exact_match=(predicted == presence).all(axis=1).mean(), per_class=per_class)


def costs(results):
    """Delivery and wire cost of one condition over all its tunnels."""
    delay = np.concatenate([result["delay"] for result in results])
    done = delay >= 0
    spent = {name: sum(result["bytes"][name] for result in results)
             for name in results[0]["bytes"]}
    wire = sum(spent.values())
    late = np.percentile(delay[done], [95, 99]) / 1e6 if done.any() else [np.nan, np.nan]
    return dict(
        feasible=bool(done.all()),
        tunnels_completed_percent=100 * np.mean([(result["delay"] >= 0).all()
                                                 for result in results]),
        chunks_completed_percent=100 * done.mean(),
        outstanding_bytes=sum(result["outstanding"] for result in results),
        wire_bytes=wire, delivered_payload_bytes=sum(result.get("delivered_payload_bytes",
                                                             result["bytes"]["payload"])
                                                    for result in results),
        wire_bytes_per_payload_byte=wire / max(sum(result.get("delivered_payload_bytes",
                                                            result["bytes"]["payload"])
                                                   for result in results), 1),
        packets_per_chunk=sum(result["packets"] for result in results) / len(delay),
        mean_delay_ms=delay[done].mean() / 1e6 if done.any() else np.nan,
        p95_delay_ms=late[0], p99_delay_ms=late[1],
        mean_duration_s=np.mean([result["duration_ns"] for result in results]) / 1e9,
        bytes=spent)
