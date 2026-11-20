"""Multi-stream tunnel workloads shared by attacker training and every policy.

One sample is one tunnel of streams_per_tunnel distinct flows that all start at
time 0. Its target is the set of service classes present. Tunnels are built inside
one split, so the capture-disjoint train/validation/test separation is preserved.
"""
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from features import extract_features
from simulation import simulate


def load(folder, split):
    """Flows ([ns, direction, bytes] packets), class labels and capture names of one split."""
    with np.load(folder / f"{split}.npz", allow_pickle=False) as archive:
        if str(archive["time_unit"]) != "ns" or str(archive["schema"]) != "packet-npz-v1":
            raise ValueError(f"{folder}/{split}: expected packet-npz-v1 in nanoseconds")
        flows = np.split(archive["packets"], archive["offsets"][1:-1])
        return flows, archive["labels"], archive["captures"]


def validate_splits(folder):
    """Check capture separation before any grouping or fitting, without loading packets."""
    seen = set()
    support = {}
    for split in ("train", "val", "test"):
        with np.load(folder / f"{split}.npz", allow_pickle=False) as archive:
            captures, labels = archive["captures"], archive["labels"]
            current = set(captures.tolist())
            if current & seen:
                raise ValueError(f"{folder}: captures overlap across splits")
            seen.update(current)
            support[split] = dict(flows=len(labels), captures=len(current),
                                  class_flows=np.bincount(labels, minlength=4).tolist(),
                                  class_captures=[len(np.unique(captures[labels == i]))
                                                  for i in range(4)])
    return support


def build(labels, captures, classes, rounds, settings, streams, seed):
    """Group flows into tunnels; returns flow indices (n, streams) and presence (n, classes).

    Each round adds one tunnel for every nonempty class subset that fits the stream
    count: one flow per selected class, the remaining slots filled from classes of
    the subset chosen uniformly. Flows are taken least-used first, preferring a
    capture not yet in the tunnel, and no flow joins more than max_flow_reuse
    tunnels; a subset that can no longer be filled is skipped, so rare classes end
    up under-represented rather than over-reused.
    """
    rng = np.random.default_rng(seed)
    subsets = [subset for size in range(1, streams + 1)
               for subset in combinations(range(classes), size)]
    uses = np.zeros(len(labels), dtype=int)
    groups = []
    for _ in range(rounds):
        for subset in subsets:
            group = []
            for label in list(subset) + list(rng.choice(subset, streams - len(subset))):
                pool = np.flatnonzero((labels == label) & (uses < settings["max_flow_reuse"]))
                pool = pool[~np.isin(pool, group)]
                if not len(pool):
                    break
                repeat = np.isin(captures[pool], captures[group])
                group.append(pool[np.lexsort((rng.random(len(pool)), uses[pool], repeat))[0]])
            else:
                uses[group] += 1
                groups.append(rng.permutation(group))
    groups = np.array(groups)
    presence = np.stack([(labels[groups] == label).any(axis=1) for label in range(classes)], axis=1)
    return groups, presence


def run(streams, policy, config, entropy, services=None):
    """One tunnel: observer features plus what the cost metrics need."""
    result = simulate(streams, policy, config["shim"], np.random.default_rng(entropy), services)
    return extract_features(result.trace, config["feature_extraction"]), dict(
        delay=result.delay, outstanding=result.outstanding, bytes=result.bytes,
        delivered_payload_bytes=sum(flow[:, 2].sum() for flow in streams) - result.outstanding,
        packets=len(result.trace), duration_ns=(result.trace[-1, 0] - result.trace[0, 0]
                                               if len(result.trace) else 0))


def observe(flows, groups, policy, config, seed, labels=None):
    """Simulate every tunnel under a policy; returns observer features and per-tunnel results.

    Tunnel i draws its randomness from (seed, i), so the outcome does not depend on
    how the work is spread over processes. Traces are not kept. With labels, each
    stream's class is passed to the sender, which knows its own applications.
    """
    outcomes = Parallel(n_jobs=config["workers"])(
        delayed(run)([flows[flow] for flow in group], policy, config, [seed, index],
                     None if labels is None else labels[group])
        for index, group in enumerate(groups))
    features, results = zip(*outcomes)
    return np.stack(features), results
