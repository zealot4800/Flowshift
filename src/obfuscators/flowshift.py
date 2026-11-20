"""FlowShift (this work): cooperative, reversible, target-driven reshaping.

The endpoints are trusted, so the sender knows the service of each stream. A
generator seeded per tunnel (seed xi) cuts time into holding intervals
dt_k ~ U[interval_ms] and draws, for each, a policy p_k = (S, T, V) from the
selected set P and a target service absent from the tunnel (Sec. V-D, eq. 6-8):

  S  size     each real chunk is mapped to the target's size at its quantile in
              its own service's size distribution and padded toward it by at
              most max_pad_bytes; larger chunks are forwarded, since every
              fragment would cost a full set of headers (eq. 2, pad and forward)
  V  volume   dummies replay a real training flow of the target service inside
              the interval, so packet count, sizes and directions follow the
              target, with a share `decoy` of its packets (eq. 3). `targets`
              picks one absent service at random (one), the one that misleads a
              proxy most per decoy byte (best), or every absent service (all)
  T  timing   real chunks are released at target-service gaps,
              r_j = max(a_j, r_{j-1} + delta_j), but never held over hold_ms (eq. 5)

Cost cap: padding and dummies never exceed overhead_floor_bytes plus
overhead_share of the real wire bytes sent so far in the tunnel; anything over
it is skipped. The floor lets small tunnels carry a decoy at all. The cap only looks at the
past, so a real sender can enforce it.

The receiver drops dummies and padding and reassembles fragments; the simulator
checks that every payload byte is delivered once and in order (eq. 15).

Policy selection (Sec. VI, eq. 16-21): every variant runs on tunnels of held-out
training captures, where the shared simulator measures bandwidth and delay
costs. Variants within the budget whose displacement D is within
displacement_slack of the best are kept if no other is cheaper in both
bandwidth and delay.

Changes to the paper's sketch: profiles are real training flows (not summary
statistics); targets are services absent from the tunnel, one or all at a time;
displacement is the error of independent proxy classifiers, trained like the
attacker but never on its weights, on capture-disjoint tunnels, rather than a raw
feature distance; among near-best variants only the cheapest are rotated;
holding is bounded so the delay budget holds; costs are simulated, not estimated.
"""
from functools import partial
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from metrics import thresholds
from simulation import DUMMY, Plan, identity, records, simulate, wire_bytes
from workload import build

PERCENTILES = np.linspace(0, 1, 101)


def fit(settings, shim, context):
    rng = np.random.default_rng(settings["random_seed"])
    profiles = build_profiles(settings, context.flows, context.labels,
                              context.presence.shape[1], rng)
    selected, table, efficiency = select(settings, shim, context, profiles, rng)
    policy = partial(plan, settings=settings, policies=selected, **profiles)
    policy.training_metadata = dict(selected=selected, candidates=table,
                                    target_rank=profiles["rank"], target_efficiency=efficiency,
                                    templates=[len(t) for t in profiles["templates"]])
    return policy


def build_profiles(settings, flows, labels, services, rng):
    """Per service: size and gap quantiles, and decoy templates (training flows)."""
    labels = np.asarray(labels)
    sizes, gaps, templates = [], [], []
    for service in range(services):
        mine = np.flatnonzero(labels == service)
        sizes.append(np.quantile(np.concatenate([flows[i][:, 2] for i in mine]), PERCENTILES))
        gaps.append(np.quantile(np.concatenate([np.diff(flows[i][:, 0]) for i in mine]),
                                PERCENTILES))
        small = [i for i in mine if flows[i][:, 2].sum() <= settings["template_bytes"]]
        chosen = rng.permutation(small)[:settings["template_count"]]
        templates.append([flows[i] for i in chosen])
    return dict(sizes=np.array(sizes), gaps=np.array(gaps), templates=templates,
                rank=list(range(services)))


def select(settings, shim, context, profiles, rng):
    """Transformation Policy Selector: feasible, non-dominated variants (eq. 19-21).

    Displacement D is the error rate (tunnels not exactly right), averaged over
    proxy models fitted on unshaped training tunnels and scored on tunnels of
    held-out training captures; the evaluated attacker is never queried.
    First each service is ranked by D gained per decoy byte when it alone is
    imitated (sets profiles["rank"] for targets: best). Feasible means complete
    delivery within the budget and D within displacement_slack of the best; the
    cheapest feasible variants are kept.
    """
    groups, presence = np.asarray(context.groups), context.presence
    originals = context.features()
    logged = np.sign(originals) * np.log1p(np.abs(originals))
    mean, scale = logged.mean(axis=0), np.maximum(logged.std(axis=0), 1e-8)

    def standardize(x):
        return np.clip((np.sign(x) * np.log1p(np.abs(x)) - mean) / scale, -6, 6)

    # Capture-disjoint scoring: per service, a quarter of its training captures
    # (at least one; a quarter of its flows if it has a single capture) is held
    # out. Proxies are fitted on tunnels without them; policies are scored on fresh
    # tunnels built from held-out flows only, the way the workload builds tunnels.
    labels, captures = np.asarray(context.labels), np.asarray(context.captures)
    unseen = np.zeros(len(labels), dtype=bool)
    for service in range(presence.shape[1]):
        mine = labels == service
        names = np.unique(captures[mine])
        if len(names) > 1:
            unseen |= mine & np.isin(captures, rng.choice(names, -(-len(names) // 4),
                                                          replace=False))
        else:
            unseen |= mine & (rng.random(len(labels)) < 0.25)
    fitted = ~unseen[groups].any(axis=1)
    pool = np.flatnonzero(unseen)
    local, truth = build(labels[pool], captures[pool], presence.shape[1], settings["tps_rounds"],
                         context.workload, shim["streams_per_tunnel"], settings["random_seed"])
    tunnels = pool[local]

    def costs(policy):
        """Per scoring tunnel: features, wire bytes, chunk delays."""
        outcomes = Parallel(n_jobs=getattr(context, "workers", 1))(
            delayed(trial)([context.flows[i] for i in group], policy, shim, labels[group],
                           [settings["random_seed"], index], context.featurize)
            for index, group in enumerate(tunnels))
        features, wire, delay = zip(*outcomes)
        return np.stack(features), np.array(wire), delay

    # Proxies follow the attacker's recipe, not its weights: per-class thresholds
    # maximize per-class accuracy on the unshaped scoring tunnels. Different models
    # are averaged so that D rewards changes that transfer between models.
    plain = costs(identity)
    seed, proxies = settings["random_seed"], []
    for model in (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=seed),
                  MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=seed + 1),
                  RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                         class_weight="balanced", random_state=seed)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(standardize(originals[fitted]), presence[fitted].astype(int))
        proxies.append((model, thresholds(chances(model, standardize(plain[0])), truth)))

    def judge(policy):
        """Proxy-averaged wrong per tunnel, wire bytes, chunk delays."""
        features, wire, delay = plain if policy is identity else costs(policy)
        z = standardize(features)
        wrong = np.mean([((chances(model, z) >= cuts) != truth).any(axis=1)
                         for model, cuts in proxies], axis=0)
        return wrong, wire, delay

    base = judge(identity)
    single = dict(decoy=1.0, targets="best", size_mapping=False, max_pad_bytes=0, hold_ms=0)
    efficiency = []
    for service in profiles["rank"]:
        wrong, wire, _ = judge(partial(plan, settings=settings, policies=[single],
                                       **dict(profiles, rank=[service])))
        efficiency.append(float((wrong.mean() - base[0].mean())
                                / max(wire.sum() - base[1].sum(), 1)))
    profiles["rank"] = [int(c) for c in np.argsort(efficiency)[::-1]]
    variants = settings["variants"]
    results = [judge(partial(plan, settings=settings, policies=[v], **profiles)) for v in variants]
    rows, reference = [], np.concatenate(base[2])
    for index, (wrong, wire, delay) in enumerate(results):
        delay = np.concatenate(delay)
        done = (delay >= 0) & (reference >= 0)
        rows.append(dict(
            variant=index, displacement=float(wrong.mean() - base[0].mean()),
            overhead_percent=float(100 * (wire.sum() / base[1].sum() - 1)),
            added_p99_delay_ms=float(np.percentile(delay[done] - reference[done], 99) / 1e6),
            complete=bool((delay >= 0).all())))
    feasible = [row for row in rows if row["complete"]
                and row["overhead_percent"] <= context.budget["overhead_percent"]
                and row["added_p99_delay_ms"] <= context.budget["added_p99_delay_ms"]]
    if feasible:
        floor = max(r["displacement"] for r in feasible)
        floor -= settings["displacement_slack"] * abs(floor)
        feasible = [row for row in feasible if row["displacement"] >= floor]
    keys = ("overhead_percent", "added_p99_delay_ms")
    cost = {id(row): np.array([row[k] for k in keys]) for row in feasible}
    pareto = [p for p in feasible if not any(
        (cost[id(q)] <= cost[id(p)]).all() and (cost[id(q)] < cost[id(p)]).any()
        for q in feasible)]
    if not pareto:  # nothing feasible: the cheapest variant that delivers everything
        pareto = [min([r for r in rows if r["complete"]] or rows,
                      key=lambda r: r["overhead_percent"])]
    for row in rows:
        row["selected"] = any(row is p for p in pareto)
    return [variants[row["variant"]] for row in pareto], rows, efficiency


def chances(model, x):
    """Presence probability per class from a multi-label sklearn model."""
    out = model.predict_proba(x)
    return np.column_stack([p[:, -1] for p in out]) if isinstance(out, list) else out


def trial(streams, policy, shim, services, entropy, featurize):
    """Features, wire bytes and per-chunk delays of one tunnel."""
    result = simulate(streams, policy, shim, np.random.default_rng(entropy), services)
    return featurize(result.trace), sum(result.bytes.values()), result.delay


def plan(arrivals, shim, rng, settings, policies, sizes, gaps, templates, rank):
    count = len(arrivals.arrival)
    absent = [c for c in rank if templates[c] and not (arrivals.service == c).any()]
    if not absent:
        return identity(arrivals, shim, rng)
    # Shared policy-duration sequence (eq. 6-8).
    low, high = np.array(settings["interval_ms"]) * 1e6
    horizon = shim["arrival_seconds"] * 1e9
    bounds = np.cumsum(np.r_[0, rng.uniform(low, high, int(horizon // low) + 1)])
    which = rng.integers(len(policies), size=len(bounds))
    best = np.array([policies[i]["targets"] == "best" for i in which])
    target = np.where(best, absent[0], np.array(absent)[rng.integers(len(absent),
                                                                     size=len(bounds))])
    decoy = np.array([policies[i]["decoy"] for i in which])
    mapping = np.array([policies[i]["size_mapping"] for i in which])
    max_pad = np.array([policies[i]["max_pad_bytes"] for i in which])
    hold = np.array([policies[i]["hold_ms"] for i in which]) * 1e6
    every = np.array([policies[i]["targets"] == "all" for i in which])
    k = np.searchsorted(bounds, arrivals.arrival, side="right") - 1
    aim = target[k]

    def per_class(classes, values, table, forward):
        """Quantile level of values in their class (forward) or the value at a level."""
        out = np.zeros(count)
        for c in np.unique(classes):
            m = classes == c
            out[m] = (np.interp(values[m], table[c], PERCENTILES) if forward
                      else np.interp(values[m], PERCENTILES, table[c]))
        return out

    # T: release at target gaps, bounded by hold_ms (eq. 5), per stream and direction.
    ready = arrivals.arrival.astype(float)
    delta = per_class(aim, rng.random(count), gaps, False)
    lane = arrivals.stream * 2 + arrivals.direction
    for key in np.unique(lane):
        mine = lane == key
        total = np.cumsum(delta[mine])
        ready[mine] = total + np.maximum.accumulate(arrivals.arrival[mine] - total)
    ready = np.minimum(ready, arrivals.arrival + hold[k])
    for key in np.unique(lane):  # keep each lane's order
        ready[lane == key] = np.maximum.accumulate(ready[lane == key])

    # S: quantile-map sizes to the target and pad toward it (eq. 2, pad or forward).
    room = (shim["mtu_bytes"] - shim["transport_header_bytes"] - shim["frame_header_bytes"]
            - shim["encryption_bytes"])  # padding never adds a packet
    goal = per_class(aim, per_class(arrivals.service, arrivals.payload, sizes, True), sizes, False)
    goal = np.minimum(goal.astype(np.int64), room)
    pad = np.where(mapping[k], np.clip(goal - arrivals.payload, 0, max_pad[k]), 0)

    # V: dummies replay a target-service flow inside the intervals that target it (eq. 3).
    dummies = [records(0)]
    for service in np.unique(np.r_[target, absent] if every.any() else target):
        flow = templates[service][rng.integers(len(templates[service]))]
        j = np.searchsorted(bounds, flow[:, 0], side="right") - 1
        keep = ((target[j] == service) | every[j]) & (rng.random(len(flow)) < decoy[j])
        dummies.append(records(keep.sum(), direction=flow[keep, 1], ready=flow[keep, 0],
                               stream=-1, kind=DUMMY, filler=flow[keep, 2]))
    dummies = np.concatenate(dummies)

    # Hard cost cap, enforceable online: padding and dummies, in wire bytes, never
    # exceed overhead_floor_bytes plus overhead_share of the real wire bytes that
    # have arrived so far.
    real = wire_bytes(arrivals.payload, shim)[1]
    times = np.r_[arrivals.arrival, dummies["ready"]]
    extra = np.r_[wire_bytes(arrivals.payload + pad, shim)[1] - real,
                  wire_bytes(dummies["filler"], shim)[1]]
    earned = settings["overhead_floor_bytes"] + settings["overhead_share"] * np.r_[
        0, np.cumsum(real)][np.searchsorted(arrivals.arrival, times, side="right")]
    allowed, spent = np.zeros(len(times), dtype=bool), 0
    for i in np.argsort(times, kind="stable").tolist():
        if extra[i] and spent + extra[i] <= earned[i]:
            allowed[i], spent = True, spent + extra[i]
    data = records(count, direction=arrivals.direction, ready=np.rint(ready).astype(np.int64),
                   stream=arrivals.stream, start=arrivals.offset, payload=arrivals.payload,
                   filler=np.where(allowed[:count], pad, 0))
    return Plan(np.concatenate([data, dummies[allowed[count:]]]), False)
