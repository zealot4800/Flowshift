"""Securitas-inspired policy (Securitas, NSDI 2026).

The paper fragments IP packets and inserts small-TTL fake packets; here the same
choices act on records. A strategy holds one integer per chunk position within a
stream: 0 keeps the chunk, and x > 0 does one of two things, fixed per position:

  SPLIT(x)   two data records of x and L - x payload bytes
  INSERT(x)  the data record followed by a dummy record of x bytes; the relay may
             answer a client dummy with an ICMP-sized reply. A split that does not
             fit (x >= L) falls back to an insertion, as in the paper.

Strategies are either `random` or trained by REINFORCE through a 32-unit
autoregressive LSTM. A separate training-only proxy MLP supplies feedback;
`guide: attacker` is explicitly the ideal white-box condition. The simulator
is not differentiated. This remains a multi-label application-shim adaptation.
"""
from functools import partial
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from lstm_agent import SequencePolicy
from simulation import DUMMY, REPLY, Plan, records, simulate, wire_bytes


def fit(settings, shim, context):
    rng = np.random.default_rng(settings["random_seed"])
    if settings["strategy"] == "learned":
        strategy, metadata = learn(settings, shim, context, rng)
    else:
        strategy = rng.integers(0, settings["max_action_bytes"] + 1, settings["strategy_length"])
        metadata = dict(trainer="untrained random strategy", guide="none")
    policy = partial(plan, settings=settings, strategy=strategy)
    policy.training_metadata = dict(metadata, strategy=strategy.tolist())
    return policy


def plan(arrivals, shim, rng, settings, strategy):
    count, payload = len(arrivals.arrival), arrivals.payload
    known = arrivals.position < len(strategy)
    x = np.zeros(count, dtype=np.int64)
    x[known] = strategy[arrivals.position[known]]
    # Algorithm 1 draws split-or-insert from a fixed seed, so it is fixed per position.
    splitting = np.zeros(count, dtype=bool)
    splitting[known] = (np.random.default_rng(0).random(len(strategy))
                        < settings["split_probability"])[arrivals.position[known]]
    acted = ((arrivals.direction == 0) | settings["both_directions"]) & (x > 0)
    split = acted & splitting & (x < payload)
    insert = acted & ~split

    # Up to three records per chunk, in order: first part, second part, dummy.
    table = records((count, 3), direction=arrivals.direction[:, None],
                    stream=arrivals.stream[:, None], ready=arrivals.arrival[:, None])
    table["start"][:, 0], table["payload"][:, 0] = arrivals.offset, np.where(split, x, payload)
    table["start"][:, 1], table["payload"][:, 1] = arrivals.offset + x, payload - x
    table["filler"][:, 2], table["kind"][:, 2] = x, DUMMY
    table["glued"][:, 1:] = True
    used = np.column_stack([np.ones(count, dtype=bool), split, insert])

    # Some client dummies draw an ICMP-sized reply (paper Section 4.2), arriving at
    # a uniform time between the dummy and the stream's next server packet; with
    # no later server packet it follows the dummy at once.
    replied = insert & (arrivals.direction == 0) & (rng.random(count)
                                                    < settings["reply_probability"])
    gap = np.zeros(count)
    for stream in np.unique(arrivals.stream[replied]):
        mine = arrivals.stream == stream
        server = arrivals.arrival[mine & (arrivals.direction == 1)]
        later = np.searchsorted(server, arrivals.arrival[mine], side="right")
        known = later < len(server)
        gap[np.flatnonzero(mine)[known]] = server[later[known]] - arrivals.arrival[mine][known]
    replies = records(replied.sum(), direction=1, kind=REPLY,
                      after=(np.cumsum(used) - 1).reshape(count, 3)[replied, 2],
                      ready=rng.random(replied.sum()) * gap[replied],
                      filler=settings["reply_bytes"])
    return Plan(np.concatenate([table[used], replies]), False)


def observe(streams, strategy, settings, shim, entropy, featurize):
    """One training tunnel under a strategy: features, wire bytes, unsent and total payload."""
    result = simulate(streams, partial(plan, settings=settings, strategy=strategy), shim,
                      np.random.default_rng(entropy))
    return (featurize(result.trace), sum(result.bytes.values()), result.outstanding,
            sum(stream[:, 2].sum() for stream in streams))


def learn(settings, shim, context, rng):
    """Offline LSTM policy gradient; only context's training tunnels are consulted.

    The paper's printed Similar distance contradicts its prose. We explicitly
    minimize 1/(1 + standardized feature distance), a bounded similarity, rather
    than claiming to reproduce the inconsistent formula literally.
    """
    started = time.monotonic()
    originals = context.features()
    logged = np.sign(originals) * np.log1p(np.abs(originals))
    mean, scale = logged.mean(axis=0), logged.std(axis=0)
    scale[scale < 1e-8] = 1

    def standardize(features):
        return np.clip((np.sign(features) * np.log1p(np.abs(features)) - mean) / scale, -6, 6)

    converged = None
    if settings["guide"] == "proxy":
        proxy = MLPClassifier(hidden_layer_sizes=(64,), batch_size=128,
                              max_iter=settings.get("proxy_max_epochs", 500),
                              alpha=0.001, random_state=settings["random_seed"] + 17)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            proxy.fit(standardize(originals), context.presence.astype(int))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        guide = lambda features: proxy.predict_proba(standardize(features))
    elif settings["guide"] == "attacker":
        guide = context.attacker
    else:
        raise ValueError("guide must be proxy or attacker")
    length, batch = settings["strategy_length"], settings["batch_size"]
    policy = SequencePolicy(settings["max_action_bytes"] + 1,
                            settings.get("hidden_units", 32), seed=settings["random_seed"])
    groups = np.asarray(context.groups)
    plain = np.array([wire_bytes(np.concatenate([context.flows[i] for i in group])[:, 2],
                                shim)[1].sum() for group in groups])
    original_z = standardize(originals)
    prior = context.presence.mean(axis=0)
    # Strategy positions that act on some stream of each tunnel; only these get credit.
    eligible = np.zeros((len(groups), length), dtype=bool)
    for row, group in enumerate(groups):
        for i in group:
            outbound = (context.flows[i][:length, 1] == 0) | settings["both_directions"]
            eligible[row, :len(outbound)] |= outbound

    def losses(ids, strategies, entropy):
        outcomes = Parallel(n_jobs=getattr(context, "workers", 1))(
            delayed(observe)([context.flows[i] for i in groups[index]], strategy, settings,
                             shim, [entropy, int(index)], context.featurize)
            for index, strategy in zip(ids, strategies))
        features, wire, outstanding, payload = (np.array(column) for column in zip(*outcomes))
        extra, incomplete = wire / plain[ids] - 1, outstanding / np.maximum(payload, 1)
        probability = np.clip(guide(features), 1e-6, 1 - 1e-6)
        if settings.get("objective", "true_confidence") == "prior_confusion":
            belief = -(prior * np.log(probability) + (1 - prior) * np.log1p(-probability)).mean(1)
        else:
            belief = np.where(context.presence[ids], probability, 1 - probability).mean(1)
        distance = np.linalg.norm(standardize(features) - original_z[ids], axis=1)
        distance /= np.maximum(np.linalg.norm(original_z[ids], axis=1), 1)
        similarity = 1 / (1 + distance)
        loss = (belief + settings["overhead_weight"] * np.asarray(extra)
                + settings.get("similarity_weight", 0.1) * similarity
                + 10 * np.asarray(incomplete))
        return loss, dict(loss=float(loss.mean()), belief=float(belief.mean()),
                          overhead=float(np.mean(extra)), similarity=float(similarity.mean()))

    probe = rng.choice(len(groups), min(16, len(groups)), replace=False)
    best_strategy = np.zeros(length, dtype=int)
    _, initial = losses(probe, np.tile(best_strategy, (len(probe), 1)), 0)
    best, history = initial["loss"], [dict(step=0, **initial)]
    with threadpool_limits(limits=1):
        for step in range(1, settings["training_steps"] + 1):
            ids = rng.integers(0, len(groups), batch)
            actions, tape = policy.forward(batch, length, rng)
            loss, details = losses(ids, actions, step)
            gradients = policy.gradients(tape, loss - loss.mean(), eligible[ids],
                                         settings.get("entropy_weight", 0.001))
            norm = policy.update(gradients, settings["learning_rate"])
            if step % 10 == 0 or step == settings["training_steps"]:
                greedy = policy.forward(1, length, greedy=True)[0][0]
                _, probe_score = losses(probe, np.tile(greedy, (len(probe), 1)), 0)
                history.append(dict(step=step, gradient_norm=norm, **details,
                                    probe_loss=probe_score["loss"]))
                if probe_score["loss"] < best:
                    best, best_strategy = probe_score["loss"], greedy.copy()
    return best_strategy, dict(
        trainer="autoregressive LSTM REINFORCE with Adam", hidden_units=policy.hidden,
        embedding_units=16, learning_rate=settings["learning_rate"],
        max_action_bytes=settings["max_action_bytes"],
        guide=settings["guide"], proxy_model="independent MLP(64)" if converged is not None else None,
        proxy_converged=converged, objective=settings.get("objective", "true_confidence"),
        proxy_epochs=proxy.n_iter_ if converged is not None else None,
        proxy_training_loss=float(proxy.loss_) if converged is not None else None,
        similarity="1/(1 + normalized standardized feature distance)",
        training_tunnels=len(groups), steps=settings["training_steps"], batch_size=batch,
        initial_probe_loss=initial["loss"], best_probe_loss=best,
        seconds=time.monotonic() - started, history=history)
