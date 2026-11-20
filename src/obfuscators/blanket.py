"""Blanket-inspired policy (Nasr et al., USENIX Security 2021): blind adversarial perturbations.

A generator G, one hidden layer of hidden_units ReLU units (paper: 500), maps a
random trigger z ~ U[0,1]^trigger_size to four vectors (l, v, a, g) over the first
`length` chunk positions of a tunnel, both directions in arrival order. G(z) never
sees the traffic: at session start the sender draws z and shares G(z) with the
receiver, then shapes chunk i as it arrives, without buffering or knowing later
chunks (paper Sec. 4). Different z give different perturbations.

Remapping functions M enforce the network constraints (Sec. 5):
  insert  Alg. 4: the `inserts` positions with the largest |l_i| get a dummy record
          when chunk i is sent, client to relay if l_i > 0, else relay to client,
          with a body of sigmoid(v_i) of the room in one packet
  size    Alg. 3: positions in descending order of a_i are padded by
          clip(a_i, 0, 1) * max_pad_bytes until pad_budget_bytes is spent; padding
          never adds a packet (no cell sizes, s = 1)
  timing  chunk i is held d_i: the ranks of g map onto quantiles of a zero-mean
          Laplace law of standard deviation delay_ms, so the jitter follows it
          exactly (the GAN regularizer of Alg. 2, made exact; eq. 10 bounds hold).
          Negative values send at once and each direction keeps its order.

Training maximizes the guide's binary cross-entropy on remapped training tunnels
against their true classes (eq. 9, untargeted; the clean label stands in for f(x)).
The paper backpropagates through the classifier and custom gradients of M (eq.
11-12). Here the guide sees hand-crafted features of a simulated trace, neither
differentiable, so the gradient with respect to the raw output G(z) is estimated by
antithetic finite differences along a random direction, per tunnel with its own z,
and backpropagated exactly through G, with Adam. `guide: attacker` is the paper's
white-box setting; `proxy` is its black-box transfer setting (Sec. 9). Zero
training steps leave G random.
"""
from functools import partial
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from simulation import DUMMY, Plan, records, simulate


def fit(settings, shim, context):
    rng = np.random.default_rng(settings["random_seed"])
    hidden, size = settings["hidden_units"], settings["trigger_size"]
    weights = dict(w1=rng.normal(0, 1 / np.sqrt(size), (size, hidden)), b1=np.zeros(hidden),
                   w2=rng.normal(0, 1 / np.sqrt(hidden), (hidden, 4 * settings["length"])),
                   b2=np.zeros(4 * settings["length"]))
    history = learn(settings, shim, context, weights, rng) if settings["training_steps"] else []
    weights = {key: value.astype(np.float32) for key, value in weights.items()}
    policy = partial(plan, settings=settings, weights=weights)
    policy.training_metadata = dict(guide=settings["guide"] if history else "none",
                                    history=history)
    return policy


def generate(weights, z):
    """G(z) and its hidden layer."""
    hidden = np.maximum(z @ weights["w1"] + weights["b1"], 0)
    return hidden @ weights["w2"] + weights["b2"], hidden


def plan(arrivals, shim, rng, settings, weights):
    trigger = rng.random((1, settings["trigger_size"]))
    return perturb(arrivals, shim, rng, settings, generate(weights, trigger)[0][0])


def perturb(arrivals, shim, rng, settings, output):
    """Apply the remapping functions M to a raw generator output."""
    l, v, a, g = output.reshape(4, -1).astype(float)
    length, count = len(l), len(arrivals.arrival)
    n = min(length, count)
    body = shim["mtu_bytes"] - shim["transport_header_bytes"]
    room = body - shim["frame_header_bytes"] - shim["encryption_bytes"]

    # Timing: Laplace quantiles at the ranks of g, held per chunk, order kept per direction.
    u = (np.argsort(np.argsort(g)) + 0.5) / length - 0.5
    hold = -settings["delay_ms"] * 1e6 / np.sqrt(2) * np.sign(u) * np.log1p(-2 * np.abs(u))
    ready = arrivals.arrival.copy()
    ready[:n] += np.rint(np.maximum(hold[:n], 0)).astype(np.int64)
    for side in (0, 1):
        ready[arrivals.direction == side] = np.maximum.accumulate(
            ready[arrivals.direction == side])

    # Size: spend the padding budget in descending order of a, never adding a packet.
    order = np.argsort(-a)
    want = np.floor(np.clip(a[order], 0, 1) * settings["max_pad_bytes"])
    pad = np.zeros(length)
    pad[order] = np.clip(settings["pad_budget_bytes"] - (np.cumsum(want) - want), 0, want)
    spare = -(arrivals.payload[:n] + shim["frame_header_bytes"] + shim["encryption_bytes"]) % body
    filler = np.zeros(count, dtype=np.int64)
    filler[:n] = np.minimum(pad[:n], spare)

    # Insertion: dummies after the chunks at the top positions of |l|.
    top = np.sort(np.argsort(-np.abs(l))[:settings["inserts"]])
    top = top[top < count]
    dummies = records(len(top), direction=(l[top] <= 0).astype(int), ready=ready[top],
                      stream=-1, kind=DUMMY,
                      filler=np.rint(room / (1 + np.exp(-v[top]))).astype(np.int64))
    data = records(count, direction=arrivals.direction, ready=ready, stream=arrivals.stream,
                   start=arrivals.offset, payload=arrivals.payload, filler=filler)
    return Plan(np.insert(data, top + 1, dummies), False)


def trial(streams, output, settings, shim, featurize):
    """Features of one training tunnel under a raw generator output."""
    policy = partial(perturb, settings=settings, output=output)
    return featurize(simulate(streams, policy, shim, np.random.default_rng(0)).trace)


def learn(settings, shim, context, weights, rng):
    """Offline training of G by Adam ascent on the guide's loss; updates weights in place."""
    originals = context.features()
    logged = np.sign(originals) * np.log1p(np.abs(originals))
    mean, scale = logged.mean(axis=0), np.maximum(logged.std(axis=0), 1e-8)
    if settings["guide"] == "proxy":
        proxy = MLPClassifier(hidden_layer_sizes=(64,), batch_size=128, max_iter=500,
                              alpha=0.001, random_state=settings["random_seed"] + 17)
        z = np.clip((logged - mean) / scale, -6, 6)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            proxy.fit(z, context.presence.astype(int))
        guide = lambda x: proxy.predict_proba(
            np.clip((np.sign(x) * np.log1p(np.abs(x)) - mean) / scale, -6, 6))
    elif settings["guide"] == "attacker":
        guide = context.attacker
    else:
        raise ValueError("guide must be proxy or attacker")

    groups, batch, step_size = np.asarray(context.groups), settings["batch_size"], settings["smoothing"]
    moment = {key: np.zeros_like(value) for key, value in weights.items()}
    variance = {key: np.zeros_like(value) for key, value in weights.items()}
    history = []
    for step in range(1, settings["training_steps"] + 1):
        ids = rng.integers(0, len(groups), batch)
        trigger = rng.random((batch, settings["trigger_size"]))
        output, hidden = generate(weights, trigger)
        direction = rng.standard_normal(output.shape)
        features = Parallel(n_jobs=getattr(context, "workers", 1))(
            delayed(trial)([context.flows[i] for i in groups[index]], raw, settings, shim,
                           context.featurize)
            for index, raw in zip(np.r_[ids, ids], np.r_[output + step_size * direction,
                                                           output - step_size * direction]))
        chance = np.clip(guide(np.stack(features)), 1e-6, 1 - 1e-6)
        truth = np.tile(context.presence[ids], (2, 1))
        loss = -np.where(truth, np.log(chance), np.log1p(-chance)).mean(axis=1)
        # d loss / d G(z) per tunnel, then exactly through G; ascend with Adam.
        slope = ((loss[:batch] - loss[batch:]) / (2 * step_size))[:, None] * direction
        back = slope @ weights["w2"].T * (hidden > 0)
        gradients = dict(w1=trigger.T @ back / batch, b1=back.mean(axis=0),
                         w2=hidden.T @ slope / batch, b2=slope.mean(axis=0))
        for key, gradient in gradients.items():
            moment[key] = 0.9 * moment[key] + 0.1 * gradient
            variance[key] = 0.999 * variance[key] + 0.001 * gradient ** 2
            weights[key] += (settings["learning_rate"] * moment[key] / (1 - 0.9 ** step)
                             / (np.sqrt(variance[key] / (1 - 0.999 ** step)) + 1e-8))
        history.append(dict(step=step, loss=float(loss.mean())))
    return history
