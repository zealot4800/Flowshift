"""Small NumPy LSTM for autoregressive categorical policy-gradient training.

The traffic simulator stays outside the differentiable graph. Each sampled
strategy receives a scalar loss; REINFORCE differentiates its log probability.
No GPU or additional learning-library dependency is required.
"""
import numpy as np
from scipy.special import expit


class SequencePolicy:
    def __init__(self, actions, hidden=32, embedding=16, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.weights = dict(
            embedding=rng.normal(0, 0.1, (actions, embedding)),
            recurrent=rng.normal(0, 1 / np.sqrt(hidden + embedding),
                                 (embedding + hidden, 4 * hidden)),
            bias=np.zeros(4 * hidden),
            output=rng.normal(0, 0.1, (hidden, actions)), output_bias=np.zeros(actions))
        self.weights["bias"][hidden:2 * hidden] = 1  # forget gate
        self.moment = {name: np.zeros_like(value) for name, value in self.weights.items()}
        self.variance = {name: np.zeros_like(value) for name, value in self.weights.items()}
        self.steps = 0

    def forward(self, batch, length, rng=None, actions=None, greedy=False):
        """Sample, greedily decode, or teacher-force strategies; retain a BPTT tape."""
        w, hidden = self.weights, self.hidden
        h, c = np.zeros((batch, hidden)), np.zeros((batch, hidden))
        previous = np.zeros(batch, dtype=int)
        sequence, tape = np.zeros((batch, length), dtype=int), []
        for position in range(length):
            joined = np.column_stack([w["embedding"][previous], h])
            gates = joined @ w["recurrent"] + w["bias"]
            i, f, o = np.split(expit(gates[:, :3 * hidden]), 3, axis=1)
            g = np.tanh(gates[:, 3 * hidden:])
            next_c = f * c + i * g
            next_h = o * np.tanh(next_c)
            logits = next_h @ w["output"] + w["output_bias"]
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            if actions is not None:
                chosen = actions[:, position]
            elif greedy:
                chosen = probabilities.argmax(axis=1)
            else:
                chosen = (rng.random((batch, 1)) > probabilities.cumsum(axis=1)).sum(axis=1)
                chosen = np.minimum(chosen, probabilities.shape[1] - 1)
            sequence[:, position] = chosen
            tape.append((previous, joined, c, i, f, o, g, next_c, next_h,
                         probabilities, chosen))
            previous, h, c = chosen, next_h, next_c
        return sequence, tape

    def gradients(self, tape, advantage, used=None, entropy=0.0):
        """Gradient of mean(advantage * log P(strategy)) minus entropy bonus."""
        w, hidden, batch = self.weights, self.hidden, len(advantage)
        gradients = {name: np.zeros_like(value) for name, value in w.items()}
        dh, dc = np.zeros((batch, hidden)), np.zeros((batch, hidden))
        if used is None:
            used = np.ones((batch, len(tape)))
        for position in range(len(tape) - 1, -1, -1):
            previous, joined, c, i, f, o, g, next_c, h, chance, chosen = tape[position]
            one_hot = np.zeros_like(chance)
            one_hot[np.arange(batch), chosen] = 1
            dl = (advantage * used[:, position])[:, None] * (one_hot - chance) / batch
            if entropy:
                logp = np.log(np.maximum(chance, 1e-12))
                dl += entropy * used[:, position, None] * chance * (
                    logp - (chance * logp).sum(axis=1, keepdims=True)) / batch
            gradients["output"] += h.T @ dl
            gradients["output_bias"] += dl.sum(axis=0)
            dh += dl @ w["output"].T
            tanh_c = np.tanh(next_c)
            dc += dh * o * (1 - tanh_c ** 2)
            di, df, do, dg = dc * g, dc * c, dh * tanh_c, dc * i
            gates = np.column_stack([di * i * (1 - i), df * f * (1 - f),
                                     do * o * (1 - o), dg * (1 - g ** 2)])
            gradients["recurrent"] += joined.T @ gates
            gradients["bias"] += gates.sum(axis=0)
            dj = gates @ w["recurrent"].T
            width = w["embedding"].shape[1]
            np.add.at(gradients["embedding"], previous, dj[:, :width])
            dh, dc = dj[:, width:], dc * f
        return gradients

    def update(self, gradients, rate, clip=1.0):
        """Adam update with global gradient clipping."""
        norm = np.sqrt(sum(np.square(value).sum() for value in gradients.values()))
        self.steps += 1
        for name, gradient in gradients.items():
            gradient = gradient * min(1, clip / max(norm, 1e-12))
            self.moment[name] = 0.9 * self.moment[name] + 0.1 * gradient
            self.variance[name] = 0.999 * self.variance[name] + 0.001 * gradient ** 2
            mean = self.moment[name] / (1 - 0.9 ** self.steps)
            variance = self.variance[name] / (1 - 0.999 ** self.steps)
            self.weights[name] -= rate * mean / (np.sqrt(variance) + 1e-8)
        return float(norm)
