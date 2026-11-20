"""Minos-inspired policy (Minos, USENIX ATC 2025).

  Schedule module  one queue per stream, served round-robin, so concurrent
                   streams interleave on the shared tunnel
  Flow count       streams seen within flow_timeout_ms; morphing runs while fewer
                   than flow_threshold are active
  Padding module   random padding inside the record, never enough to add a packet
  Dummy module     a dummy record follows a real one with dummy_probability during
                   each stream's first dummy_window_packets
"""
from functools import partial

import numpy as np

from simulation import DUMMY, Plan, records


def fit(settings, shim, context):
    return partial(plan, settings=settings)


def plan(arrivals, shim, rng, settings):
    count = len(arrivals.arrival)
    # Streams with a chunk in either direction within the timeout, counted at arrival.
    active = np.zeros(count, dtype=int)
    for stream in np.unique(arrivals.stream):
        seen = arrivals.arrival[arrivals.stream == stream]
        last = np.searchsorted(seen, arrivals.arrival, side="right") - 1
        active += (last >= 0) & (arrivals.arrival - seen[last]
                                 <= settings["flow_timeout_ms"] * 1e6)
    morphing = active < settings["flow_threshold"]

    size = arrivals.payload + shim["frame_header_bytes"] + shim["encryption_bytes"]
    spare = -size % (shim["mtu_bytes"] - shim["transport_header_bytes"])  # room in the last packet
    low, high = settings["padding_bytes"]
    padding = np.minimum(rng.integers(low, high + 1, count), spare) * morphing
    covered = (morphing & (arrivals.position < settings["dummy_window_packets"])
               & (rng.random(count) < settings["dummy_probability"]))
    low, high = settings["dummy_bytes"]
    data = records(count, direction=arrivals.direction, ready=arrivals.arrival,
                   stream=arrivals.stream, start=arrivals.offset, payload=arrivals.payload,
                   filler=padding)
    dummies = records(covered.sum(), direction=arrivals.direction[covered], kind=DUMMY,
                      glued=True, filler=rng.integers(low, high + 1, covered.sum()))
    return Plan(np.insert(data, np.flatnonzero(covered) + 1, dummies), True)
