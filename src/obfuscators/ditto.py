"""Ditto-inspired policy (Ditto, NDSS 2022), as the Securitas paper simulates it.

A pattern of pattern_length record sizes comes from size percentiles of the
training chunks (Ditto Eq. 2), capped so each fits one packet. Every chunk joins
the queue of the smallest state that holds it and is padded to that size; a chunk
larger than every state is cut into pieces of the largest one. While anything is
queued in a direction, records leave back to back at the link rate and cycle
through the pattern; a state whose queue is empty sends a chaff record of its
size instead (Ditto's round robin over priority queue pairs). The cycle keeps its
place across idle gaps, but nothing is sent while nothing is queued, so periods
of activity stay visible: unlike the original constant-rate Ditto.
"""
from collections import deque
from functools import partial

import numpy as np

from simulation import DATA, DUMMY, Plan, records, wire_bytes


def fit(settings, shim, context):
    length = settings["pattern_length"]
    sizes = np.concatenate([flow[:, 2] for flow in context.flows])
    states = np.percentile(sizes, 100 * np.arange(1, length + 1) / length).astype(np.int64)
    room = (shim["mtu_bytes"] - shim["transport_header_bytes"] - shim["frame_header_bytes"]
            - shim["encryption_bytes"])
    bodies = np.minimum(states, room)
    policy = partial(plan, bodies=bodies)
    policy.training_metadata = dict(pattern=bodies.tolist())
    return policy


def plan(arrivals, shim, rng, bodies):
    ns_per_byte = 8000 / shim["link_mbps"]
    sizes = np.unique(bodies)  # one queue per distinct state size
    slot_queue = np.searchsorted(sizes, bodies)
    slot_ns = np.rint(wire_bytes(bodies, shim)[1] * ns_per_byte).astype(np.int64)
    tables = []
    for direction in (0, 1):
        mine = arrivals.direction == direction
        # Pieces no larger than the largest state, in arrival order.
        pieces = np.maximum(-(-arrivals.payload[mine] // sizes[-1]), 1)
        first = np.repeat(np.cumsum(pieces) - pieces, pieces)
        index = np.arange(pieces.sum()) - first
        payload = np.repeat(arrivals.payload[mine], pieces)
        part = np.minimum(payload - index * sizes[-1], sizes[-1])
        start = np.repeat(arrivals.offset[mine], pieces) + index * sizes[-1]
        arrival = np.repeat(arrivals.arrival[mine], pieces).tolist()
        queue_of = np.searchsorted(sizes, part).tolist()
        queues = [deque() for _ in sizes]
        rows, clock, slot, next_piece, waiting = [], 0, 0, 0, 0
        while next_piece < len(arrival) or waiting:
            if not waiting:
                clock = max(clock, arrival[next_piece])
            while next_piece < len(arrival) and arrival[next_piece] <= clock:
                queues[queue_of[next_piece]].append(next_piece)
                next_piece, waiting = next_piece + 1, waiting + 1
            state = slot % len(bodies)
            queue = queues[slot_queue[state]]
            piece = queue.popleft() if queue else -1
            waiting -= piece >= 0
            rows.append((clock, piece))
            clock, slot = clock + slot_ns[state], slot + 1
        ready, piece = np.array(rows, dtype=np.int64).reshape(-1, 2).T
        real = piece >= 0
        body = bodies[np.arange(len(piece)) % len(bodies)]
        carried = np.where(real, part[piece], 0)
        tables.append(records(len(piece), direction=direction, ready=ready,
                              start=np.where(real, start[piece], 0), payload=carried,
                              filler=body - carried, kind=np.where(real, DATA, DUMMY)))
    return Plan(np.concatenate(tables), False)
