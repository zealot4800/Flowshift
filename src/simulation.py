"""Shared application-transport shim simulator.

    application -> sender shim -> encrypted connection -> receiver shim -> peer

Every policy runs through `simulate`, so they share one traffic interpretation,
framing, packetization, link model and delivery accounting. Declared model:

  input      Each captured packet is one application message (a chunk) whose
             payload is its captured length; pure ACKs are chunks like any other.
             Direction 0, the direction of a flow's first packet, is client to relay.
  records    [frame header | body] under authenticated encryption. A record costs
             frame_header_bytes + encryption_bytes on top of its body, which holds
             real payload and filler (padding, or the body of a dummy or reply).
  packets    A record starts a new packet and continues over as many packets as
             it needs; each packet adds transport_header_bytes and fits the MTU.
             Real TCP may merge or cut records differently.
  link       One link per direction at link_mbps with no propagation delay, loss,
             congestion control, ACKs or handshake. A policy fixes the service
             order and earliest release times; it cannot send faster than the link.
  observer   Sees packet times, directions and sizes during [0, observation_seconds].
  delivery   A chunk completes when the record holding its last byte has left and
             every earlier chunk of its stream, in that direction, has completed.

Arrivals are replayed from the trace: delaying a request does not move its response.
"""
from collections import deque
from typing import NamedTuple

import numpy as np

DATA, DUMMY, REPLY = 0, 1, 2

RECORD = np.dtype([
    ("direction", "i8"),  # 0 client to relay, 1 relay to client
    ("ready", "i8"),      # earliest release in ns; after `after` completes if that is set
    ("after", "i8"),      # record that must complete first, or -1
    ("stream", "i8"),     # service queue under round-robin
    ("glued", "?"),       # sent back to back with the previous record, in the same turn
    ("kind", "i8"),       # DATA, DUMMY or REPLY
    ("start", "i8"),      # first payload byte carried, as an offset in its direction's bytes
    ("payload", "i8"),    # real bytes carried
    ("filler", "i8"),     # padding, or the body of a dummy or reply
])


class Chunks(NamedTuple):
    """Application arrivals of one tunnel, ordered by arrival time."""
    stream: np.ndarray
    position: np.ndarray   # index within its stream
    arrival: np.ndarray    # ns
    direction: np.ndarray
    payload: np.ndarray    # bytes
    offset: np.ndarray     # first byte's offset within its direction's bytes
    service: np.ndarray = None  # class of the chunk's stream, known to the trusted sender


class Plan(NamedTuple):
    records: np.ndarray    # RECORD table, in service order among equally ready records
    round_robin: bool      # serve backlogged streams in turn instead of first come first served


class Result(NamedTuple):
    trace: np.ndarray      # observed wire packets [ns, direction, bytes]
    delay: np.ndarray      # per chunk: completion - arrival in ns, or -1 if unfinished
    outstanding: int       # payload bytes that had not left when observation ended
    bytes: dict            # all observed packets by purpose, including incomplete records


def records(count, **columns):
    """A RECORD table with the given columns set and the rest at their defaults."""
    table = np.zeros(count, dtype=RECORD)
    table["after"] = -1
    for name, values in columns.items():
        table[name] = values
    return table


def chunks(streams, shim, services=None):
    """Turn streams ([ns, direction, bytes] packets, each starting at 0) into Chunks."""
    packets = np.concatenate(streams)
    if packets[:, 0].min() < 0 or packets[:, 0].max() >= shim["arrival_seconds"] * 1e9:
        raise ValueError("chunks must arrive inside the arrival window")
    stream = np.repeat(np.arange(len(streams)), [len(stream) for stream in streams])
    position = np.concatenate([np.arange(len(stream)) for stream in streams])
    order = np.lexsort((position, packets[:, 0]))
    arrival, direction, payload = packets[order].T
    offset = np.zeros(len(order), dtype=np.int64)
    for side in (0, 1):
        offset[direction == side] = np.cumsum(payload[direction == side]) - payload[direction == side]
    service = None if services is None else np.asarray(services)[stream[order]]
    return Chunks(stream[order], position[order], arrival, direction, payload, offset, service)


def identity(arrivals, shim, rng):
    """Reference shim: one record per chunk, first come first served, nothing added."""
    return Plan(records(len(arrivals.arrival), direction=arrivals.direction,
                        ready=arrivals.arrival, stream=arrivals.stream,
                        start=arrivals.offset, payload=arrivals.payload), False)


def wire_bytes(body, shim):
    """Packets and wire bytes of records with the given body sizes."""
    size = body + shim["frame_header_bytes"] + shim["encryption_bytes"]
    packets = -(-size // (shim["mtu_bytes"] - shim["transport_header_bytes"]))
    return packets, size + packets * shim["transport_header_bytes"]


def serve(ready, stream, link_ns, round_robin):
    """Start time of each turn on one link that sends one turn at a time."""
    order = np.argsort(ready, kind="stable")
    ready, link_ns = ready[order], link_ns[order]
    start = np.empty(len(order), dtype=np.int64)
    if not round_robin:
        busy = np.cumsum(link_ns) - link_ns  # link time used by earlier turns
        start[order] = busy + np.maximum.accumulate(ready - busy)
        return start
    # One FIFO queue per stream; backlogged queues take turns.
    times, owners, costs = ready.tolist(), stream[order].tolist(), link_ns.tolist()
    queues, ring, sent = {}, deque(), [0] * len(times)
    clock = i = 0
    while i < len(times) or ring:
        if not ring:
            clock = max(clock, times[i])
        while i < len(times) and times[i] <= clock:
            queue = queues.setdefault(owners[i], deque())
            if not queue:
                ring.append(queue)
            queue.append(i)
            i += 1
        queue = ring.popleft()
        turn = queue.popleft()
        if queue:
            ring.append(queue)
        sent[turn] = clock
        clock += costs[turn]
    start[order] = sent
    return start


def simulate(streams, policy, shim, rng, services=None):
    """Run one tunnel: policy(chunks, shim, rng) -> Plan, then the shared link and accounting."""
    arrivals = chunks(streams, shim, services)
    plan = policy(arrivals, shim, rng)
    table = plan.records
    header, mtu = shim["transport_header_bytes"], shim["mtu_bytes"]
    overhead = shim["frame_header_bytes"] + shim["encryption_bytes"]
    ns_per_byte = 8000 / shim["link_mbps"]
    horizon = int(shim["observation_seconds"] * 1e9)
    packets, wire = wire_bytes(table["payload"] + table["filler"], shim)
    begin = np.zeros(len(table), dtype=np.int64)
    finish = np.full(len(table), -1, dtype=np.int64)
    emitted_packets = np.zeros(len(table), dtype=np.int64)
    trace = []
    for direction in (0, 1):
        ids = np.flatnonzero(table["direction"] == direction)
        ready, after = table["ready"][ids], table["after"][ids]
        waiting = after >= 0
        if (finish[after[waiting]] < 0).any():
            raise ValueError("a record may only wait for a client-to-relay record")
        ready[waiting] += finish[after[waiting]]
        head = ~table["glued"][ids]
        if len(ids) and not head[0]:
            raise ValueError("the first record of a direction cannot be glued")
        turn = np.cumsum(head) - 1
        sent_before = np.cumsum(wire[ids]) - wire[ids]
        turn_bytes = np.bincount(turn, wire[ids], minlength=head.sum())
        start = serve(ready[head], table["stream"][ids][head],
                      np.rint(turn_bytes * ns_per_byte).astype(np.int64), plan.round_robin)
        begin[ids] = start[turn] + np.rint(
            (sent_before - sent_before[head][turn]) * ns_per_byte).astype(np.int64)
        finish[ids] = begin[ids] + np.rint(wire[ids] * ns_per_byte).astype(np.int64)
        # A record's packets are full-size except the last.
        owner = np.repeat(ids, packets[ids])
        index = np.arange(len(owner)) - np.repeat(np.cumsum(packets[ids]) - packets[ids],
                                                  packets[ids])
        size = np.where(index < packets[owner] - 1, mtu, wire[owner] - (packets[owner] - 1) * mtu)
        time = begin[owner] + np.rint(index * mtu * ns_per_byte).astype(np.int64)
        emitted_packets += np.bincount(owner[time <= horizon], minlength=len(table))
        trace.append(np.column_stack([time, np.full(len(owner), direction), size]))
    trace = np.concatenate(trace)
    trace = trace[trace[:, 0] <= horizon]
    trace = trace[np.argsort(trace[:, 0], kind="stable")]

    # Delivery: data records must carry each direction's bytes once, in arrival order.
    sent = finish <= horizon
    delay = np.full(len(arrivals.arrival), -1, dtype=np.int64)
    for direction in (0, 1):
        data = np.flatnonzero((table["direction"] == direction) & (table["kind"] == DATA))
        data = data[np.argsort(table["start"][data], kind="stable")]
        end = table["start"][data] + table["payload"][data]
        mine = arrivals.direction == direction
        total = arrivals.payload[mine].sum()
        if (table["start"][data] != np.r_[0, end[:-1]]).any() or (len(end) and end[-1] > total):
            raise ValueError("data records must carry every payload byte exactly once")
        last = np.searchsorted(end, arrivals.offset[mine] + arrivals.payload[mine])
        carried = last < len(data)
        record = data[np.minimum(last, len(data) - 1)] if len(data) else last
        if (begin[record[carried]] < arrivals.arrival[mine][carried]).any():
            raise ValueError("a record cannot carry bytes that have not arrived")
        done = carried.copy()
        done[carried] = sent[record[carried]]
        delay[np.flatnonzero(mine)[done]] = (finish[record[done]]
                                             - arrivals.arrival[mine][done])
    # The receiver hands each stream's bytes over in order, so a chunk that overtook
    # an earlier one waits for it; one never delivered blocks the rest of its stream.
    never = np.iinfo(np.int64).max
    ready_at = np.where(delay >= 0, arrivals.arrival + delay, never)
    lane = arrivals.stream * 2 + arrivals.direction
    for key in np.unique(lane):
        ready_at[lane == key] = np.maximum.accumulate(ready_at[lane == key])
    delay = np.where(ready_at < never, ready_at - arrivals.arrival, -1)
    # Charge the complete packet at its observed send timestamp, even if the
    # containing record has not completed by the deadline. For the byte ledger
    # only, model encoded order as framing, encryption, real bytes, then filler.
    encoded = np.minimum(emitted_packets * (mtu - header),
                         overhead + table["payload"] + table["filler"])
    framing = np.minimum(encoded, shim["frame_header_bytes"])
    encryption = np.minimum(np.maximum(encoded - shim["frame_header_bytes"], 0),
                            shim["encryption_bytes"])
    payload = np.minimum(np.maximum(encoded - overhead, 0), table["payload"])
    filler = np.maximum(encoded - overhead - table["payload"], 0)
    kind = table["kind"]
    count = dict(payload=payload.sum(), framing=framing.sum(), encryption=encryption.sum(),
                 headers=emitted_packets.sum() * header,
                 padding=filler[kind == DATA].sum(),
                 dummy=filler[kind == DUMMY].sum(), reply=filler[kind == REPLY].sum())
    delivered_payload = table["payload"][sent].sum()
    return Result(trace, delay, int(arrivals.payload.sum() - delivered_payload),
                  {name: int(value) for name, value in count.items()})
