"""Bounded causal TCP tunnel sender for files and live video."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .planner import TransformationPlanner
from .policy import Action, FlowConfig, FlowShift
from .protocol import Frame, FrameType, HEADER, ProtocolError, recv_frame, send_frame


def _json_frame(kind, value, sequence=0):
    payload = json.dumps(value, separators=(',', ':')).encode()
    return Frame(kind, sequence=sequence, valid_length=len(payload), payload=payload)


def read_unit(read, config, clock=time.monotonic, wait=time.sleep):
    """Coalesce observed bytes into one bounded FIFO batch, within its deadline."""
    limit = config.queue_bytes
    data = read(limit)
    arrival = clock()
    if data is None or not data:
        return data, arrival
    data = bytearray(data)
    deadline = arrival + config.coalesce_ms / 1000
    while len(data) < limit and deadline - clock() > 1e-6:
        extra = read(limit - len(data))
        if extra == b'':
            break
        if extra is None:
            wait(min(.0001, max(0., deadline - clock())))
        else:
            data.extend(extra)
    if len(data) > limit:
        raise ValueError('Reader exceeded the bounded admission size')
    return bytes(data), arrival


def _send_transformed(sock, filename, expected_size, config, read, *, controller=None, video=False, finish=lambda: None):
    """Common pipeline for every obfuscator; only `controller` (the policy) differs."""
    controller = controller or FlowShift(config)
    hello = dict(filename=filename, original_size=expected_size, config=config.to_dict(), policy=controller.name)
    if video:
        hello['mode'] = 'video'
    framed_bytes = send_frame(sock, _json_frame(FrameType.HELLO, hello))
    pending, object_id, sequence = b'', 0, 1
    digest = hashlib.sha256()
    data_frames = dummy_frames = original_bytes = data_framed_bytes = max_queue = 0
    max_delay = violations = policy_cpu = 0.
    max_frame = 0
    started, cpu_started = time.monotonic(), time.process_time()
    arrival = started
    while True:
        if not pending:
            chunk, arrival = read_unit(read, config)
            if chunk == b'':
                break
            if chunk is None:
                time.sleep(.001)
                continue
            pending = memoryview(chunk)
            backlogged = len(pending) > config.read_chunk_bytes or len(pending) == config.queue_bytes
            digest.update(chunk)
            original_bytes += len(chunk)
            max_queue = max(max_queue, len(pending))
        before_policy = time.process_time()
        payload_bytes = min(len(pending), controller.capacity())
        action = controller.select(payload_bytes, arrival, time.monotonic(),
                                   queued_bytes=len(pending), backlogged=backlogged)
        policy_cpu += time.process_time() - before_policy
        if not action.sizes:
            time.sleep(.001)
            continue
        if action.delay_s:
            wait = min(action.delay_s, max(0., arrival + (config.max_delay_ms - config.deadline_guard_ms) / 1000 - time.monotonic()))
            if wait:
                time.sleep(wait)
        if action.kind != 'dummy' and time.monotonic() - arrival >= (config.max_delay_ms - config.deadline_guard_ms) / 1000:
            action = Action('unchanged', (payload_bytes + HEADER.size,))
        controller.reserve(action)
        for frame in TransformationPlanner.transform(pending[:payload_bytes], action):
            frame = replace(frame, sequence=sequence)
            encoded = frame.encode()
            sent_at = time.monotonic()
            if frame.frame_type is FrameType.DATA:
                delay = sent_at - arrival
                max_delay = max(max_delay, delay)
                violations += delay > config.max_delay_ms / 1000
                data_frames += 1
            else:
                dummy_frames += 1
            sock.sendall(encoded)
            controller.record(len(encoded), frame.valid_length, sent_at)
            framed_bytes += len(encoded)
            data_framed_bytes += len(encoded)
            max_frame = max(max_frame, len(encoded))
            sequence += 1
        if action.kind != 'dummy':
            pending = pending[payload_bytes:]
            object_id += 1
    finish()
    checksum = digest.hexdigest()
    end = dict(original_size=original_bytes, sha256=checksum, data_frames=data_frames, dummy_frames=dummy_frames)
    framed_bytes += send_frame(sock, _json_frame(FrameType.END, end, sequence))
    result = recv_frame(sock)
    if result.frame_type is not FrameType.RESULT:
        raise ProtocolError('Receiver did not return a result')
    response = json.loads(result.payload[:result.valid_length])
    if not response.get('ok'):
        raise ProtocolError(f'Receiver rejected transfer: {response.get("error")}')
    baseline = original_bytes + object_id * HEADER.size
    assert controller.payload_bytes == original_bytes
    return dict(policy=controller.name, policy_settings=controller.settings, original_bytes=original_bytes, framed_bytes=framed_bytes,
                data_framed_bytes=data_framed_bytes, baseline_data_framed_bytes=baseline,
                data_framing_overhead_percent=100 * (data_framed_bytes - baseline) / baseline if baseline else 0.,
                charged_extra_bytes=controller.extra_bytes, credit_remaining=controller.credit,
                credit_initial=config.initial_credit, byte_fraction=config.byte_fraction,
                cost_basis='data tunnel framing plus configured transport-header charge per extra frame; not captured IP wire bytes',
                data_frames=data_frames, dummy_frames=dummy_frames, decisions=sum(controller.actions.values()),
                actions=controller.actions, max_queue_bytes=max_queue, max_frame_bytes=max_frame,
                max_controller_delay_s=max_delay, controller_deadline_violations=int(violations),
                delay_basis='admission to real-frame socket-write start, including earlier action writes; excludes upstream backpressure and subsequent delivery',
                duration_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu_started,
                selection_cpu_seconds=policy_cpu, sha256=checksum)

def _connect(host: str, port: int, bind_address: str | None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bind_address:
        sock.bind((bind_address, 0))
    sock.connect((host, port))
    return sock


def send_file(
    host: str,
    port: int,
    path: str | Path,
    config: FlowConfig,
    *,
    bind_address: str | None = None,
    controller=None,
) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with _connect(host, port, bind_address) as sock, source.open("rb") as handle:
        return _send_transformed(
            sock,
            source.name,
            source.stat().st_size,
            config,
            handle.read,
            controller=controller,
        )


def send_video(
    host: str,
    port: int,
    path: str | Path,
    config: FlowConfig,
    duration_seconds: int,
    width: int,
    height: int,
    video_bitrate_kbps: int,
    audio_bitrate_kbps: int,
    *,
    bind_address: str | None = None,
    controller=None,
) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if (
        duration_seconds < 1
        or width < 16
        or height < 16
        or video_bitrate_kbps < 1
        or audio_bitrate_kbps < 1
    ):
        raise ValueError("invalid video settings")
    sock = _connect(host, port, bind_address)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-stream_loop", "-1", "-re", "-i", str(source),
                "-t", str(duration_seconds), "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-threads", "4",
                "-tune", "zerolatency",
                "-vf", (
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-b:v", f"{video_bitrate_kbps}k",
                "-maxrate", f"{video_bitrate_kbps}k",
                "-bufsize", f"{video_bitrate_kbps * 2}k",
                "-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k",
                "-f", "mpegts", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None:
            raise RuntimeError("FFmpeg output pipe was not created")
        os.set_blocking(process.stdout.fileno(), False)

        def read(limit) -> bytes | None:
            try:
                return os.read(process.stdout.fileno(), limit)
            except BlockingIOError:
                return None

        def finish() -> None:
            status = process.wait()
            if status:
                raise RuntimeError(f"FFmpeg exited with status {status}")

        return _send_transformed(
            sock,
            f"{source.stem}.ts",
            -1,
            config,
            read,
            controller=controller,
            video=True,
            finish=finish,
        )
    finally:
        sock.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
