"""FlowShift receiver-side configuration and restoration manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from .policy import FlowConfig
from .protocol import Frame, FrameType, HEADER, ProtocolError, recv_frame, send_frame


@dataclass(slots=True)
class ReceiveStats:
    output_path: str
    recovered_bytes: int
    data_frames: int
    dummy_frames: int
    duration_seconds: float
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_filename(value: object) -> str:
    name = Path(str(value)).name
    if name in {"", ".", ".."}:
        raise ProtocolError("invalid output filename")
    return name


def _result(ok: bool, *, error: str | None = None) -> Frame:
    value: dict[str, object] = {"ok": ok}
    if error:
        value["error"] = error
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return Frame(
        frame_type=FrameType.RESULT,
        valid_length=len(payload),
        payload=payload,
    )


def receive_connection(
    sock: socket.socket,
    output_dir: str | Path,
    *,
    play: bool = False,
) -> ReceiveStats:
    hello = recv_frame(sock)
    if hello.frame_type is not FrameType.HELLO:
        raise ProtocolError("first frame must be HELLO")
    metadata = json.loads(hello.payload[: hello.valid_length])
    config = FlowConfig.from_dict(metadata["config"])
    filename = _safe_filename(metadata["filename"])
    expected_size = int(metadata["original_size"])
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = directory / f".{filename}.flowshift.part"

    expected_sequence = 1
    fragment_index = 0
    active_fragment_count = 0
    digest = hashlib.sha256()
    recovered_bytes = 0
    data_frames = 0
    dummy_frames = 0
    started = time.monotonic()
    player: subprocess.Popen[bytes] | None = None
    player_pipe_open = False

    if play:
        if metadata.get("mode") != "video":
            raise ProtocolError("live playback requires a video stream")
        player = subprocess.Popen(
            [
                "ffplay",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-autoexit",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
        player_pipe_open = player.stdin is not None

    try:
        with temporary.open("wb") as output:
            while True:
                frame = recv_frame(sock, max(1024, config.frame_bytes - HEADER.size))
                if frame.frame_type is FrameType.END:
                    if frame.sequence != expected_sequence:
                        raise ProtocolError("END frame sequence mismatch")
                    end = json.loads(frame.payload[: frame.valid_length])
                    if fragment_index:
                        raise ProtocolError("transfer ended during fragmented object")
                    if (
                        (expected_size >= 0 and recovered_bytes != expected_size)
                        or recovered_bytes != int(end["original_size"])
                    ):
                        raise ProtocolError("recovered file size mismatch")
                    if digest.hexdigest() != str(end["sha256"]):
                        raise ProtocolError("recovered file hash mismatch")
                    break
                if frame.frame_type not in {FrameType.DATA, FrameType.DUMMY}:
                    raise ProtocolError("unexpected frame in transfer")
                if frame.sequence != expected_sequence:
                    raise ProtocolError("non-contiguous frame sequence")
                expected_sequence += 1
                if len(frame.payload) + HEADER.size > config.frame_bytes:
                    raise ProtocolError("data frame exceeds negotiated framing limit")
                if frame.frame_type is FrameType.DUMMY:
                    if frame.valid_length != 0:
                        raise ProtocolError("dummy frame contains valid application data")
                    dummy_frames += 1
                    continue

                data_frames += 1
                if not 1 <= frame.fragment_count <= 2 or frame.valid_length < 1:
                    raise ProtocolError("invalid bounded data fragmentation")
                if not fragment_index:
                    active_fragment_count = frame.fragment_count
                if frame.fragment_count != active_fragment_count:
                    raise ProtocolError("fragment count changed within an object")
                if frame.fragment_index != fragment_index:
                    raise ProtocolError("fragment arrived out of order")
                restored = frame.payload[:frame.valid_length]
                output.write(restored)
                if player is not None and player_pipe_open and player.stdin:
                    try:
                        player.stdin.write(restored)
                        player.stdin.flush()
                    except BrokenPipeError:
                        player_pipe_open = False
                digest.update(restored)
                recovered_bytes += len(restored)
                fragment_index += 1
                if fragment_index == active_fragment_count:
                    fragment_index = 0
                    active_fragment_count = 0
            output.flush()
        if player is not None and player.stdin and player_pipe_open:
            player.stdin.close()
            player_pipe_open = False
        if player is not None:
            try:
                player.wait(timeout=10)
            except subprocess.TimeoutExpired:
                player.terminate()
                player.wait()
        temporary.replace(destination)
        stats = ReceiveStats(
            output_path=str(destination),
            recovered_bytes=recovered_bytes,
            data_frames=data_frames,
            dummy_frames=dummy_frames,
            duration_seconds=time.monotonic() - started,
            sha256=digest.hexdigest(),
        )
        send_frame(sock, _result(True))
        return stats
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if player is not None and player.poll() is None:
            player.terminate()
        try:
            send_frame(sock, _result(False, error=str(exc)))
        except OSError:
            pass
        raise


def receive_once(
    bind_address: str,
    port: int,
    output_dir: str | Path,
    *,
    play: bool = False,
) -> ReceiveStats:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_address, port))
        listener.listen(1)
        connection, _peer = listener.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return receive_connection(connection, output_dir, play=play)
