"""Binary framing used between the FlowShift sender and receiver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import socket
import struct


MAGIC = b"FSH1"
VERSION = 3
HEADER = struct.Struct("!4sBBIHHBB")
MAX_PAYLOAD = 65535


class ProtocolError(RuntimeError):
    """Raised when a peer sends an invalid FlowShift frame."""


class FrameType(IntEnum):
    HELLO = 1
    DATA = 2
    DUMMY = 3
    END = 4
    RESULT = 5


@dataclass(frozen=True, slots=True)
class Frame:
    frame_type: FrameType
    sequence: int = 0
    fragment_index: int = 0
    fragment_count: int = 0
    valid_length: int = 0
    payload: bytes = b""

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD:
            raise ProtocolError("frame payload exceeds safety limit")
        if not 0 <= self.valid_length <= len(self.payload):
            raise ProtocolError("valid_length is outside the payload")
        if not 0 <= self.sequence <= 0xFFFFFFFF or not 0 <= self.fragment_index <= 255 or not 0 <= self.fragment_count <= 255:
            raise ProtocolError("sequence or fragment index exceeds protocol field")
        header = HEADER.pack(
            MAGIC,
            VERSION,
            int(self.frame_type),
            self.sequence,
            self.valid_length,
            len(self.payload),
            self.fragment_index,
            self.fragment_count,
        )
        return header + self.payload


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket, max_payload: int = MAX_PAYLOAD) -> Frame:
    raw = recv_exact(sock, HEADER.size)
    (
        magic,
        version,
        frame_type,
        sequence,
        valid_length,
        payload_length,
        fragment_index,
        fragment_count,
    ) = HEADER.unpack(raw)
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("invalid FlowShift protocol marker or version")
    if payload_length > min(MAX_PAYLOAD, max_payload):
        raise ProtocolError("declared payload exceeds safety limit")
    if valid_length > payload_length:
        raise ProtocolError("declared valid data exceeds payload")
    try:
        kind = FrameType(frame_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown frame type {frame_type}") from exc
    payload = recv_exact(sock, payload_length)
    return Frame(
        frame_type=kind,
        sequence=sequence,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
        valid_length=valid_length,
        payload=payload,
    )


def send_frame(sock: socket.socket, frame: Frame) -> int:
    encoded = frame.encode()
    sock.sendall(encoded)
    return len(encoded)
