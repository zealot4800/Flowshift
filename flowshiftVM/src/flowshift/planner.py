"""Reversible bounded application-frame edits, not IP packet fragmentation."""
import os
from .protocol import Frame, FrameType, HEADER


class TransformationPlanner:
    @staticmethod
    def transform(data, action):
        if action.kind == 'dummy':
            return [Frame(FrameType.DUMMY, payload=os.urandom(action.sizes[0] - HEADER.size))]
        frames, offset = [], 0
        for index, size in enumerate(action.sizes):
            capacity = size - HEADER.size
            fragment = bytes(data[offset:offset + capacity])
            offset += len(fragment)
            frames.append(Frame(FrameType.DATA, fragment_index=index,
                                fragment_count=len(action.sizes), valid_length=len(fragment),
                                payload=fragment + os.urandom(capacity - len(fragment))))
        if offset != len(data):
            raise ValueError('Action did not preserve all queued data')
        return frames
