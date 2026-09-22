"""Private parent/child pipes carrying CPU tensors across Python versions."""
import io
import os
import select
import struct
import time

import torch


class TensorChannel:
    def __init__(self, read_fd, write_fd, timeout=600):
        self.read_fd, self.write_fd, self.timeout = read_fd, write_fd, timeout

    def _transfer(self, data, *, writing):
        remaining = len(data)
        offset = 0
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while remaining:
            timeout = None if deadline is None else max(0, deadline - time.monotonic())
            readable, writable, _ = select.select(
                [] if writing else [self.read_fd], [self.write_fd] if writing else [], [], timeout)
            if not readable and not writable:
                raise TimeoutError('V-JEPA worker pipe timed out')
            if writing:
                # A bounded write avoids blocking after select reports a writable pipe.
                size = os.write(self.write_fd, data[offset:offset + min(remaining, 4096)])
            else:
                chunk = os.read(self.read_fd, min(remaining, 1024 * 1024))
                size = len(chunk)
                if not size:
                    raise EOFError('V-JEPA worker pipe closed')
                data[offset:offset + size] = chunk
            offset += size
            remaining -= size

    def send(self, value):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        payload = buffer.getbuffer()
        self._transfer(memoryview(struct.pack('!Q', len(payload))), writing=True)
        self._transfer(payload, writing=True)

    def receive(self):
        header = bytearray(8)
        self._transfer(header, writing=False)
        size, = struct.unpack('!Q', header)
        if size > 4 * 1024**3:
            raise ValueError(f'Oversized V-JEPA message: {size}')
        payload = bytearray(size)
        self._transfer(payload, writing=False)
        return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)

    def close(self):
        for name in ('read_fd', 'write_fd'):
            descriptor = getattr(self, name)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)
