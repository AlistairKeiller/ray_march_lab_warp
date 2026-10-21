"""Guarded output arrays for the puzzles.

Each output array sits inside a larger allocation. The elements on either side of it hold
NaN, so a kernel that writes past the end of its output leaves a real number behind and
`verify` reports it. A kernel can write every correct value into `output` and still write
outside it, which comparing values cannot detect.

Detection reaches MARGIN elements past each end. A write further out than that misses the
margins altogether and goes unnoticed."""

import math
import sys

import numpy as np
import warp as wp

MARGIN_BYTES = 256  # a whole 256-byte block, so the kernel's view stays 256-byte aligned


class PuzzleMemory:
    """Allocates guarded output arrays on one device and checks their margins."""

    def __init__(self, device=None):
        self.device = wp.get_device(device)
        self._allocations = []

    def output(self, shape, dtype=wp.float32):
        """A zeroed array of `shape`. The margins lie outside the returned view, so a kernel
        reaches them only by indexing out of bounds. Float dtypes only: the margins hold NaN."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        margin = MARGIN_BYTES // wp.types.type_size_in_bytes(dtype)
        count = math.prod(shape)
        allocation = wp.full(count + 2 * margin, float("nan"), dtype=dtype, device=self.device)
        view = allocation[margin : margin + count]
        view.zero_()
        self._allocations.append((allocation, margin, count))
        return view.reshape(shape)

    def verify(self):
        """Fails the puzzle if any output array's margin was written to."""
        if not self.margins_intact():
            print("❌ Write detected outside of output buffer")
            sys.exit(1)

    def margins_intact(self):
        """Whether every output array's margins still hold NaN."""
        for allocation, margin, count in self._allocations:
            buf = allocation.numpy()
            if not (np.isnan(buf[:margin]).all() and np.isnan(buf[margin + count :]).all()):
                return False
        return True
