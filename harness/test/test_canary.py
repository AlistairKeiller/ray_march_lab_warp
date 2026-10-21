"""Tests that the canary stays quiet on a correct kernel and fires on a write past either
end of the output."""

import warp as wp

from harness.canary import PuzzleMemory

wp.config.log_level = wp.LOG_WARNING

SIZE = 8


@wp.kernel
def write_in_bounds(output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = 1.0


@wp.kernel
def write_one_past_the_end(output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = 1.0
    if i == 0:
        output[SIZE] = 1.0


@wp.kernel
def write_one_before_the_start(output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = 1.0
    if i == 0:
        # Warp wraps one negative index Python-style, so -SIZE - 1 lands one before the start.
        output[-SIZE - 1] = 1.0


def run(kernel):
    mem = PuzzleMemory()
    out = mem.output(SIZE)
    wp.launch(kernel, dim=SIZE, inputs=[out])
    wp.synchronize_device()
    return mem.margins_intact()


if __name__ == "__main__":
    wp.init()
    assert run(write_in_bounds), "a correct kernel tripped the canary"
    assert not run(write_one_past_the_end), "a write past the end went unnoticed"
    assert not run(write_one_before_the_start), "a write before the start went unnoticed"
    print("Canary harness tests passed ✅")
