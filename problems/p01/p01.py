import numpy as np
import warp as wp

from harness.canary import PuzzleMemory

wp.config.log_level = wp.LOG_WARNING  # hide the init banner and module-load timings

# ANCHOR: px_to_m
SIZE = 4
RES = 0.05  # meters per pixel, the F1TENTH map default


@wp.kernel
def px_to_m(edt: wp.array(dtype=float), res: float, output: wp.array(dtype=float)):
    i = wp.tid()
    # FILL ME IN (roughly 1 line)
    output[i] = edt[i] * res


# ANCHOR_END: px_to_m


if __name__ == "__main__":
    wp.init()
    mem = PuzzleMemory()
    out = mem.output(SIZE)
    edt = wp.array(np.arange(SIZE, dtype=np.float32))

    wp.launch(px_to_m, dim=SIZE, inputs=[edt, RES], outputs=[out])
    wp.synchronize_device()

    expected = edt.numpy() * np.float32(RES)
    print("out:", out.numpy())
    print("expected:", expected)
    np.testing.assert_array_equal(out.numpy(), expected)
    mem.verify()
    print("Puzzle 01 complete ✅")
