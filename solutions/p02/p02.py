import numpy as np
import warp as wp
from scipy.ndimage import distance_transform_edt

from harness.canary import PuzzleMemory

wp.config.log_level = wp.LOG_WARNING

NUM_CARS = 4
SIZE = 8  # the map is a SIZE x SIZE px room with a 1 px wall all round
RES = 0.5  # m/px, coarse so the whole room fits in a print
ORIGIN = (-2.0, -2.0)  # world x, y of the bottom-left pixel
HALF_DIAG = float(np.hypot(0.42, 0.58) / 2.0)  # m, half the car's diagonal


# ANCHOR: clearance_solution
@wp.kernel
def clearance(
    cars: wp.array2d(dtype=float),  # x, y, psi
    edt: wp.array2d(dtype=float),  # (row, col) px distance to nearest wall
    res: float,
    origin: wp.vec2,
    output: wp.array(dtype=float),
):
    i = wp.tid()
    col = wp.clamp(wp.int32((cars[i, 0] - origin[0]) / res), 0, edt.shape[1] - 1)
    row = wp.clamp(wp.int32(float(edt.shape[0] - 1) - (cars[i, 1] - origin[1]) / res), 0, edt.shape[0] - 1)
    output[i] = edt[row, col] * res - HALF_DIAG


# ANCHOR_END: clearance_solution


def room(size):
    """Distance-to-wall raster of a square room: free inside, wall on the border."""
    free = np.zeros((size, size), bool)
    free[1:-1, 1:-1] = True
    return distance_transform_edt(free).astype(np.float32)


if __name__ == "__main__":
    wp.init()
    mem = PuzzleMemory()
    out = mem.output(NUM_CARS)
    edt = wp.array(room(SIZE))
    cars = wp.array(
        np.array(
            [
                [0.0, 0.0, 0.0],  # middle of the room
                [1.2, -1.4, 0.0],  # one pixel from the bottom-right corner
                [-0.9, 0.3, 0.0],  # two pixels from the left wall
                [5.0, 9.0, 0.0],  # off the map entirely
            ],
            np.float32,
        )
    )

    wp.launch(clearance, dim=NUM_CARS, inputs=[cars, edt, RES, wp.vec2(ORIGIN)], outputs=[out])
    wp.synchronize_device()

    # The same transform in numpy, in float32 like the kernel.
    c = cars.numpy()
    col = np.clip(((c[:, 0] - np.float32(ORIGIN[0])) / np.float32(RES)).astype(np.int32), 0, SIZE - 1)
    row = np.clip((np.float32(SIZE - 1) - (c[:, 1] - np.float32(ORIGIN[1])) / np.float32(RES)).astype(np.int32), 0, SIZE - 1)
    expected = edt.numpy()[row, col] * np.float32(RES) - np.float32(HALF_DIAG)

    print("room (px to wall):")
    print(edt.numpy())
    print("out:", out.numpy())
    print("expected:", expected)
    np.testing.assert_array_equal(out.numpy(), expected)
    mem.verify()
    print("Puzzle 02 complete ✅")
