import numpy as np
import warp as wp
from scipy.ndimage import distance_transform_edt

from harness.canary import PuzzleMemory

wp.config.log_level = wp.LOG_WARNING

# ANCHOR: raycast
SIZE = 16  # a SIZE x SIZE px room with a 1 px wall all round and a 2 x 2 px pillar
NUM_RAYS = 8
MAX_PX = 8.0  # give up, and report MAX_PX, after marching this far


@wp.func
def raycast(edt: wp.array2d(dtype=float), p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """Sphere-march from pixel p along unit direction d (col, row); returns distance in px."""
    dist = float(0.0)
    # FILL ME IN (roughly 9 lines)
    return wp.min(dist, max_px)


@wp.kernel
def march(
    edt: wp.array2d(dtype=float),
    starts: wp.array(dtype=wp.vec2),  # (col, row) per ray
    dirs: wp.array(dtype=wp.vec2),  # unit (col, row) per ray
    max_px: float,
    output: wp.array(dtype=float),
):
    i = wp.tid()
    output[i] = raycast(edt, starts[i], dirs[i], max_px)


# ANCHOR_END: raycast


def room(size):
    free = np.zeros((size, size), bool)
    free[1:-1, 1:-1] = True
    free[4:6, 10:12] = False  # the pillar
    return distance_transform_edt(free).astype(np.float32)


def reference(edt, starts, dirs, max_px):
    """The same march in numpy, in float32 like the kernel."""
    f = np.float32
    out = np.zeros(len(starts), np.float32)
    for k, ((x, y), (dx, dy)) in enumerate(zip(starts, dirs)):
        dist = f(0.0)
        while dist < max_px:
            col, row = int(x), int(y)
            if col < 0 or col >= edt.shape[1] or row < 0 or row >= edt.shape[0]:
                break
            step = edt[row, col]
            if step == 0.0:
                break
            x, y, dist = f(x + dx * step), f(y + dy * step), f(dist + step)
        out[k] = min(dist, max_px)
    return out


if __name__ == "__main__":
    wp.init()
    mem = PuzzleMemory()
    out = mem.output(NUM_RAYS)
    edt_np = room(SIZE)
    angles = np.arange(NUM_RAYS) * (2 * np.pi / NUM_RAYS)  # every 45 degrees
    starts_np = np.full((NUM_RAYS, 2), SIZE / 2 + 0.5, np.float32)  # centre of pixel (8, 8)
    dirs_np = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)

    edt = wp.array(edt_np)
    starts = wp.array(starts_np, dtype=wp.vec2)
    dirs = wp.array(dirs_np, dtype=wp.vec2)

    wp.launch(march, dim=NUM_RAYS, inputs=[edt, starts, dirs, MAX_PX], outputs=[out])
    wp.synchronize_device()

    expected = reference(edt_np, starts_np, dirs_np, np.float32(MAX_PX))
    print("out:", out.numpy())
    print("expected:", expected)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4)
    mem.verify()
    print("Puzzle 03 complete ✅")
