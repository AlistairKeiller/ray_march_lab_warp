from pathlib import Path

import numpy as np
import warp as wp

from harness.canary import PuzzleMemory
from harness.track import Track

wp.config.log_level = wp.LOG_WARNING

MAP = Path(__file__).parents[2] / "maps" / "Spielberg_map.yaml"
NUM_CARS = 4
NUM_LIDAR = 108
LIDAR_FOV = float(np.radians(270.0))
LIDAR_RANGE = 20.0  # m
LIDAR_MOUNT_X = 0.2733  # m, base_link -> laser offset


@wp.func
def raycast(edt: wp.array2d(dtype=float), p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """Sphere-march from pixel p along unit direction d (col, row); returns distance in px."""
    dist = float(0.0)
    while dist < max_px:
        col = wp.int32(p[0])
        row = wp.int32(p[1])
        if col < 0 or col >= edt.shape[1] or row < 0 or row >= edt.shape[0]:
            break
        step = edt[row, col]
        if step == 0.0:
            break
        p += d * step
        dist += step
    return wp.min(dist, max_px)


# ANCHOR: lidar_solution
@wp.kernel
def lidar(
    cars: wp.array2d(dtype=float),  # x, y, psi
    edt: wp.array2d(dtype=float),
    lidar_dirs: wp.array(dtype=wp.vec2),  # unit beam directions in the car frame
    res: float,
    origin: wp.vec2,
    scan: wp.array2d(dtype=float),  # (car, beam) range in m
):
    """One thread per (car, beam): rays from the mount point, rotated into the heading
    frame (the row axis flips y)."""
    i, j = wp.tid()
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = wp.vec2(
        (cars[i, 0] + LIDAR_MOUNT_X * ch - origin[0]) / res,
        float(edt.shape[0] - 1) - (cars[i, 1] + LIDAR_MOUNT_X * sh - origin[1]) / res,
    )
    d = wp.vec2(
        ch * lidar_dirs[j][0] - sh * lidar_dirs[j][1],
        -(sh * lidar_dirs[j][0] + ch * lidar_dirs[j][1]),
    )
    scan[i, j] = raycast(edt, p, d, LIDAR_RANGE / res) * res


# ANCHOR_END: lidar_solution


def reference(track, cars, dirs):
    """The same scan in numpy, in float32 like the kernel."""
    f = np.float32
    res, ox, oy = f(track.res), f(track.ox), f(track.oy)
    max_px = f(LIDAR_RANGE) / res
    scan = np.zeros((len(cars), len(dirs)), np.float32)
    for i, (x, y, psi) in enumerate(cars):
        ch, sh = np.cos(psi), np.sin(psi)
        px = (x + f(LIDAR_MOUNT_X) * ch - ox) / res
        py = f(track.h - 1) - (y + f(LIDAR_MOUNT_X) * sh - oy) / res
        for j, (bx, by) in enumerate(dirs):
            dx, dy = ch * bx - sh * by, -(sh * bx + ch * by)
            qx, qy, dist = px, py, f(0.0)
            while dist < max_px:
                col, row = int(qx), int(qy)
                if col < 0 or col >= track.w or row < 0 or row >= track.h:
                    break
                step = track.edt[row, col]
                if step == 0.0:
                    break
                qx, qy, dist = f(qx + dx * step), f(qy + dy * step), f(dist + step)
            scan[i, j] = min(dist, max_px) * res
    return scan


if __name__ == "__main__":
    wp.init()
    track = Track(MAP)
    rng = np.random.default_rng(0)
    cars_np = track.sample_poses(NUM_CARS, rng)
    beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
    dirs_np = np.column_stack([np.cos(beams), np.sin(beams)]).astype(np.float32)

    mem = PuzzleMemory()
    scan = mem.output((NUM_CARS, NUM_LIDAR))
    cars = wp.array(cars_np)
    edt = wp.array(track.edt)
    lidar_dirs = wp.array(dirs_np, dtype=wp.vec2)

    wp.launch(
        lidar,
        dim=(NUM_CARS, NUM_LIDAR),
        inputs=[cars, edt, lidar_dirs, track.res, wp.vec2(track.ox, track.oy)],
        outputs=[scan],
    )
    wp.synchronize_device()

    expected = reference(track, cars_np, dirs_np)
    print(f"map: {track.name} ({track.h} x {track.w} px, {track.res} m/px)")
    print("car 0 pose (x, y, psi):", cars_np[0])
    print("out (car 0, every 18th beam):", scan.numpy()[0, ::18])
    print("expected:", expected[0, ::18])
    print(f"max |out - expected|: {np.abs(scan.numpy() - expected).max():.2e} m")
    np.testing.assert_allclose(scan.numpy(), expected, atol=track.res)  # within one pixel
    mem.verify()
    print("Puzzle 04 complete ✅")
