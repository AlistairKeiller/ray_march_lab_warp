from pathlib import Path

import numpy as np
import warp as wp

from harness.canary import PuzzleMemory
from harness.track import Track, stack

wp.config.log_level = wp.LOG_WARNING

# ANCHOR: lidar_multi
MAPS = [Path(__file__).parents[2] / "maps" / f for f in ("berlin.yaml", "skirk.yaml", "Spielberg_map.yaml")]
CARS_PER_MAP = 2
NUM_LIDAR = 108
LIDAR_FOV = float(np.radians(270.0))
LIDAR_RANGE = 20.0  # m
LIDAR_MOUNT_X = 0.2733  # m, base_link -> laser offset


@wp.func
def raycast(edt: wp.array3d(dtype=float), mid: int, p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """Sphere-march from pixel p along unit direction d (col, row); returns distance in px.
    Padding rows/cols are zero (= wall), so rays stop at each map's true bounds."""
    dist = float(0.0)
    while dist < max_px:
        col = wp.int32(p[0])
        row = wp.int32(p[1])
        if col < 0 or col >= edt.shape[2] or row < 0 or row >= edt.shape[1]:
            break
        step = edt[mid, row, col]
        if step == 0.0:
            break
        p += d * step
        dist += step
    return wp.min(dist, max_px)


@wp.kernel
def lidar_multi(
    cars: wp.array2d(dtype=float),  # x, y, psi
    map_id: wp.array(dtype=int),  # which map each car drives on
    edt: wp.array3d(dtype=float),  # (n_maps, H, W), zero-padded to the largest map
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),  # world origin per map
    res: wp.array(dtype=float),  # meters/px per map
    map_h: wp.array(dtype=int),  # unpadded rows per map
    scan: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    # FILL ME IN (roughly 12 lines)


# ANCHOR_END: lidar_multi


def reference(track, cars, dirs):
    """One map's scans in numpy, in float32 like the kernel."""
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
    tracks = [Track(p) for p in MAPS]
    num_cars = CARS_PER_MAP * len(tracks)
    rng = np.random.default_rng(0)
    cars_np = np.concatenate([t.sample_poses(CARS_PER_MAP, rng) for t in tracks])
    map_id_np = np.repeat(np.arange(len(tracks), dtype=np.int32), CARS_PER_MAP)
    beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
    dirs_np = np.column_stack([np.cos(beams), np.sin(beams)]).astype(np.float32)

    mem = PuzzleMemory()
    scan = mem.output((num_cars, NUM_LIDAR))
    cars = wp.array(cars_np)
    map_id = wp.array(map_id_np)
    edt = wp.array(stack(tracks))
    lidar_dirs = wp.array(dirs_np, dtype=wp.vec2)
    origin = wp.array([wp.vec2(t.ox, t.oy) for t in tracks], dtype=wp.vec2)
    res = wp.array([t.res for t in tracks], dtype=float)
    map_h = wp.array([t.h for t in tracks], dtype=int)

    wp.launch(
        lidar_multi,
        dim=(num_cars, NUM_LIDAR),
        inputs=[cars, map_id, edt, lidar_dirs, origin, res, map_h],
        outputs=[scan],
    )
    wp.synchronize_device()

    expected = np.concatenate(
        [reference(t, cars_np[map_id_np == k], dirs_np) for k, t in enumerate(tracks)]
    )
    print(f"edt stack: {edt.shape} from", ", ".join(f"{t.name} ({t.h} x {t.w})" for t in tracks))
    for k, t in enumerate(tracks):
        i = k * CARS_PER_MAP
        print(f"{t.name} car {i}")
        print("  out (every 18th beam):", scan.numpy()[i, ::18])
        print("  expected:             ", expected[i, ::18])
    print(f"max |out - expected|: {np.abs(scan.numpy() - expected).max():.2e} m")
    np.testing.assert_allclose(scan.numpy(), expected, atol=max(t.res for t in tracks))
    mem.verify()
    print("Puzzle 05 complete ✅")
