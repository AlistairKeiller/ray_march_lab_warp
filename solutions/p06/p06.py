import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import warp as wp

from harness.track import load_tracks, spawn, stack

# fast_math: SFU trig for the lidar transcendentals (+22% sim throughput in warporacer;
# domain randomization dwarfs the precision loss). No autodiff anywhere.
wp.set_module_options({"fast_math": True, "enable_backward": False})
wp.config.log_level = wp.LOG_WARNING

NUM_LIDAR = 108
LIDAR_FOV = float(np.radians(270.0))
LIDAR_RANGE = 20.0  # m
LIDAR_MOUNT_X = 0.2733  # m, base_link -> laser offset
MAX_ACTIVE_MAPS = 8  # warporacer trains on a rotating pool of this many maps


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


@wp.func
def raycast_fixed_step(edt: wp.array3d(dtype=float), mid: int, p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """The march you would write without a distance field: advance 1 px at a time and test
    for wall. Same stopping rules as raycast(), one read per pixel of range."""
    dist = float(0.0)
    while dist < max_px:
        col = wp.int32(p[0])
        row = wp.int32(p[1])
        if col < 0 or col >= edt.shape[2] or row < 0 or row >= edt.shape[1]:
            break
        if edt[mid, row, col] == 0.0:
            break
        p += d
        dist += 1.0
    return wp.min(dist, max_px)


@wp.func
def raycast_tex(tex: wp.Texture3D, midf: float, p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """raycast() against the edt texture: border sampling returns 0 (wall) outside the
    volume, so no explicit bounds check is needed."""
    dist = float(0.0)
    while dist < max_px:
        step = wp.texture_sample(tex, wp.vec3(p[0], p[1], midf), dtype=float)
        if step == 0.0:
            break
        p += d * step
        dist += step
    return wp.min(dist, max_px)


@wp.func
def laser_px(x: float, y: float, ch: float, sh: float, org: wp.vec2, r: float, map_h: int) -> wp.vec2:
    """Pixel position of the laser mount for a car at world (x, y) with heading (cos, sin):
    the Puzzle 5 start point, shared by every kernel below."""
    return wp.vec2(
        (x + LIDAR_MOUNT_X * ch - org[0]) / r,
        float(map_h - 1) - (y + LIDAR_MOUNT_X * sh - org[1]) / r,
    )


@wp.func
def beam_px(b: wp.vec2, ch: float, sh: float) -> wp.vec2:
    """A car-frame beam rotated into the heading frame, with the row axis flipped."""
    return wp.vec2(ch * b[0] - sh * b[1], -(sh * b[0] + ch * b[1]))


@wp.kernel
def lidar_kernel(
    cars: wp.array2d(dtype=float),  # x, y, psi
    map_id: wp.array(dtype=int),
    edt: wp.array3d(dtype=float),  # (n_maps, H, W), zero-padded to the largest map
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),  # world origin per map
    res: wp.array(dtype=float),  # meters/px per map
    map_h: wp.array(dtype=int),  # unpadded rows per map
    scan: wp.array2d(dtype=float),
):
    """One thread per (car, beam), beams adjacent: the layout warporacer ships."""
    i, j = wp.tid()
    mid = map_id[i]
    r = res[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = laser_px(cars[i, 0], cars[i, 1], ch, sh, origin[mid], r, map_h[mid])
    scan[i, j] = raycast(edt, mid, p, beam_px(lidar_dirs[j], ch, sh), LIDAR_RANGE / r) * r


@wp.kernel
def lidar_fixed_step_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=int),
    edt: wp.array3d(dtype=float),
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=int),
    scan: wp.array2d(dtype=float),
):
    """lidar_kernel with the 1 px stepper in place of the sphere march: the baseline the
    distance field replaces."""
    i, j = wp.tid()
    mid = map_id[i]
    r = res[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = laser_px(cars[i, 0], cars[i, 1], ch, sh, origin[mid], r, map_h[mid])
    scan[i, j] = raycast_fixed_step(edt, mid, p, beam_px(lidar_dirs[j], ch, sh), LIDAR_RANGE / r) * r


@wp.kernel
def lidar_transposed_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=int),
    edt: wp.array3d(dtype=float),
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=int),
    scan: wp.array2d(dtype=float),
):
    """lidar_kernel launched the other way round, dim=(beam, car): adjacent threads are
    adjacent cars looking along one beam instead of one car's adjacent beams."""
    j, i = wp.tid()
    mid = map_id[i]
    r = res[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = laser_px(cars[i, 0], cars[i, 1], ch, sh, origin[mid], r, map_h[mid])
    scan[i, j] = raycast(edt, mid, p, beam_px(lidar_dirs[j], ch, sh), LIDAR_RANGE / r) * r


@wp.kernel
def lidar_tex_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=int),
    edt_tex: wp.Texture3D,
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=int),
    scan: wp.array2d(dtype=float),
):
    """lidar_kernel sampling the edt as a 3D texture (CUDA only): the texture cache fits the
    scattered sphere-march reads much better than global memory (+14% sim throughput).
    Nearest-sampling at (col, row, map) matches edt[mid, int(row), int(col)] exactly."""
    i, j = wp.tid()
    mid = map_id[i]
    r = res[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = laser_px(cars[i, 0], cars[i, 1], ch, sh, origin[mid], r, map_h[mid])
    scan[i, j] = raycast_tex(edt_tex, float(mid) + 0.5, p, beam_px(lidar_dirs[j], ch, sh), LIDAR_RANGE / r) * r


# ANCHOR: lidar_car_kernel_solution
@wp.kernel
def lidar_car_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=int),
    edt: wp.array3d(dtype=float),
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=int),
    scan: wp.array2d(dtype=float),
):
    """One thread per car that loops over its beams: 108x fewer threads, each doing 108x
    the work. Same rays, same answers; only the layout differs."""
    i = wp.tid()
    mid = map_id[i]
    r = res[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = laser_px(cars[i, 0], cars[i, 1], ch, sh, origin[mid], r, map_h[mid])
    for j in range(lidar_dirs.shape[0]):
        scan[i, j] = raycast(edt, mid, p, beam_px(lidar_dirs[j], ch, sh), LIDAR_RANGE / r) * r


# ANCHOR_END: lidar_car_kernel_solution


class Lidar:
    """The lidar half of warporacer's Env: a padded map stack, a map id per car, cars spawned
    at random drivable poses, and pre-recorded per-shard launches.

    Warp CPU kernels are single-threaded, so on CPU the cars are sharded across a thread
    pool (the kernel call releases the GIL). CUDA runs everything in one shard."""

    def __init__(self, tracks, num_envs: int, seed: int = 0, device=None):
        self.tracks = tracks
        self.num_envs = num_envs
        self.device = d = wp.get_device(device)

        self.edt = wp.array(stack(tracks), device=d)
        if d.is_cuda:
            self.edt_tex = wp.Texture3D(
                self.edt,
                filter_mode=wp.TextureFilterMode.CLOSEST,
                address_mode=wp.TextureAddressMode.BORDER,
                normalized_coords=False,
                device=d,
            )
        self.origin = wp.array([wp.vec2(t.ox, t.oy) for t in tracks], dtype=wp.vec2, device=d)
        self.res = wp.array([t.res for t in tracks], dtype=float, device=d)
        self.map_h = wp.array([t.h for t in tracks], dtype=int, device=d)
        beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
        self.lidar_dirs = wp.array(
            np.column_stack([np.cos(beams), np.sin(beams)]).astype(np.float32), dtype=wp.vec2, device=d
        )

        cars, map_id = spawn(tracks, num_envs, np.random.default_rng(seed))
        self.cars = wp.array(cars, device=d)
        self.map_id = wp.array(map_id, device=d)
        self.scan = wp.zeros((num_envs, NUM_LIDAR), dtype=float, device=d)

        if d.is_cuda:
            self.shards = 1
            self._pool = None
        else:
            self.shards = max(1, min(os.cpu_count() or 1, num_envs))
            self._pool = ThreadPoolExecutor(self.shards)
        self._bounds = np.linspace(0, num_envs, self.shards + 1).astype(int).tolist()

    def record(self, kernel, layout: str, tex: bool = False):
        """One pre-packed launch per shard. Replaying these is much cheaper than re-slicing
        and re-packing the arguments every step. `layout` picks the launch shape: "beam" for
        dim=(cars, beams), "car" for dim=cars, "transposed" for dim=(beams, cars)."""
        kw = {"device": self.device, "record_cmd": True} | ({"block_dim": 128} if self.device.is_cuda else {})
        launches = []
        for s, e in zip(self._bounds[:-1], self._bounds[1:]):
            n = e - s
            dim = {"beam": (n, NUM_LIDAR), "car": n, "transposed": (NUM_LIDAR, n)}[layout]
            launches.append(
                wp.launch(
                    kernel,
                    dim=dim,
                    inputs=[
                        self.cars[s:e], self.map_id[s:e], self.edt_tex if tex else self.edt,
                        self.lidar_dirs, self.origin, self.res, self.map_h, self.scan[s:e],
                    ],
                    **kw,
                )
            )
        return launches

    def step(self, launches):
        if self._pool is None:
            launches[0].launch()
        else:
            list(self._pool.map(lambda cmd: cmd.launch(), launches))


def bench(env, launches, steps):
    """ms per step over `steps` replays, synchronizing only at the ends. The first launch
    also compiles the module, hence the warm-up."""
    for _ in range(3):
        env.step(launches)
    wp.synchronize_device(env.device)
    t = time.perf_counter()
    for _ in range(steps):
        env.step(launches)
    wp.synchronize_device(env.device)
    return (time.perf_counter() - t) / steps * 1e3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lidar throughput at warporacer scale.")
    ap.add_argument("maps", nargs="?", type=Path, default=Path(__file__).parents[2] / "maps")
    ap.add_argument("--num-envs", type=int, default=None, help="cars (default 8192 on CUDA, 4096 on CPU)")
    ap.add_argument("--steps", type=int, default=None, help="timed steps (default 500 on CUDA, 100 on CPU)")
    ap.add_argument("--max-active-maps", type=int, default=MAX_ACTIVE_MAPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
    args = ap.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    num_envs = args.num_envs or (8192 if device.is_cuda else 4096)
    steps = args.steps or (500 if device.is_cuda else 100)
    rng = np.random.default_rng(args.seed)
    tracks = load_tracks(args.maps, args.max_active_maps, rng)
    env = Lidar(tracks, num_envs, seed=args.seed, device=device)

    print("Puzzle 06: Lidar Benchmark")
    print("=" * 70)
    print()
    print(
        f"Configuration: {num_envs} envs x {NUM_LIDAR} beams, {LIDAR_RANGE:g} m range,"
        f" device {device} ({env.shards} shard{'s' if env.shards > 1 else ''})"
    )
    print(f"maps: {', '.join(t.name for t in tracks)}")
    print(f"edt stack: {tuple(env.edt.shape)} float32, {env.edt.size * 4 / 1e6:.0f} MB")
    print("-" * 60)

    BEAM, CAR, TRANSPOSED, FIXED = "per (car, beam)", "per car", "per (beam, car)", "per (car, beam), 1 px steps"
    variants = [
        (BEAM, lidar_kernel, "beam", False),
        (CAR, lidar_car_kernel, "car", False),
        (TRANSPOSED, lidar_transposed_kernel, "transposed", False),
        (FIXED, lidar_fixed_step_kernel, "beam", False),
    ]
    if device.is_cuda:
        variants.append(("per (car, beam), texture", lidar_tex_kernel, "beam", True))

    print("Testing Correctness...")
    scans, launches = {}, {}
    for name, kernel, layout, tex in variants:
        launches[name] = env.record(kernel, layout, tex)
        env.scan.zero_()
        env.step(launches[name])
        wp.synchronize_device(device)
        scans[name] = env.scan.numpy().copy()
    ref = scans[BEAM]
    tol = max(t.res for t in tracks)  # one pixel of the coarsest map
    ok = ref.max() > 0.0
    for name in list(scans)[1:]:
        diff = np.abs(scans[name] - ref)
        if name == FIXED:
            # Grazing rays: a sphere trace and a 1 px stepper can stop on different sides of
            # a wall pixel's corner, so the two only have to agree on nearly every ray.
            within = (diff <= 2 * tol).mean()
            ok &= within >= 0.95
            print(f"   {name:28s} within 2 px of per (car, beam) on {within:.1%} of rays")
        else:
            ok &= diff.max() <= tol
            print(f"   {name:28s} max difference vs per (car, beam): {diff.max():.2e} m")
    if not ok:
        print("   ❌ Layouts disagree")
        sys.exit(1)
    print("   ✅ All layouts agree")

    print()
    print(f"Benchmarking ({steps} steps, recorded launches)...")
    print()
    print("Performance Results:")
    ms = {}
    for name in scans:
        ms[name] = bench(env, launches[name], steps)
        rays = num_envs * NUM_LIDAR / (ms[name] * 1e-3)
        print(f"   {name:28s} {ms[name]:8.3f} ms/step {rays / 1e6:9.1f} M rays/s {num_envs / (ms[name] * 1e-3) / 1e6:7.2f} M env-steps/s")
    print(f"   sphere tracing is {ms[FIXED] / ms[BEAM]:.2f}x faster than 1 px steps")
    fast, slow = (BEAM, CAR) if ms[BEAM] < ms[CAR] else (CAR, BEAM)
    print(f"   {fast} is {ms[slow] / ms[fast]:.2f}x faster than {slow}")
    print(f"   {BEAM} is {ms[TRANSPOSED] / ms[BEAM]:.2f}x faster than {TRANSPOSED}")

    print()
    print("Key Learning Points:")
    print("• The distance field turns a loop over every pixel of range into a loop over a handful of lookups")
    print("• Same rays, same results, two thread layouts: 108x more threads or 108x more work per thread")
    print("• A GPU needs far more than 8k threads in flight to hide memory latency; a CPU core does not")
    print("• Which 32 rays share a warp decides how long it idles for its slowest lane; only a GPU shows this")
    print("• The timed loop replays recorded launches and syncs once, so it measures the kernel, not Python")
    print("• warporacer's whole step (physics + lidar + PPO bookkeeping) runs ~7M env-steps/s at 8k envs on one GPU")
    print("Puzzle 06 complete ✅")
