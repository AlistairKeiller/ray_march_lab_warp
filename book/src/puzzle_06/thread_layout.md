# Thread Layout: One Thread per Car vs One per Beam

In this puzzle you'll implement the second of four launches that produce the
same scan, then benchmark all four.

## One thread per (car, beam)

This is the Puzzle 5 kernel, `lidar_kernel`, and it's given. The launch is
`dim=(num_envs, NUM_LIDAR)`; each thread computes its car's start point and
heading, rotates one beam and marches one ray. Consecutive threads are
consecutive beams of one car.

**Thread organisation:**

- **Launch**: `dim=(num_envs, 108)`, 884,736 threads at 8,192 cars, with
  `block_dim=128` on CUDA as in warporacer
- **Per thread**: a `cos`/`sin`, a start point, one rotation, one march
- **Memory pattern**: a block of 128 threads holds 128 consecutive beams, one
  car and part of the next, fanning out from one or two points

## One thread per car

The kernel you write. The launch is `dim=num_envs`; each thread computes its
start point once and loops over all 108 beams. 108× fewer threads, each doing
108× the marching.

**Thread organisation:**

- **Launch**: `dim=num_envs`, 8,192 threads at full scale
- **Per thread**: one `cos`/`sin` and start point, then
  `for j in range(108)` with a rotation and a march per iteration
- **Memory pattern**: 32 adjacent threads are 32 different cars, on up to 8
  different maps, reading unrelated parts of a 100 MB raster

## The two launches that are given

- **`lidar_fixed_step_kernel`** is `lidar_kernel` with `raycast_fixed_step`
  in place of `raycast`: advance one pixel, test for wall, repeat. It is the
  march you would write without a distance field, and it is what the field
  is measured against.
- **`lidar_transposed_kernel`** is `lidar_kernel` launched the other way
  round, `dim=(NUM_LIDAR, num_envs)` with `j, i = wp.tid()`, so consecutive
  threads are consecutive cars looking along one beam instead of one car's
  consecutive beams. Same work, different neighbours.

The start point and the beam rotation from Puzzle 5 are factored into two
`@wp.func`s, `laser_px` and `beam_px`, so that the five kernels in the file
share them.

### Code to complete

```python
{{#include ../../../problems/p06/p06.py:lidar_car_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p06/p06.py" class="filename">View full file: problems/p06/p06.py</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

- The start point depends only on the car, so look up `mid`, `r`, `ch`, `sh`
  and `p = laser_px(...)` once, before the loop, exactly as `lidar_kernel`
  does
- Loop `for j in range(lidar_dirs.shape[0]):`, build the direction with
  `beam_px(lidar_dirs[j], ch, sh)` and write `scan[i, j]`
- The result has to match `lidar_kernel` to the bit; the benchmark checks that
  before it times anything

</div>
</details>

## The benchmark harness

The rest of the file is the lidar half of warporacer's `Env`, and it's worth
reading:

- **`harness.track.load_tracks`** picks up to `MAX_ACTIVE_MAPS = 8` yamls at
  random from the directory you pass (or just the one yaml you pass), like
  warporacer's `load_tracks()`, and **`harness.track.spawn`** gives each car a
  random map and a random drivable pose on it
- **`Lidar.__init__`** stacks the maps with `harness.track.stack`, builds the
  per-map `origin`/`res`/`map_h` arrays and allocates the `(num_envs, 108)`
  scan. On CUDA it also wraps the stack in a `wp.Texture3D`
- **`Lidar.record`** calls `wp.launch(..., record_cmd=True)` once per shard and
  keeps the returned `Launch` objects. Replaying one is a single call that skips
  re-slicing and re-packing the arguments. warporacer does this because at 60 Hz
  over millions of steps the Python side of a launch adds up
- **`Lidar.step`** replays the shards: directly on CUDA, through a
  `ThreadPoolExecutor` on CPU, where each shard's kernel runs on its own core
  (the kernel call releases the GIL)
- **`bench`** warms up (the first launch compiles the module), then times
  `steps` replays between two `wp.synchronize_device()` calls. Without the
  sync a CUDA launch returns immediately and you'd time the queueing, not the
  kernel

`wp.set_module_options({"fast_math": True, "enable_backward": False})` at the
top is also from warporacer: SFU trig for the `cos`/`sin`, and no adjoint
kernels compiled.

## Run the code

```bash
uv run python problems/p06/p06.py                                   # 8 random maps, default size for your device
uv run python problems/p06/p06.py maps/ --num-envs 8192 --steps 500
uv run python problems/p06/p06.py maps/Spielberg_map.yaml --device cpu --num-envs 1024
```

Until the kernel is filled in, the correctness check fails and nothing is timed:

```text
Testing Correctness...
   per car                      max difference vs per (car, beam): 2.00e+01 m
   per (beam, car)              max difference vs per (car, beam): 0.00e+00 m
   per (car, beam), 1 px steps  within 2 px of per (car, beam) on 99.1% of rays
   ❌ Layouts disagree
```

When successful, you should see output similar to:

```text
Puzzle 06: Lidar Benchmark
======================================================================

Configuration: 4096 envs x 108 beams, 20 m range, device cpu (18 shards)
maps: Sepang_map, BrandsHatch_map, skirk, MoscowRaceway_map, Melbourne_map, IMS_map, Austin_map, Oschersleben_map
edt stack: (8, 1772, 1759) float32, 100 MB
------------------------------------------------------------
Testing Correctness...
   per car                      max difference vs per (car, beam): 0.00e+00 m
   per (beam, car)              max difference vs per (car, beam): 0.00e+00 m
   per (car, beam), 1 px steps  within 2 px of per (car, beam) on 99.1% of rays
   ✅ All layouts agree

Benchmarking (100 steps, recorded launches)...

Performance Results:
   per (car, beam)                 3.137 ms/step     141.0 M rays/s    1.31 M env-steps/s
   per car                         2.536 ms/step     174.4 M rays/s    1.61 M env-steps/s
   per (beam, car)                 6.859 ms/step      64.5 M rays/s    0.60 M env-steps/s
   per (car, beam), 1 px steps     2.989 ms/step     148.0 M rays/s    1.37 M env-steps/s
   sphere tracing is 0.95x faster than 1 px steps
   per car is 1.24x faster than per (car, beam)
   per (car, beam) is 2.19x faster than per (beam, car)

Key Learning Points:
• The distance field turns a loop over every pixel of range into a loop over a handful of lookups
• Same rays, same results, two thread layouts: 108x more threads or 108x more work per thread
• A GPU needs far more than 8k threads in flight to hide memory latency; a CPU core does not
• Which 32 rays share a warp decides how long it idles for its slowest lane; only a GPU shows this
• The timed loop replays recorded launches and syncs once, so it measures the kernel, not Python
• warporacer's whole step (physics + lidar + PPO bookkeeping) runs ~7M env-steps/s at 8k envs on one GPU
Puzzle 06 complete ✅
```

> These numbers are from an 18-core Apple M5 Pro on Warp's CPU backend, 18
> shards, Warp 1.18. On CUDA you'll see a fifth row for the texture variant, a
> larger default (8,192 envs, 500 steps) and, as the
> [next page](./performance.md) explains, the opposite winner on every ratio.
> The same page has the GPU numbers from the Mojo twin on the same laptop.

## Solution

<details class="solution-details">
<summary></summary>

```python
{{#include ../../../solutions/p06/p06.py:lidar_car_kernel_solution}}
```

<div class="solution-explanation">

The per-car kernel is `lidar_kernel`'s body with the beam loop moved inside the
thread. Everything that depends only on the car (`mid`, `r`, the heading's
`cos` and `sin`, the start point `p`) is computed once; the loop over `j`
rotates each beam and marches it.

Because the two kernels run the same arithmetic on the same values in the same
order, the scans agree to the bit even with `fast_math` on, and the benchmark
requires that before timing anything. A layout change that altered the answer
would not be a layout change.

</div>
</details>
