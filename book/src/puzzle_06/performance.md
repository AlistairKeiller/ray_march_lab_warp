# Performance Comparison

Same rays, same results, two thread layouts. On the laptop CPU in the previous
page the per-car layout came out 1.3× ahead. On a GPU, expect the per-(car,
beam) layout to win, and by a lot more than 1.3×. This page is about why, and
about what else warporacer does to make the lidar cheap.

## Why the CPU prefers per-car

A Warp CPU kernel is a compiled C++ loop over `tid`, running on one core per
launch. "Threads" cost nothing to create, but each one redoes its setup: in the
per-(car, beam) layout every one of a car's 108 threads recomputes `cos`,
`sin` and the start point, and re-reads the car's row and its map's
parameters. The per-car layout does that once and spends the rest of its time
marching. The marching is identical in both, so the difference is the setup,
which is the 20–30% you measured.

Cache locality favours per-car too. One core walks one car's 108 rays, which
all start at the same pixel and sweep a 20 m disc of the map; by the time it
moves to the next car the relevant part of the raster is in its L2.

Sharding is what makes CPU numbers respectable at all. One shard, meaning one
core, manages around 20 to 25 M rays/s on this machine; 18 shards reach 160 to
200 M. The pool costs a few hundred microseconds per step to dispatch, which is
why the benchmark's CPU default is 4,096 envs: at 1,024 the dispatch is about a
third of the step.

## Why the GPU prefers per-(car, beam)

Three reasons, in order of importance.

**Occupancy.** A GPU hides memory latency by keeping many more threads resident
than it has execution units, and switching between them while loads are in
flight. An A100 holds 2,048 resident threads per SM across 108 SMs, about
221,000. 8,192 per-car threads fill 4% of that; 884,736 per-(car, beam)
threads fill it four times over. The scattered `edt` reads in the march are
exactly the latency this mechanism exists for: each read's address depends on
the previous read's value, so the only way to keep the memory system busy is
more independent rays in flight.

**Divergence.** Threads run in warps of 32 that execute in lockstep, and a
`while` loop runs until the *slowest* lane exits. In the per-(car, beam) layout
a warp is 32 adjacent beams of one car. Their rays start at the same point and
point in similar directions, so they take similar numbers of steps. In the
per-car layout a warp is 32 different cars, each looping over 108 beams; the
warp finishes when the car with the longest total march does, and every other
lane idles for the difference. Across eight maps with different corridor widths
that difference is large.

**Access pattern.** Adjacent lanes in a per-(car, beam) warp read neighbouring
pixels early in the march and share cache lines. Adjacent lanes in a per-car
warp read from 32 unrelated places in a 100 MB raster from the first step.

None of this is specific to lidar. It's the usual GPU rule of mapping the
finest independent unit of work to a thread, and it's why warporacer
parallelises over (car, beam).

## What else warporacer does

The benchmark reproduces three things from `warporacer/sim.py` that are worth
knowing about even if you never change them.

### Recorded launches

`wp.launch(..., record_cmd=True)` returns a `Launch` with the kernel and its
arguments already packed, and `launch.launch()` replays it. The sim steps at
60 Hz over 8,192 envs for millions of steps, and the Python side of a normal
`wp.launch` (type checks, argument packing, slicing the per-shard arrays) is
paid once instead of every step.

### Texture sampling (CUDA)

On CUDA the benchmark also times `lidar_tex_kernel`, which reads the stack
through a `wp.Texture3D` with nearest filtering and border addressing. The
texture cache is built for exactly this kind of spatially scattered, read-only
access, and border addressing returns 0 (wall) outside the volume so the bounds
check disappears from the loop. warporacer measured +14% sim throughput from
it. The sample coordinate is `(col, row, mid + 0.5)` so that nearest sampling
picks `edt[mid, int(row), int(col)]` exactly, which is why the correctness
check expects the texture row to match the other two to the bit.

### Streams

Not in the benchmark, but in the sim: the kernels launch on PyTorch's current
CUDA stream (`wp.stream_from_torch`), so the policy network reads `obs`
straight out of the Warp buffer with no synchronisation between the frameworks.
The observation, reward and done arrays are zero-copy torch views of Warp
arrays for the same reason.

## Putting the number in context

warporacer's README quotes ~7M env-steps/s at 8k envs on a single GPU for the
full loop: physics kernel, lidar kernel, policy forward pass, and the PPO update
amortised over the rollout. Whatever `M env-steps/s` the per-(car, beam) row
shows on your GPU bounds the lidar's share of that budget: at 30M, lidar is at
most about a quarter of the step. If you want to move the number, the levers
are the ones above: more rays in flight, less divergence, cheaper reads. Not
fewer flops.

## Key concepts

| Concept                   | per car                                  | per (car, beam)                  |
|---------------------------|------------------------------------------|----------------------------------|
| **Threads at 8k envs**    | 8,192                                    | 884,736                          |
| **Work per thread**       | 108 marches                              | 1 march                          |
| **Setup per car**         | once                                     | 108 times                        |
| **Warp contents (GPU)**   | 32 cars, 32 unrelated addresses          | 32 beams of one car              |
| **Loop divergence**       | slowest car × 108 beams                  | slowest of 32 similar rays       |
| **CPU, 18 shards**        | 207 M rays/s                             | 159 M rays/s                     |
| **GPU**                   | 4% of resident thread slots              | the layout warporacer ships      |

The core lesson: a thread layout is a decision about **how many independent
memory accesses the hardware can have in flight**. On a CPU core the answer is
"a few", so setup cost and cache locality decide. On a GPU the answer is "tens
of thousands", and a layout that can't supply them leaves the machine idle.
