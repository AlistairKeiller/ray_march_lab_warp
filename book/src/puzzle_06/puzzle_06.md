# Puzzle 6: Lidar Benchmark

> ## Thread layout and performance
>
> Part III turns the kernel you built into a measurement. Puzzles 4 and 5
> launched one thread per (car, beam), which is the layout warporacer ships.
> The obvious alternative is one thread per car that loops over its 108 beams.
> Both produce the same scan. Here you write the second one and time both at
> warporacer's scale, so you can see which layout the hardware prefers, and why
> the answer is different on a CPU and a GPU.

## Overview

You'll implement `lidar_car_kernel`, a per-car version of the Puzzle 5 kernel,
and run a benchmark that reproduces the lidar half of warporacer's `Env`:

- loads up to 8 random maps from `maps/` and pads them into one `(8, H, W)`
  raster, as `Env.rotate()` does
- spawns 8,192 cars on CUDA (4,096 on CPU) at random drivable poses with random
  headings, spread across the maps
- records one launch per shard and replays it, syncing only at the ends, the way
  the sim's `step()` does
- on CPU, shards the cars across a thread pool, because a Warp CPU kernel runs
  on one core
- on CUDA, also times warporacer's texture-sampling variant

and reports ms per step, rays per second and env-steps per second for each
layout.

## Background: what the sim pays for lidar

warporacer's step is two kernels: physics/reward/respawn (an RK4 kinematic
bicycle at six substeps, a wall check, centerline progress) and the lidar. Per
car the physics is a few hundred flops on values that live in registers. The
lidar is 108 rays, each taking somewhere between 2 and 30 data-dependent reads
from a raster that does not fit in cache. The README quotes ~7M env-steps/s
for the whole training loop at 8k envs on one GPU, policy forward pass and PPO
bookkeeping included, so the lidar kernel on its own has to run a good deal
faster than that. The benchmark tells you how much faster, on your hardware,
and whether the layout matters.

## Learning path

This puzzle is in two parts:

### **[Thread layout](./thread_layout.md)**

Write the per-car kernel and run the benchmark.

**What you'll do:**

- Complete `lidar_car_kernel`: one thread per car, a `for` loop over beams
- Run both layouts on your device and see which wins
- Scale the run with `--num-envs`, `--steps` and `--device`

### **[Performance comparison](./performance.md)**

Why the two layouts land where they do, and what the sim does on top.

**What you'll learn:**

- Why 8k threads can't keep a GPU busy and 884k can
- How divergence in a `while` loop interacts with the two layouts
- What recorded launches, CPU sharding and texture sampling each buy

## Getting started

Start with **[Thread layout](./thread_layout.md)**, then read
**[Performance comparison](./performance.md)** with your own numbers in front of
you.
