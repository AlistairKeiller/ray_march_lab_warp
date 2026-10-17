# Puzzle 1: Map

## Overview

Implement a kernel that converts each wall distance in `edt`, measured in
pixels, to meters by multiplying it by the map resolution `res`, storing the
result in `output`.

**Note:** _You have 1 thread per position._

## Key concepts

In this puzzle, you'll learn about:

- Writing a GPU kernel with `@wp.kernel`
- Thread indexing with `wp.tid()`
- Typed array arguments: `wp.array(dtype=float)`
- Scalar kernel arguments
- Launching with `wp.launch(kernel, dim=..., inputs=..., outputs=...)`

For each position \\(i\\):
\\[\Large output[i] = edt[i] \times res\\]

### Where this sits in the sim

Every track in warporacer is a ROS map: an image whose white pixels are
drivable, plus a yaml with the map's `resolution` in meters per pixel. The sim
precomputes the distance from every pixel to the nearest wall, in pixels, and
converts to meters whenever it needs a physical quantity, such as how far a car
is from the wall. This kernel is that conversion.

## Configuration

- Array size: `SIZE = 4` elements
- Resolution: `RES = 0.05` m/px, the F1TENTH default
- One thread per element: `wp.launch(..., dim=SIZE)`

Notes:

- **Kernel**: `@wp.kernel` compiles a Python function to CUDA, or to C++ on
  the CPU. Every argument needs a type annotation
- **Thread index**: `wp.tid()` returns this thread's index in the launch. With
  `dim=SIZE` it runs from `0` to `SIZE - 1`
- **Arrays**: `wp.array(dtype=float)` is a 1D array of `float32`; `edt[i]`
  reads and `output[i] = ...` writes
- **Launch**: `wp.launch` queues the kernel and `wp.synchronize_device()`
  waits for it. On CPU the launch is synchronous already; the pattern is the
  same so the file runs unchanged on CUDA

## Code to complete

```python
{{#include ../../../problems/p01/p01.py:px_to_m}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.py" class="filename">View full file: problems/p01/p01.py</a>

> **Note**: The skeleton declares `i = wp.tid()` for you, but nothing uses `i`
> until you fill in the kernel. It still compiles and runs; it just leaves
> `output` as the zeros it was handed. Later puzzles scaffold their variables
> the same way.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. `i = wp.tid()` is already there
2. Multiply `edt[i]` by `res`
3. Store the result in `output[i]`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run python problems/p01/p01.py
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0. 0. 0. 0.]
expected: [0.   0.05 0.1  0.15]
```

## Solution

<details class="solution-details">
<summary></summary>

```python
{{#include ../../../solutions/p01/p01.py:px_to_m_solution}}
```

<div class="solution-explanation">

This solution:

- Gets the thread index with `i = wp.tid()`
- Scales the pixel distance to meters: `output[i] = edt[i] * res`

Two things worth noticing. Warp's `float` is 32-bit, so `edt[i] * res` is a
single-precision multiply and matches the NumPy reference (also computed in
`float32`) exactly. And there is no bounds check, because the launch has exactly
`SIZE` threads and the array has exactly `SIZE` elements. Puzzle 2 is where
indices start needing guards.

</div>
</details>
