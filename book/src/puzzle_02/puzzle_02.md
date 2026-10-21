# Puzzle 2: World to Pixel

## Overview

Implement a kernel that, for each car, looks up the wall distance under the car
and converts it to the car's **clearance** in meters: the distance from the
car's centre to the nearest wall, minus half the car's diagonal. A negative
clearance is a crash.

**Note:** _You have 1 thread per car. Cars can be anywhere, including off the
map, so the pixel indices need clamping._

## Key concepts

In this puzzle, you'll learn about:

- 2D arrays: `wp.array2d(dtype=float)`, indexed `edt[row, col]`
- Vector arguments: `wp.vec2`, with components `origin[0]` and `origin[1]`
- Float to int conversion with `wp.int32()`, which truncates toward zero
- Clamping indices with `wp.clamp` instead of branching
- Reading an array's shape inside a kernel: `edt.shape[0]`

For each car \\(i\\) at world position \\((x_i, y_i)\\):

\\[\Large col = \left\lfloor \frac{x_i - o_x}{res} \right\rfloor \qquad
row = \left\lfloor (H - 1) - \frac{y_i - o_y}{res} \right\rfloor\\]

\\[\Large output[i] = edt[row, col] \times res - \frac{D}{2} \qquad
D^2 = W_{car}^2 + L_{car}^2\\]

where \\((o_x, o_y)\\) is the world position of the map's bottom-left pixel,
\\(res\\) its resolution in meters per pixel, \\(H\\) its height in pixels,
and \\(W_{car} = 0.42\\) m and \\(L_{car} = 0.58\\) m are the F1TENTH's width
and length. \\(D\\) is the car's diagonal, so \\(D / 2\\) is the furthest any
corner of the car reaches from its centre; in the code it is `HALF_DIAG`.

### The row flip

A ROS map's `origin` is the world position of the **bottom-left** pixel, and
world \\(y\\) increases upward. Image rows increase downward from the top. So
\\(x\\) maps straight onto columns, but \\(y\\) maps onto rows from the bottom:
row \\(H - 1\\) is \\(y = o_y\\). Get this flip wrong and every car drives a
mirrored track, and nothing complains until the lidar looks wrong.

### Why clamp

A car that drives off the edge of the map has a pixel index outside the raster.
Reading `edt[-3, 12]` is an out-of-bounds memory access, and Warp does not check
for it in release builds. Clamping to the border pixel reads a wall (`edt` is 0
there), which is also the right answer: off the map counts as crashed.

## Configuration

- Cars: `NUM_CARS = 4`, as rows of `[x, y, psi]`
- Map: an `8 x 8` px room with a 1 px wall all round, `RES = 0.5` m/px,
  `ORIGIN = (-2, -2)`
- Car size: `CAR_W = 0.42` m by `CAR_L = 0.58` m, the F1TENTH's width and length;
  `HALF_DIAG = hypot(CAR_W, CAR_L) / 2` is half its diagonal
- One thread per car: `wp.launch(..., dim=NUM_CARS)`

Notes:

- **Shape in kernel**: `edt.shape[0]` is the number of rows, `edt.shape[1]` the
  number of columns
- **vec2 argument**: pass `wp.vec2(ORIGIN)` from Python; index it `origin[0]`,
  `origin[1]` in the kernel
- **Module constants**: `HALF_DIAG` is a plain Python float at module level,
  and Warp bakes it into the kernel as a constant

## Code to complete

```python
{{#include ../../../problems/p02/p02.py:clearance}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p02/p02.py" class="filename">View full file: problems/p02/p02.py</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Column: `(cars[i, 0] - origin[0]) / res`, truncated with `wp.int32(...)`
2. Row: `float(edt.shape[0] - 1) - (cars[i, 1] - origin[1]) / res`, truncated
   the same way
3. Clamp both to `[0, shape - 1]`: `wp.clamp(value, 0, edt.shape[k] - 1)`
4. `output[i] = edt[row, col] * res - HALF_DIAG`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run python problems/p02/p02.py
```

Your output will look like this if the puzzle isn't solved yet:

```txt
room (px to wall):
[[0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 1. 1. 1. 1. 1. 0.]
 [0. 1. 2. 2. 2. 2. 1. 0.]
 [0. 1. 2. 3. 3. 2. 1. 0.]
 [0. 1. 2. 3. 3. 2. 1. 0.]
 [0. 1. 2. 2. 2. 2. 1. 0.]
 [0. 1. 1. 1. 1. 1. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0.]]
out: [0. 0. 0. 0.]
expected: [ 1.1419497   0.14194971  0.6419497  -0.3580503 ]
```

## Solution

<details class="solution-details">
<summary></summary>

```python
{{#include ../../../solutions/p02/p02.py:clearance_solution}}
```

<div class="solution-explanation">

This solution:

- Converts world \\(x\\) to a column and world \\(y\\) to a row, flipping the
  row axis with `float(edt.shape[0] - 1) - ...`
- Truncates with `wp.int32()` and clamps both indices into the raster
- Reads the wall distance, scales it to meters, and subtracts `HALF_DIAG`

Following the four cars through:

```txt
Car 0 at (0.0, 0.0):    col = (0 + 2) / 0.5 = 4,   row = 7 - 4   = 3 -> edt[3, 4] = 3 px -> 1.5 - 0.358 =  1.142 m
Car 1 at (1.2, -1.4):   col = 3.2 / 0.5 = 6.4 -> 6, row = 7 - 1.2 = 5.8 -> 5 -> edt[5, 6] = 1 px -> 0.142 m
Car 2 at (-0.9, 0.3):   col = 1.1 / 0.5 = 2.2 -> 2, row = 7 - 4.6 = 2.4 -> 2 -> edt[2, 2] = 2 px -> 0.642 m
Car 3 at (5.0, 9.0):    col = 14 -> clamped to 7,   row = -15 -> clamped to 0 -> edt[0, 7] = 0 px -> -0.358 m (crash)
```

`wp.int32` truncates toward zero, so a car a fraction of a pixel left of the
map (say `(x - o_x) / res = -0.3`) lands in column 0, the wall, before the clamp
even runs. Either way it reads 0 and reports a crash, which is what the sim
wants.

This is the lookup in warporacer's `step_kernel`, where the result feeds the
crash test and a wall-proximity penalty. The only difference is that the sim
reads the map's height and width from per-map arrays, because it runs several
maps at once. That is Puzzle 5.

</div>
</details>
