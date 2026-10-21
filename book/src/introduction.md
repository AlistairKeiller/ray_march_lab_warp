# Ray March Lab 🔦

> _"For the things we have to learn before we can do them, we learn by doing
> them."_ Aristotle (Nicomachean Ethics)

Welcome to a short, hands-on set of GPU puzzles in
[NVIDIA Warp](https://nvidia.github.io/warp/), the Python framework that
compiles ordinary-looking Python functions into CUDA (or C++) kernels. The six
puzzles rebuild the lidar simulator from
[warporacer](https://github.com/uci-f1tenth/warporacer), a GPU-parallel
F1TENTH racing sim that marches 8,192 cars × 108 laser beams through a track
image sixty times a simulated second, one piece at a time.

The same six puzzles exist in Mojo, in the twin lab
[ray_march_lab](https://github.com/AlistairKeiller/ray_march_lab). The two
books share their text, their figures, their map loader and their numbering;
only the kernels differ. If you want to see how the same launch looks in the
other language, open the same page there.

## Why ray marching?

A racing policy sees the world through a lidar: a fan of ranges, one per beam,
to the nearest wall. Simulating that for one car is easy. Simulating it for
thousands of cars at once, fast enough to train a reinforcement learning policy
in minutes, is a GPU problem, and the trick that makes it cheap is **sphere
marching** (also called sphere tracing). Instead of stepping along a ray one
pixel at a time, you precompute, for every pixel of the track, the distance to
the nearest wall. Then from any point on a ray you can safely jump exactly that
far without passing through a wall. Wide corridors take a few big jumps; the
ray only slows down as it closes in on something.

That precomputed field is the Euclidean distance transform (EDT) of the
drivable pixels, and it is the only piece of track geometry the sim needs.

## Why Warp?

[Warp](https://github.com/NVIDIA/warp) lets you write the kernel in Python:

- `@wp.kernel` and `@wp.func` compile to CUDA on a GPU and to C++ on a CPU, so
  the same file runs on a laptop and a cluster node
- Kernel arguments are typed arrays and vectors (`wp.array2d(dtype=float)`,
  `wp.vec2`), so most mistakes are caught at compile time
- Launches are plain function calls with a `dim`, with no grid/block arithmetic
  unless you want it
- Arrays convert to and from NumPy (and PyTorch, which is how warporacer feeds
  observations to its policy) without copies when they share a device

The result is that the full racing sim in warporacer is two kernels and a
couple hundred lines of Python.

## Why learn through puzzles?

Each puzzle hands you a working program with one gap in it, a kernel body you
have to write, and a check that fails until you get it right. Reading about
`wp.tid()` is one thing; watching an array of zeros turn into the right answer
is another. The gaps grow with the puzzles, from a one-line map to a full
multi-map lidar kernel, and the last puzzle turns the kernel you wrote into a
benchmark so you can see what it costs at scale.

> **Acknowledgement**: The format of this book, and the shape of the first
> puzzles, follow Modular's
> [Mojo GPU Puzzles](https://puzzles.modular.com/), which in turn follows Sasha
> Rush's [GPU Puzzles](https://github.com/srush/GPU-Puzzles). The kernels are
> warporacer's.

## The GPU programming mindset

Three ideas carry you through every puzzle here.

### One thread per ray

The CPU way to scan 8,192 cars is a loop over cars inside a loop over beams.
The GPU way is to launch 8,192 × 108 threads and let each one march a single
ray. `wp.tid()` tells a thread which ray it owns; the loop disappears.

```python
# CPU
for i in range(num_cars):
    for j in range(num_beams):
        scan[i, j] = march(cars[i], dirs[j])

# GPU (Warp)
@wp.kernel
def lidar(...):
    i, j = wp.tid()
    scan[i, j] = march(cars[i], dirs[j])
```

### Fit the thread grid to the data

Puzzle 6 launches the same march four ways: one thread per (car, beam), one
thread per car, the grid turned round so that neighbouring threads are
neighbouring cars, and the per-beam launch with the one-pixel stepper the
distance field replaces. It times all four. The answers differ between a CPU
and a GPU, and understanding why is most of what GPU performance work is about.

### Data movement is the cost

A ray's samples are scattered reads from a raster that may be 100 MB across
eight maps. Which pixel a thread reads next depends on the value it just read,
so nothing can be prefetched. How many bytes each step touches, and whether
neighbouring threads touch neighbouring pixels, matters more than the handful
of multiplies in between.

## What you will learn

| Part                         | Puzzles | You will write                                               |
|------------------------------|---------|--------------------------------------------------------------|
| I: Fundamentals              | 1–2     | A kernel, a launch, 2D indexing, vectors, clamps             |
| II: Ray marching             | 3–5     | The sphere march, a 108-beam lidar, a padded multi-map stack |
| III: Performance             | 6       | A second thread layout, and a benchmark at warporacer scale  |

**Part I: Fundamentals (Puzzles 1–2)**

- Write a `@wp.kernel`, index it with `wp.tid()`, launch it with `wp.launch`
- Read a 2D raster at a world position: origin, resolution, and the row flip
- Clamp indices so a car off the map lands on a wall instead of out of bounds

**Part II: Ray marching (Puzzles 3–5)**

- Write the sphere-march loop as a `@wp.func` and call it from a kernel
- Rotate a beam into a car's heading and shoot it from the lidar mount point
- Launch a 2D grid over (car, beam) and write a real 108-beam scan
- Stack maps of different sizes into one padded raster and index it per car

**Part III: Performance (Puzzle 6)**

- Write the one-thread-per-car layout and compare it with one-thread-per-beam
- Time the one-pixel stepper the distance field replaces, and a transposed grid
- Benchmark all four at warporacer scale with recorded launches, thread-pool
  sharding on CPU, and (on CUDA) texture sampling

## Ready to get started?

Head to [How to Use This Book](howto.md) for setup, then open Puzzle 1.
