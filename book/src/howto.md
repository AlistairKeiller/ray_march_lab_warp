## How to Use This Book

Each puzzle keeps the same structure:

- **Overview**: what the kernel has to do and where it sits in the sim
- **Key concepts**: the Warp features and the geometry the puzzle introduces
- **Configuration**: sizes, units, and the layout of the arrays involved
- **Code to complete**: the file in `problems/pXX/` with a `# FILL ME IN` gap
- **Tips**: hints you can open one at a time without seeing the answer
- **Running the code**: the command, and what the output looks like before and
  after you solve it
- **Solution**: the reference implementation with an explanation

The puzzles build on each other, so work through them in order. Puzzle 4 uses
the function you write in Puzzle 3, and Puzzle 6 benchmarks the kernel from
Puzzle 5.

## Running the code

Every puzzle is a Python script that launches your kernel, compares the result
with a reference computed in NumPy, and prints `Puzzle XX complete ✅` when they
agree. The reference runs the same arithmetic in float32, so on CPU the
comparison is exact; on CUDA the tolerance is one pixel, to absorb the ulp or
two that fused multiply-adds move the samples by.

## Prerequisites

### System requirements

- **Python 3.14** and [uv](https://docs.astral.sh/uv/), which installs the
  pinned interpreter and dependencies for you
- **A C++ compiler** for Warp's CPU backend: Xcode command-line tools on macOS,
  gcc or clang on Linux, MSVC build tools on Windows
- **Optionally an NVIDIA GPU** with a driver that supports CUDA 12. Warp uses
  `cuda:0` when it finds one and `cpu` otherwise; every puzzle runs on both

There is no Apple GPU backend in Warp, so on a Mac the kernels run on the CPU.
That is fine for Puzzles 1–5, which are small, and Puzzle 6 scales its defaults
down on CPU.

### Programming knowledge

Basic Python and NumPy. No prior GPU programming is assumed; Warp's kernel
language is Python syntax with types, and each puzzle introduces the parts it
needs.

## Setting up your environment

1. Clone the repository:

   ```bash
   git clone https://github.com/AlistairKeiller/ray_march_lab_warp
   cd ray_march_lab_warp
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you
   do not have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install everything and run your first puzzle:

   ```bash
   uv sync

   # This fails waiting for your implementation. Follow the book.
   uv run python problems/p01/p01.py
   ```

   `uv sync` creates `.venv` with Python 3.14, Warp, NumPy, SciPy and the
   image/yaml readers the map loader needs. The first launch of each puzzle
   compiles its kernels (a second or so on CPU); later runs hit Warp's kernel
   cache.

## Working with puzzles

### Project structure

- **`problems/`**: where you implement your solutions (this is where you work!)
- **`solutions/`**: the reference solutions, which the book quotes
- **`harness/`**: the guarded output arrays and the map loader every puzzle
  imports
- **`maps/`**: the track maps from warporacer, as ROS map yaml + image pairs

### Workflow

1. Open `problems/pXX/pXX.py` and find the `# FILL ME IN` comment
2. Write the kernel body
3. Run `uv run python problems/pXX/pXX.py` until it prints the ✅
4. Compare with `solutions/pXX/pXX.py` and read the explanation in the book

### Guarded output arrays

Every output array a puzzle hands your kernel sits inside a larger allocation
whose margins hold NaN. After the kernel runs the puzzle checks those margins,
and fails if your kernel wrote into one:

```text
❌ Write detected outside of output buffer
```

A kernel can write every expected value into `output` and still write past the
end of it, which comparing values alone cannot detect. This message means your
indexing ran outside the output region. Warp does not bounds-check array
accesses in its default release mode, so without the margins such a write
would corrupt whatever happened to be next in memory, silently.

The margins extend 256 bytes past each end, so a write further out than that
goes unreported. A clean check is good evidence, not proof. Puzzle 6 drives
its kernels from a benchmark loop rather than the harness and has no guarded
array; it checks the two layouts against each other instead.

### Essential commands

```bash
# Run puzzles
uv run python problems/pXX/pXX.py        # your implementation
uv run python solutions/pXX/pXX.py       # the reference

# Test solutions
bash solutions/run.sh                    # all of them
bash solutions/run.sh p03                # one

# The benchmark at full scale (defaults: 8192 envs on CUDA, 1024 on CPU)
uv run python problems/p06/p06.py maps/ --num-envs 8192 --steps 500
uv run python problems/p06/p06.py maps/Spielberg_map.yaml --device cpu

# Checks the book relies on
uv run python scripts/check_skeletons.py # problems == solutions outside the FILL regions
uv run python scripts/check_book_refs.py # every include, link and image in the book resolves

# Figures (needs the viz extra)
uv sync --extra viz
uv run python book/src/puzzle_03/puzzle_03_viz.py
```

## Device support matrix

| Puzzle                 | CPU | CUDA | Notes                                              |
|------------------------|-----|------|----------------------------------------------------|
| 1 - Map                | ✅  | ✅   |                                                    |
| 2 - World to Pixel     | ✅  | ✅   |                                                    |
| 3 - Sphere March       | ✅  | ✅   |                                                    |
| 4 - Lidar              | ✅  | ✅   | Loads one real track map                           |
| 5 - Many Maps          | ✅  | ✅   | Three maps, padded into one raster                 |
| 6 - Lidar Benchmark    | ✅  | ✅   | Texture-sampling variant runs on CUDA only         |

Warp has no Metal or ROCm backend, so AMD and Apple GPUs run the CPU path.
