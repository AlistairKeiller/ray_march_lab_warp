# ray_march_lab_warp

GPU ray marching puzzles in [NVIDIA Warp](https://github.com/NVIDIA/warp), in the format of
[Mojo GPU Puzzles](https://puzzles.modular.com/). Six puzzles build the lidar kernel from
[warporacer](https://github.com/uci-f1tenth/warporacer) one piece at a time: pixels to meters, world to
pixel, one sphere-marched ray, a 108-beam scan on a real track, a padded multi-map stack, and a benchmark
at warporacer scale (8k cars x 108 beams x 20 m on 8 maps). Every puzzle runs on CPU or CUDA; Warp picks
`cuda:0` when there is one.

```bash
uv run python problems/p01/p01.py                    # work a puzzle: fill in the FILL ME IN region until it passes
uv run python solutions/p01/p01.py                   # the reference
bash solutions/run.sh                                 # test every solution (or one: bash solutions/run.sh p03)
uv run python problems/p06/p06.py maps/ --num-envs 8192   # the benchmark at full warporacer scale
uv run python scripts/check_skeletons.py             # problems == solutions outside the fill-in regions
uv run python scripts/check_book_refs.py             # every book include/link/image resolves
uv run --extra viz python book/src/puzzle_03/puzzle_03_viz.py  # regenerate a figure
mdbook serve book --open                             # the book (brew install mdbook)
```

## Layout

| file | what it does |
|---|---|
| `problems/pNN/pNN.py` | the puzzle: a kernel with a `# FILL ME IN` gap and a check that fails until you fill it |
| `solutions/pNN/pNN.py` | the same file with the gap filled; the book includes both via `# ANCHOR` markers |
| `harness/canary.py` | guarded output arrays: NaN margins either side of the output catch out-of-bounds writes |
| `harness/track.py` | map yaml + image → wall-distance field in px, random drivable poses, zero-padded multi-map stacks |
| `maps/` | ROS-style track maps from warporacer: the F1TENTH racetracks plus two hallway maps |
| `book/` | the mdbook: one page per puzzle with the code to complete, tips, expected output and the solution |
| `scripts/` | CI-style checks: problem/solution sync, book reference resolution |
