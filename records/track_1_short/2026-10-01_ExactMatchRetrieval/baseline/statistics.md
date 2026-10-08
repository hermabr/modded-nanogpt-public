# Baseline (ANVIL2, record #92) statistics — same VM, interleaved

- code: upstream master `4ea6b937` (ANVIL2 record after the `track_1_short` refactor), 1194 trained steps, with only the train and val paths in `track_1_short/config.py` changed from `data/fineweb10B` to `data/fineweb100B`
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb` (Intel Xeon Platinum 8468, 128 threads, 1575 GiB)
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- data: FineWeb-100B (`data/fineweb100B`), validated on its `fineweb_val_000000.bin`
- protocol: one session (2026-10-08 01:36-03:25 UTC), legs interleaved this PR, this PR, baseline (x8); `TRAIN_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, train shards 1-350 + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `8`, all counted

| metric | value |
| --- | ---: |
| mean wall (train_time) | 40.300 s |
| wall sample std | 0.063 s |
| wall range | 40.256-40.451 s |
| mean val loss | 3.2834625000 |
| val loss sample std | 0.0050785086 |
| t-statistic vs 3.28 gate (one-sided) | -1.93 |
| one-sided p vs 3.28 gate | 9.52e-01 |

## Runs (session order)

| leg | seed | log | steps | final val loss | train_time |
| ---: | ---: | --- | ---: | ---: | ---: |
| 3 | 0 | `cb8de677-5b90-449e-99d6-578179f84944.txt` | 1194 | 3.2809 | 40.451 s |
| 6 | 1 | `014699e7-5ddb-45b5-bc30-c87f44586cb6.txt` | 1194 | 3.2815 | 40.297 s |
| 9 | 2 | `e616f85f-bc99-49c8-bc47-bf9cabb5ac3b.txt` | 1194 | 3.2929 | 40.271 s |
| 12 | 3 | `984b55f7-2f6e-4abc-8df5-0294b5cb011b.txt` | 1194 | 3.2896 | 40.258 s |
| 15 | 4 | `b5fef571-1000-4e71-a160-e68fc3f9ad2a.txt` | 1194 | 3.2777 | 40.256 s |
| 18 | 5 | `3f2f2352-a5be-48fb-be72-60280920a9d9.txt` | 1194 | 3.2818 | 40.298 s |
| 21 | 6 | `833a217b-44c1-4a4e-a4d6-bb49c3281d6f.txt` | 1194 | 3.2823 | 40.277 s |
| 24 | 7 | `5f75721b-fdf1-43a4-8a02-4ca1315f73ba.txt` | 1194 | 3.2810 | 40.290 s |
