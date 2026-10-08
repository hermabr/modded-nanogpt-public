# Exact-match retrieval record statistics (this PR)

- code: commit `9025a886`, 570 trained steps (560 scheduled + 10 extension); validation index over the first 350 training shards
- GPUs: 8x H100 80GB HBM3 (driver 580.173.02), Nebius on-demand VM `8gpu-128vcpu-1600gb` (Intel Xeon Platinum 8468, 128 threads, 1575 GiB)
- Python `3.12.3` · PyTorch `2.10.0+cu128` · Triton `3.6.0` · kernels `0.16.1` · huggingface-hub `1.29.0` · CUDA runtime 13.0
- data: FineWeb-100B (`data/fineweb100B`), validated on its `fineweb_val_000000.bin`
- protocol: one session (2026-10-08 01:36-03:25 UTC), legs interleaved this PR, this PR, baseline (x8); `TRAIN_SEED` = seed column; before every leg: no stray processes, `/dev/shm` emptied, GPU memory back to 0, train shards 1-350 + val shard re-read into page cache (warm), compile caches kept (compilation is untimed)
- runs: `16`, all counted; all 16 raw logs ship in this folder. The session was restarted once, after two legs at the parent commit `847833fa`, to add the validation index build time to the log; those legs are not included.

| metric | value |
| --- | ---: |
| mean wall (train_time) | 7.065 s |
| wall sample std | 0.013 s |
| wall range | 7.042-7.092 s |
| mean val loss | 3.2760125000 |
| val loss sample std | 0.0037489332 |
| t-statistic vs 3.28 gate (one-sided) | 4.25 |
| one-sided p vs 3.28 gate | 3.46e-04 |

Against the baseline legs of the same session (`../baseline/`, ANVIL2, n=8):

| metric | value |
| --- | ---: |
| mean wall difference | -33.235 s (-82.5%) |
| Welch t-test on wall, two-sided p | 2.8e-21 |
| mean val loss difference | -0.00745 |

## Runs (session order)

The last column is when the validation index was ready, from the start of the clock (from the log's `validation index built` line).

| leg | seed | log | steps | final val loss | train_time | val index ready |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0 | `d72ca4ea-5d6d-4f67-ab17-42ec832a7977.txt` | 570 | 3.2763 | 7.042 s | 6.48 s |
| 2 | 1 | `4d07bb5d-1e02-45cb-adc3-b5d549c6d142.txt` | 570 | 3.2759 | 7.070 s | 6.43 s |
| 4 | 2 | `010406bc-3554-4f21-822a-9bdd4c66b944.txt` | 570 | 3.2744 | 7.056 s | 6.45 s |
| 5 | 3 | `60f3e405-b03b-4928-ae56-faac37f41bd6.txt` | 570 | 3.2809 | 7.053 s | 6.46 s |
| 7 | 4 | `7773a1ab-c2a6-4650-abe4-775828a8d286.txt` | 570 | 3.2776 | 7.058 s | 6.42 s |
| 8 | 5 | `f53d5135-1e83-4085-8b6c-1b9855035cd7.txt` | 570 | 3.2717 | 7.073 s | 6.44 s |
| 10 | 6 | `2cfb6c3b-9443-4acd-8103-b6a0bb16c3ea.txt` | 570 | 3.2829 | 7.068 s | 6.42 s |
| 11 | 7 | `b2ceafe9-12b6-4e87-9e28-d215d54b0808.txt` | 570 | 3.2761 | 7.064 s | 6.48 s |
| 13 | 8 | `a261496e-acf4-4c8a-9053-6ec1044e1653.txt` | 570 | 3.2714 | 7.092 s | 6.46 s |
| 14 | 9 | `ce955a0f-567a-402f-aee7-1c56912b2817.txt` | 570 | 3.2752 | 7.056 s | 6.49 s |
| 16 | 10 | `9aa4be30-d0c1-49f5-833b-3ec7e8f10821.txt` | 570 | 3.2825 | 7.044 s | 6.46 s |
| 17 | 11 | `8ce10159-2031-467b-9563-4b337785bb51.txt` | 570 | 3.2705 | 7.082 s | 6.45 s |
| 19 | 12 | `78b4be94-5141-4d9f-bbb2-b66a7072e21a.txt` | 570 | 3.2786 | 7.060 s | 6.48 s |
| 20 | 13 | `51061197-7945-4bdf-b432-019441bab98e.txt` | 570 | 3.2727 | 7.068 s | 6.54 s |
| 22 | 14 | `331547ad-57f9-44cd-a9e6-a0aa6ec191e4.txt` | 570 | 3.2743 | 7.077 s | 6.47 s |
| 23 | 15 | `e199ae79-688d-49db-9326-5f77b926b161.txt` | 570 | 3.2752 | 7.074 s | 6.49 s |
