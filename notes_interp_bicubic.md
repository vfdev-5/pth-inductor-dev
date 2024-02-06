# PR

## Performance benchmark

- Nightly
```bash
python perf_interp_mode.py output --tag=Nightly --mode=bicubic
```

- PR
```bash
python perf_interp_mode.py output --tag=PR --mode=bicubic

pr_pkl=output/20240118-140606-upsample-bicubic-PR.pkl
ni_pkl=output/20240118-133434-upsample-bicubic-Nightly.pkl
out_name=$(date "+%Y%m%d-%H%M%S")-upsample-bicubic-pr_vs_nightly

python -u make_results_table_from_pickles.py output/${out_name}.md $pr_pkl $ni_pkl
python -u perf_results_compute_speedup_v2.py output/${out_name}-speedup.md $pr_pkl $ni_pkl --compare "Compiled .+ PR;Compiled .+ Nightly;speed-up PR vs Nightly"
```


## First Performance benchmark

- `main`
```
[-------------------------- Interpolate bicubic, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   647.2  |   2828.6
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  2163.4  |   2076.7
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   263.5  |   3284.6
      Input (3, 345, 456), torch.float32, torch.channels_last      |  3585.1  |   2400.3

Times are in microseconds (us).

[------------------------ Interpolate bicubic, AA=false, cuda -------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   17.2  |   352.0
      Input (3, 345, 456), torch.float32, torch.channels_last      |   17.5  |   357.4

Times are in microseconds (us).


[-------------------------- Interpolate bicubic, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   636.3  |   2945.4
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  2167.8  |   2092.1
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   271.4  |   3308.3
      Input (3, 345, 456), torch.float32, torch.channels_last      |  3560.5  |   2394.4

Times are in microseconds (us).

[------------------------ Interpolate bicubic, AA=false, cuda -------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   15.5  |   305.1
      Input (3, 345, 456), torch.float32, torch.channels_last      |   15.8  |   339.2

Times are in microseconds (us).
```

- PR
```
[-------------------------- Interpolate bicubic, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   634.5  |   3579.6
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  2143.6  |   2237.8
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   261.2  |   2883.2
      Input (3, 345, 456), torch.float32, torch.channels_last      |  3544.3  |   1814.4

Times are in microseconds (us).

[------------------------ Interpolate bicubic, AA=false, cuda -------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   14.3  |    43.2
      Input (3, 345, 456), torch.float32, torch.channels_last      |   14.4  |    40.1

Times are in microseconds (us).

[-------------------------- Interpolate bicubic, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   809.3  |   3626.0
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  2154.3  |   2270.1
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   265.6  |   7009.8
      Input (3, 345, 456), torch.float32, torch.channels_last      |  3560.2  |   1832.8

Times are in microseconds (us).

[------------------------ Interpolate bicubic, AA=false, cuda -------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   15.5  |    39.6
      Input (3, 345, 456), torch.float32, torch.channels_last      |   15.7  |    39.9

Times are in microseconds (us).
```