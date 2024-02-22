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

## More Performance benchmarks

- Multiple loads PR vs Nightly: f32 cpu/cuda, ui8 cpu, CL/CF, downsample/upsample
  - output for uint8 on nightly is not correct

```bash
python -u perf_interp_mode_custom.py --mode=bicubic
```

```
PR
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+git9f94979)   |  Compiled (2.3.0a0+git9f94979)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |       613.765 (+-4.241)       |       4235.248 (+-40.597)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)    |      1541.447 (+-24.051)      |      29100.077 (+-324.852)
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |       326.533 (+-3.006)       |       4525.365 (+-40.800)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)        |       717.898 (+-16.210)      |      28341.565 (+-301.909)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |      1747.792 (+-33.986)      |       6255.118 (+-52.038)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |     10889.297 (+-139.480)     |      44060.329 (+-326.867)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |      2809.926 (+-36.519)      |       6460.797 (+-115.704)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |     17533.680 (+-216.505)     |      50064.101 (+-397.326)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.3.0a0+git9f94979)   |  Compiled (2.3.0a0+git9f94979)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.442 (+-0.048)       |         97.721 (+-0.030)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.059 (+-0.033)       |        205.002 (+-1.391)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       102.428 (+-0.040)       |        103.212 (+-0.142)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       188.340 (+-0.104)       |        260.985 (+-0.094)

Times are in microseconds (us).
```

```
Nightly
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+git0d1e705)   |  Compiled (2.3.0a0+git0d1e705)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |       616.341 (+-5.594)       |       3416.567 (+-35.878)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)    |      1521.868 (+-26.354)      |      21759.632 (+-205.698)
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |       328.502 (+-3.400)       |       3573.109 (+-30.356)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)        |       718.389 (+-12.239)      |      22622.303 (+-321.970)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |      1747.660 (+-21.801)      |       2883.334 (+-20.104)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |     10812.851 (+-162.402)     |      18232.606 (+-247.851)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |      2818.818 (+-26.319)      |       3040.853 (+-33.576)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |     17538.158 (+-199.267)     |      19302.357 (+-296.963)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.3.0a0+git0d1e705)   |  Compiled (2.3.0a0+git0d1e705)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.561 (+-0.067)       |         97.696 (+-0.025)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.166 (+-0.033)       |        198.000 (+-1.160)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       100.714 (+-0.036)       |        100.697 (+-0.059)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       187.752 (+-0.086)       |        244.982 (+-0.445)

Times are in microseconds (us).
```


- Why for f32/cpu cases PR does not match nightly perfs? => not fused buffer on PR
```
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- PR
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.753 (+-0.007)        |         5.992 (+-0.013)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.870 (+-0.066)       |         44.777 (+-0.206)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        3.003 (+-0.009)        |         6.484 (+-0.021)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        18.773 (+-0.067)       |         50.167 (+-0.110)

-- Nightly
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.751 (+-0.008)        |         2.884 (+-0.008)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.847 (+-0.186)       |         18.475 (+-0.053)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.839 (+-0.020)        |         3.041 (+-0.007)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.550 (+-0.070)       |         19.283 (+-0.069)

Times are in milliseconds (ms).
```

- PR with nodes fusion
```
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- PR
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.764 (+-0.073)        |         3.829 (+-0.118)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.794 (+-0.405)       |         22.899 (+-0.739)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.810 (+-0.125)        |         4.090 (+-0.143)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.613 (+-0.631)       |         30.542 (+-1.212)

-- Nightly
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.757 (+-0.097)        |         2.888 (+-0.094)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        11.309 (+-0.587)       |         18.636 (+-0.672)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.831 (+-0.162)        |         3.068 (+-0.113)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.674 (+-0.778)       |         21.635 (+-1.270)

Times are in milliseconds (ms).
```
-> perf diff is due to absent clamping of xscale/yscale vars on Nightly


```
[------------------------------------------------------------------------------------------------ Interpolate, cpu ------------------------------------------------------------------------------------------------]
                                                                                                                                                   |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)       |       604.729 (+-21.112)      |       4643.171 (+-120.072)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)     |      2518.884 (+-115.161)     |       2738.597 (+-75.968)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)       |      1439.730 (+-64.056)      |      18363.379 (+-805.371)
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |       610.372 (+-21.982)      |       4379.546 (+-163.870)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)    |      2539.013 (+-92.057)      |       3946.814 (+-141.687)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |      1440.135 (+-68.106)      |      27478.355 (+-938.279)
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)           |       331.460 (+-13.206)      |       2984.781 (+-105.081)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)         |       766.922 (+-37.422)      |       2759.775 (+-104.016)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)           |       722.362 (+-31.347)      |      18487.445 (+-634.353)
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)          |       331.930 (+-11.282)      |       4388.895 (+-149.076)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)        |       770.590 (+-43.689)      |       3980.828 (+-139.426)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)          |       720.323 (+-28.982)      |      27609.497 (+-1095.222)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)     |      1749.559 (+-78.147)      |       3567.388 (+-96.859)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)   |      1699.452 (+-98.807)      |       3310.554 (+-140.914)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)     |     10822.919 (+-413.699)     |      22618.090 (+-702.884)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |      1760.528 (+-76.161)      |       3679.748 (+-126.815)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)  |      1691.932 (+-70.148)      |       3418.792 (+-157.347)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)    |     10845.607 (+-501.402)     |      23345.186 (+-707.452)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)         |      2871.020 (+-99.820)      |       3762.764 (+-122.563)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)       |      2676.992 (+-138.592)     |       3596.427 (+-132.600)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)         |     17888.706 (+-861.594)     |      23779.833 (+-994.828)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |      2875.350 (+-141.950)     |       4086.004 (+-108.709)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)      |      2676.847 (+-134.831)     |       3918.026 (+-181.103)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)        |     17898.730 (+-601.668)     |      26540.729 (+-985.172)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (1234, 1345)   |        97.246 (+-0.044)       |         97.645 (+-0.047)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (2345, 2456)   |       189.244 (+-0.046)       |        206.848 (+-1.086)
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        98.647 (+-0.033)       |         98.059 (+-0.063)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.439 (+-0.058)       |        208.926 (+-1.396)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (1234, 1345)       |       101.189 (+-0.058)       |        102.533 (+-0.125)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (2345, 2456)       |       188.574 (+-0.110)       |        243.214 (+-0.305)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       101.148 (+-0.052)       |        108.084 (+-0.358)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       188.648 (+-0.086)       |        262.505 (+-0.249)

Times are in microseconds (us).
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


## PyTorch tests

```
pytest -vvv test/test_decomp.py -k "interpolate_bic or upsample_bic" && \
pytest -vvv test/inductor/test_torchinductor_opinfo.py -k "interp or upsampl" && \
pytest -vvv test/inductor/test_torchinductor_dynamic_shapes.py -k "interp or upsampl" && \
pytest -vvv test/functorch/test_aotdispatch.py -k "interp or upsampl" && \
pytest -vvv test/functorch/test_ops.py -k "interp or upsampl" && \
pytest -vvv test/test_meta.py -k "interp or upsampl" && \
pytest -vvv test/test_proxy_tensor.py -k "interp or upsampl" && \
pytest -vvv test/test_decomp.py::HasDecompTest::test_has_decomposition && \
pytest -vvv test/functorch/test_vmap.py -k "interp or upsampl"
```


```
pytest -vvv test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32


============================================================================================================================ short test summary info ============================================================================================================================
FAILED [1.1356s] test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32 - Exception: Caused by sample input at index 0: SampleInput(input=Tensor[size=(2, 3, 4, 4), device="cpu", dtyp
e=torch.float32], args=((3,3)), kwargs={'scale_factor': 'None', 'mode': "'bicubic'", 'align_corners': 'True'}, broadcasts_input=False, name='')
=============================================================================================================================== 1 failed in 3.56s ===
```
