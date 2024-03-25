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

- with removed lowering
```

[------------------------------------------------------------------------------------------------ Interpolate, cpu ------------------------------------------------------------------------------------------------]
                                                                                                                                                   |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)       |       604.085 (+-16.902)      |       4633.044 (+-169.113)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)     |      2527.605 (+-82.224)      |       2515.201 (+-96.671)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)       |      1437.753 (+-63.469)      |      16863.372 (+-537.516)
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |       601.387 (+-23.759)      |       4245.030 (+-128.052)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)    |      2519.921 (+-88.057)      |       3754.295 (+-145.889)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |      1437.797 (+-67.080)      |      26636.957 (+-783.333)

      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)           |       330.492 (+-13.437)      |       2954.356 (+-83.891)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)         |       766.166 (+-36.492)      |       2726.281 (+-108.499)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)           |       720.073 (+-27.341)      |      18362.655 (+-680.200)
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)          |       331.324 (+-15.170)      |       4386.872 (+-185.608)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)        |       765.917 (+-27.948)      |       3973.441 (+-144.992)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)          |       720.477 (+-32.366)      |      27571.645 (+-801.155)

      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)     |      1753.898 (+-68.212)      |       3566.561 (+-117.720)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)   |      1687.971 (+-93.004)      |       3307.574 (+-102.981)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)     |     10832.979 (+-495.372)     |      22631.826 (+-775.372)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |      1764.605 (+-62.823)      |       3855.001 (+-129.938)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)  |      1696.811 (+-86.157)      |       3569.518 (+-137.759)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)    |     10851.580 (+-444.867)     |      24582.246 (+-964.648)

      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)         |      2997.869 (+-126.771)     |       3738.697 (+-98.715)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (200, 300)       |      2778.684 (+-135.415)     |       3599.904 (+-142.332)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (600, 700)         |     18747.305 (+-813.711)     |      23746.943 (+-747.546)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |      2981.568 (+-121.068)     |       4097.917 (+-111.184)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (200, 300)      |      2819.228 (+-135.218)     |       3913.742 (+-183.588)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)        |     18737.474 (+-843.576)     |      26464.441 (+-781.730)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (1234, 1345)   |        97.247 (+-0.078)       |         97.697 (+-0.055)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (2345, 2456)   |       189.184 (+-0.042)       |        206.908 (+-1.826)
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        98.612 (+-0.034)       |         98.049 (+-0.112)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.363 (+-0.041)       |        209.164 (+-1.187)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (1234, 1345)       |       100.181 (+-0.042)       |        102.946 (+-0.023)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (2345, 2456)       |       188.648 (+-0.057)       |        248.229 (+-0.551)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       100.336 (+-0.074)       |        108.189 (+-0.361)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       188.912 (+-0.122)       |        262.589 (+-0.299)

Times are in microseconds (us).

```

- Benchmarks on 26/02/2024
```
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- No fusion (8ef4a43a31a)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.755 (+-0.086)        |         3.761 (+-0.142)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.850 (+-0.494)       |         33.742 (+-1.240)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.876 (+-0.094)        |         4.069 (+-0.143)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.765 (+-0.876)       |         30.508 (+-0.999)

- No fusion, removed xscale/yscale clamp
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.757 (+-0.077)        |         2.879 (+-0.092)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        12.297 (+-0.530)       |         18.515 (+-0.632)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.832 (+-0.129)        |         3.045 (+-0.077)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.664 (+-0.816)       |         19.347 (+-0.782)

- Nightly (79df8976081)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.758 (+-0.056)        |         2.894 (+-0.084)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.842 (+-0.426)       |         18.591 (+-0.515)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.829 (+-0.131)        |         3.064 (+-0.087)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.717 (+-0.694)       |         19.528 (+-0.728)

Times are in milliseconds (ms).
```

- Benchmarks on 29/02/2024
```
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.3.0a0+gitXXXXXX)   |  Compiled (2.3.0a0+gitXXXXXX)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Baseline (faf22d6afc7)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.750 (+-0.055)        |         3.857 (+-0.129)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.819 (+-0.500)       |         26.272 (+-0.814)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.829 (+-0.091)        |         4.135 (+-0.142)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.608 (+-0.775)       |         30.879 (+-1.006)

- Nightly (09aefe15024)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.759 (+-0.072)        |         2.911 (+-0.093)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.894 (+-0.465)       |         18.425 (+-0.483)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.849 (+-0.093)        |         3.146 (+-0.090)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.681 (+-0.839)       |         22.773 (+-0.755)

- Single load, (wy * wx * src).sum((-1, -3))
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.875 (+-0.202)        |         7.997 (+-0.486)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        11.063 (+-0.438)       |         82.132 (+-4.637)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        3.188 (+-0.253)        |         8.096 (+-0.470)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.936 (+-0.652)       |         81.725 (+-2.891)

- Single load, (wy * (wx * src).sum(-1)).sum(-2)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.764 (+-0.075)        |         8.915 (+-0.376)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.866 (+-0.377)       |         87.545 (+-3.901)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.865 (+-0.146)        |         9.536 (+-0.412)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.647 (+-0.672)       |         90.859 (+-2.887)

Times are in milliseconds (ms).
```


- Benchmarks on 13/03/2024
```

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.3.0a0+git0d1e705)   |  Compiled (2.3.0a0+git0d1e705)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Baseline


- Using _upsample_get_cubic_coefficients with stack
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.110 (+-0.020)       |        146.472 (+-1.779)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       190.564 (+-0.033)       |        146.903 (+-0.557)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       101.165 (+-0.037)       |        148.289 (+-0.999)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       187.869 (+-0.183)       |        342.486 (+-0.685)

- Nightly
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.020 (+-0.044)       |         97.735 (+-0.019)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.265 (+-0.070)       |        198.084 (+-1.227)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       101.033 (+-0.041)       |        102.978 (+-0.087)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       187.650 (+-0.111)       |        247.638 (+-0.390)

Times are in microseconds (us).
```

- Benchmarks on 21/03/2024

```
[----------------------------------------------------------------------------------------------- Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                 |  Eager (2.4.0a0+git588d264)   |  Compiled (2.4.0a0+git588d264)
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Using _upsample_get_cubic_coefficients with stack on cpu
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.753 (+-0.009)        |         3.285 (+-0.008)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.789 (+-0.048)       |         20.450 (+-0.064)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.812 (+-0.009)        |         3.586 (+-0.008)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.569 (+-0.075)       |         24.086 (+-0.116)

      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        6.851 (+-0.046)        |         12.913 (+-0.119)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        42.884 (+-0.092)       |         81.793 (+-0.365)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        11.065 (+-0.050)       |         38.312 (+-0.101)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        70.058 (+-0.459)       |        242.863 (+-1.679)

- Nightly
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        1.754 (+-0.010)        |         2.909 (+-0.013)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        10.812 (+-0.049)       |         18.324 (+-0.056)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        2.807 (+-0.008)        |         3.141 (+-0.012)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        17.622 (+-0.115)       |         19.931 (+-0.132)

      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)  |        6.894 (+-0.068)        |         26.939 (+-0.139)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)  |        44.038 (+-1.361)       |        174.227 (+-1.944)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)      |        11.987 (+-0.456)       |         54.520 (+-4.063)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (600, 700)      |        70.464 (+-0.950)       |        321.885 (+-1.711)


Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------ Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                     |  Eager (2.4.0a0+git588d264)   |  Compiled (2.4.0a0+git588d264)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Using _upsample_get_cubic_coefficients with stack on cpu
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.563 (+-0.023)       |         97.944 (+-0.034)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.702 (+-0.017)       |        207.977 (+-1.601)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       101.633 (+-0.038)       |        109.744 (+-0.348)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       187.997 (+-0.052)       |        275.266 (+-0.188)

      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |       462.782 (+-0.025)       |        382.829 (+-0.053)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       782.453 (+-0.098)       |        835.870 (+-6.488)
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       466.985 (+-0.032)       |        384.207 (+-0.042)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       768.115 (+-0.168)       |        673.688 (+-3.832)

- Nightly
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |        97.606 (+-0.030)       |         97.930 (+-0.052)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       189.021 (+-0.024)       |        201.023 (+-1.364)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       100.460 (+-0.029)       |        103.046 (+-0.103)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       188.279 (+-0.056)       |        250.370 (+-0.536)

      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)  |       462.072 (+-0.016)       |        382.587 (+-0.044)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)  |       779.839 (+-0.110)       |        807.577 (+-5.938)
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1234, 1345)      |       466.559 (+-0.019)       |        384.116 (+-0.021)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (2345, 2456)      |       770.195 (+-0.229)       |        676.034 (+-1.447)

Times are in microseconds (us).
```




## Questions

- Why there is no fusion between loops computing weights?
  - /tmp/pth/inductor/torch_compile_debug/run_2024_02_22_13_39_38_283171-pid_50733-ac=false/torchinductor/model___9.0/output_code.py
  - /tmp/pth/inductor/torch_compile_debug/run_2024_02_22_13_39_38_283171-pid_50733-ac=false/torchinductor/model___9.0/output_code2.py
- Is it possible to compute weights max using the same single for-loop as in output_code2.py ?



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
