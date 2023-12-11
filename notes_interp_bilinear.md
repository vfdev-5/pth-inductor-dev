# PR

## Run benchmarks

- On Nightly
```bash
python perf_interp_mode.py output --tag=Nightly --mode=bilinear
```

- On PR
```bash
python perf_interp_mode.py output --tag=PR --mode=bilinear

pr_pkl=output/XYZ-pr.pkl
ni_pkl=output/ABC-nightly.pkl
out_name=$(date "+%Y%m%d-%H%M%S")-upsample-bilinear-pr_vs_nightly

python -u make_results_table_from_pickles.py output/${out_name}.md $pr_pkl $ni_pkl
python -u perf_results_compute_speedup_v2.py output/${out_name}-speedup.md $pr_pkl $ni_pkl --compare "Compiled .+ PR;Compiled .+ Nightly;speed-up PR vs Nightly"
```

## Perf measurements

- 11/12/2023, PR use `xp1 = torch.where(x < in_w - 1, x + 1, x)` instead `xp1 = (x + 1).clamp(max=in_w - 1)`
```
[------------------------------------------------------------------------------------------------ Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                  |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)  |        1.647 (+-0.006)        |         1.329 (+-0.006)

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------- Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                      |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)  |        98.289 (+-0.017)       |         97.402 (+-0.018)

Times are in microseconds (us).
```

- 11/12/2023, PR use `xp1 = (x + 1).clamp(max=in_w - 1)`
```
[------------------------------------------------------------------------------------------------ Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                  |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)  |        1.663 (+-0.050)        |         1.320 (+-0.007)

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------- Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                      |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)  |        98.276 (+-0.017)       |         97.456 (+-0.151)

Times are in microseconds (us).
```


---

- PR (one load)
```
python perf_interp_bilinear_custom.py

[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   608.5  |   1012.4
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   605.8  |   1011.2
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   280.1  |   1188.8
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   278.9  |   1191.6
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  2548.1  |    892.3
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2788.2  |    872.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   655.6  |   1033.6
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   657.8  |   1032.4
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1477.1  |   4125.2
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1468.6  |   4108.9
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   563.9  |   5040.0
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   565.5  |   5026.2

Times are in microseconds (us).


compile dynamic=True

[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   607.8  |    929.4
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   608.2  |    929.6
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   278.7  |   1167.9
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   278.4  |   1168.0
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  3597.5  |    871.2
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2604.2  |    871.4
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   654.7  |   1088.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   656.0  |   1086.8
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1471.6  |   4102.7
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1466.0  |   4122.9
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   563.6  |   5524.4
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   565.9  |   5512.6

Times are in microseconds (us).
```

- Nightly (wrong result on uint8)
```
python perf_interp_bilinear_custom.py

[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   608.5  |   1202.6
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   608.8  |   1202.1
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   277.1  |   1086.1
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   278.4  |   1083.7
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  3160.4  |    418.2
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2588.4  |    417.1
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   652.3  |    374.3
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   654.4  |    374.8
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1480.9  |   2465.8
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1480.0  |   2462.9
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   563.4  |   2200.4
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   560.3  |   2199.6

Times are in microseconds (us).


compile dynamic=False
[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   609.1  |   1205.1
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   609.6  |   1202.7
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   278.4  |   1083.5
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   277.3  |   1085.7
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  2848.1  |   1119.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2580.0  |   1122.1
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   657.0  |    999.5
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   655.2  |   1001.6
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1471.1  |   7396.1
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1468.0  |   7427.8
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   560.6  |   6633.9
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   561.1  |   6633.8

Times are in microseconds (us).


compile dynamic=True
[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   612.6  |    438.9
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   612.4  |    438.0
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   279.4  |    397.6
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   279.5  |    397.4
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  2852.3  |    417.1
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2792.9  |    416.5
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   654.2  |    371.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   652.0  |    371.7
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1475.5  |   2456.4
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1478.5  |   2465.6
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   562.3  |   2190.2
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   560.8  |   2191.4

Times are in microseconds (us).


[-------------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------------]
                                                                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=True     |   603.1  |   1846.5
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False    |   607.0  |   1847.0
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=True         |   278.0  |   2115.4
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last, ac=False        |   279.3  |   2116.0
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=True   |  2589.5  |   2542.1
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format, ac=False  |  2829.9  |   2541.1
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=True       |   649.4  |   2709.0
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last, ac=False      |   646.9  |   2709.5
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=True     |  1463.2  |  17421.4
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format, ac=False    |  1474.3  |  17435.6
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=True         |   563.7  |  18582.6
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last, ac=False        |   564.1  |  18576.9

Times are in microseconds (us).
```

- PR vs Nightly
```
[--------------------------------------------- Interpolate bilinear, AA=false, cpu ----------------------------------------------]
                                                                                            |  Eager  |  Compiled  |  Just call_fn
Nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------
      Input (2, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False  |   1.2   |    5.6     |      5.5
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False  |  613.4  |   2795.4   |     2778.0

PR
1 threads: -----------------------------------------------------------------------------------------------------------------------
      Input (2, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False  |   1.2   |    2.0     |      1.9
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format, ac=False  |  613.5  |   1034.6   |     978.4



Times are in milliseconds (ms).
```


```
python perf_interp_mode_custom.py --mode=bilinear

[------------------------------------------------------------------------------------------------ Interpolate, cpu -----------------------------------------------------------------------------------------------]
                                                                                                                                                  |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)  |        1.649 (+-0.019)        |         1.072 (+-0.020)

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------- Interpolate, cuda -------------------------------------------------------------------------------------------------]
                                                                                                                                                      |  Eager (2.2.0a0+git0b5d9e3)   |  Compiled (2.2.0a0+git0b5d9e3)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)  |        98.163 (+-0.030)       |        139.391 (+-2.124)

Times are in microseconds (us).
```


## Generated code

Code:
```python
import torch

def transform(img):
    img = torch.nn.functional.interpolate(img, size=(270, 270), mode="bilinear", antialias=False)
    return img

device = "cpu"  # "cuda"
c_transform = torch.compile(transform)
x = torch.rand(2, 3, 345, 456, device=device)
output = c_transform(x)
```

- `main`, CPU generated code
```c++
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(6L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(270L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = static_cast<float>(1.2777777777777777);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = tmp10 - tmp7;
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);
                    auto tmp14 = static_cast<long>(tmp13);
                    auto tmp15 = static_cast<long>(i2);
                    auto tmp16 = static_cast<double>(tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                    auto tmp18 = tmp17 + tmp4;
                    auto tmp19 = static_cast<float>(tmp18);
                    auto tmp20 = tmp19 + tmp7;
                    auto tmp21 = static_cast<float>(1.6888888888888889);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = tmp22 - tmp7;
                    auto tmp24 = max_propagate_nan(tmp23, tmp12);
                    auto tmp25 = static_cast<long>(tmp24);
                    auto tmp26 = in_ptr0[static_cast<long>(tmp25 + (456L*tmp14) + (157320L*i0))];
                    auto tmp27 = static_cast<float>(tmp14);
                    auto tmp28 = tmp13 - tmp27;
                    auto tmp29 = static_cast<float>(1.0);
                    auto tmp30 = tmp29 - tmp28;
                    auto tmp31 = decltype(tmp26)(tmp26 * tmp30);
                    auto tmp32 = std::ceil(tmp13);
                    auto tmp33 = static_cast<float>(344.0);
                    auto tmp34 = min_propagate_nan(tmp32, tmp33);
                    auto tmp35 = static_cast<long>(tmp34);
                    auto tmp36 = in_ptr0[static_cast<long>(tmp25 + (456L*tmp35) + (157320L*i0))];
                    auto tmp37 = decltype(tmp36)(tmp36 * tmp28);
                    auto tmp38 = tmp31 + tmp37;
                    out_ptr0[static_cast<long>(i2 + (270L*i1) + (72900L*i0))] = tmp38;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(6L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(270L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = static_cast<float>(1.2777777777777777);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = tmp10 - tmp7;
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);
                    auto tmp14 = static_cast<long>(tmp13);
                    auto tmp15 = static_cast<long>(i2);
                    auto tmp16 = static_cast<double>(tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                    auto tmp18 = tmp17 + tmp4;
                    auto tmp19 = static_cast<float>(tmp18);
                    auto tmp20 = tmp19 + tmp7;
                    auto tmp21 = static_cast<float>(1.6888888888888889);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = tmp22 - tmp7;
                    auto tmp24 = max_propagate_nan(tmp23, tmp12);
                    auto tmp25 = std::ceil(tmp24);
                    auto tmp26 = static_cast<float>(455.0);
                    auto tmp27 = min_propagate_nan(tmp25, tmp26);
                    auto tmp28 = static_cast<long>(tmp27);
                    auto tmp29 = in_ptr0[static_cast<long>(tmp28 + (456L*tmp14) + (157320L*i0))];
                    auto tmp30 = static_cast<float>(tmp14);
                    auto tmp31 = tmp13 - tmp30;
                    auto tmp32 = static_cast<float>(1.0);
                    auto tmp33 = tmp32 - tmp31;
                    auto tmp34 = decltype(tmp29)(tmp29 * tmp33);
                    auto tmp35 = std::ceil(tmp13);
                    auto tmp36 = static_cast<float>(344.0);
                    auto tmp37 = min_propagate_nan(tmp35, tmp36);
                    auto tmp38 = static_cast<long>(tmp37);
                    auto tmp39 = in_ptr0[static_cast<long>(tmp28 + (456L*tmp38) + (157320L*i0))];
                    auto tmp40 = decltype(tmp39)(tmp39 * tmp31);
                    auto tmp41 = tmp34 + tmp40;
                    out_ptr1[static_cast<long>(i2 + (270L*i1) + (72900L*i0))] = tmp41;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(1620L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(i1 + (270L*i0))];
                auto tmp21 = out_ptr1[static_cast<long>(i1 + (270L*i0))];
                auto tmp1 = static_cast<long>(i1);
                auto tmp2 = static_cast<double>(tmp1);
                auto tmp3 = static_cast<double>(1.0);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = static_cast<double>(0.0);
                auto tmp6 = tmp4 + tmp5;
                auto tmp7 = static_cast<float>(tmp6);
                auto tmp8 = static_cast<float>(0.5);
                auto tmp9 = tmp7 + tmp8;
                auto tmp10 = static_cast<float>(1.6888888888888889);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp12 = tmp11 - tmp8;
                auto tmp13 = static_cast<float>(0.0);
                auto tmp14 = max_propagate_nan(tmp12, tmp13);
                auto tmp15 = static_cast<long>(tmp14);
                auto tmp16 = static_cast<float>(tmp15);
                auto tmp17 = tmp14 - tmp16;
                auto tmp18 = static_cast<float>(1.0);
                auto tmp19 = tmp18 - tmp17;
                auto tmp20 = decltype(tmp0)(tmp0 * tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp17);
                auto tmp23 = tmp20 + tmp22;
                in_out_ptr0[static_cast<long>(i1 + (270L*i0))] = tmp23;
            }
        }
    }
}
```


- PR, CPU generated code
```c++
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(6L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(270L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = static_cast<float>(1.2777777777777777);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = tmp10 - tmp7;
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);
                    auto tmp14 = static_cast<long>(tmp13);
                    auto tmp15 = static_cast<long>(344);
                    auto tmp16 = min_propagate_nan(tmp14, tmp15);
                    auto tmp17 = static_cast<long>(1);
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = min_propagate_nan(tmp18, tmp15);
                    auto tmp20 = static_cast<long>(i2);
                    auto tmp21 = static_cast<double>(tmp20);
                    auto tmp22 = decltype(tmp21)(tmp21 * tmp2);
                    auto tmp23 = tmp22 + tmp4;
                    auto tmp24 = static_cast<float>(tmp23);
                    auto tmp25 = tmp24 + tmp7;
                    auto tmp26 = static_cast<float>(1.6888888888888889);
                    auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                    auto tmp28 = tmp27 - tmp7;
                    auto tmp29 = max_propagate_nan(tmp28, tmp12);
                    auto tmp30 = static_cast<long>(tmp29);
                    auto tmp31 = static_cast<long>(455);
                    auto tmp32 = min_propagate_nan(tmp30, tmp31);
                    auto tmp33 = tmp32 + tmp17;
                    auto tmp34 = min_propagate_nan(tmp33, tmp31);
                    auto tmp35 = in_ptr0[static_cast<long>(tmp34 + (456L*tmp19) + (157320L*i0))];
                    auto tmp36 = in_ptr0[static_cast<long>(tmp32 + (456L*tmp19) + (157320L*i0))];
                    auto tmp37 = tmp35 - tmp36;
                    auto tmp38 = static_cast<float>(tmp32);
                    auto tmp39 = tmp29 - tmp38;
                    auto tmp40 = max_propagate_nan(tmp39, tmp12);
                    auto tmp41 = static_cast<float>(1.0);
                    auto tmp42 = min_propagate_nan(tmp40, tmp41);
                    auto tmp43 = decltype(tmp37)(tmp37 * tmp42);
                    auto tmp44 = tmp36 + tmp43;
                    auto tmp45 = in_ptr0[static_cast<long>(tmp34 + (456L*tmp16) + (157320L*i0))];
                    auto tmp46 = in_ptr0[static_cast<long>(tmp32 + (456L*tmp16) + (157320L*i0))];
                    auto tmp47 = tmp45 - tmp46;
                    auto tmp48 = decltype(tmp47)(tmp47 * tmp42);
                    auto tmp49 = tmp46 + tmp48;
                    auto tmp50 = tmp44 - tmp49;
                    auto tmp51 = static_cast<float>(tmp16);
                    auto tmp52 = tmp13 - tmp51;
                    auto tmp53 = max_propagate_nan(tmp52, tmp12);
                    auto tmp54 = min_propagate_nan(tmp53, tmp41);
                    auto tmp55 = decltype(tmp50)(tmp50 * tmp54);
                    auto tmp56 = tmp49 + tmp55;
                    in_out_ptr0[static_cast<long>(i2 + (270L*i1) + (72900L*i0))] = tmp56;
                }
            }
        }
    }
}
```


- `main`, Triton generated code
```python
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 437400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 270) % 270
    x0 = xindex % 270
    x2 = (xindex // 72900)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 1.2777777777777777
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp4)
    tmp12 = tmp11.to(tl.int32)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15 + tmp4
    tmp17 = tmp16 + tmp6
    tmp18 = 1.6888888888888889
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19 - tmp6
    tmp21 = triton_helpers.maximum(tmp20, tmp4)
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr0 + (tmp22 + (456*tmp12) + (157320*x2)), xmask)
    tmp24 = tmp12.to(tl.float32)
    tmp25 = tmp11 - tmp24
    tmp26 = tmp2 - tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = tl.math.ceil(tmp11)
    tmp29 = 344.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp22 + (456*tmp31) + (157320*x2)), xmask)
    tmp33 = tmp32 * tmp25
    tmp34 = tmp27 + tmp33
    tmp35 = tl.math.ceil(tmp21)
    tmp36 = 455.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37.to(tl.int32)
    tmp39 = tl.load(in_ptr0 + (tmp38 + (456*tmp12) + (157320*x2)), xmask)
    tmp40 = tmp39 * tmp26
    tmp41 = tl.load(in_ptr0 + (tmp38 + (456*tmp31) + (157320*x2)), xmask)
    tmp42 = tmp41 * tmp25
    tmp43 = tmp40 + tmp42
    tmp44 = tmp22.to(tl.float32)
    tmp45 = tmp21 - tmp44
    tmp46 = tmp2 - tmp45
    tmp47 = tmp34 * tmp46
    tmp48 = tmp43 * tmp45
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (x4), tmp49, xmask)
```

- PR, Triton generated code
```python
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 437400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 270) % 270
    x0 = xindex % 270
    x2 = (xindex // 72900)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 1.2777777777777777
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp4)
    tmp12 = tmp11.to(tl.int32)
    tmp13 = 344
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = 1
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.minimum(tmp16, tmp13)
    tmp18 = x0
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20 + tmp4
    tmp22 = tmp21 + tmp6
    tmp23 = 1.6888888888888889
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24 - tmp6
    tmp26 = triton_helpers.maximum(tmp25, tmp4)
    tmp27 = tmp26.to(tl.int32)
    tmp28 = 455
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp29 + tmp15
    tmp31 = triton_helpers.minimum(tmp30, tmp28)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (456*tmp17) + (157320*x2)), xmask)
    tmp33 = tl.load(in_ptr0 + (tmp29 + (456*tmp17) + (157320*x2)), xmask)
    tmp34 = tmp32 - tmp33
    tmp35 = tmp29.to(tl.float32)
    tmp36 = tmp26 - tmp35
    tmp37 = triton_helpers.maximum(tmp36, tmp4)
    tmp38 = triton_helpers.minimum(tmp37, tmp2)
    tmp39 = tmp34 * tmp38
    tmp40 = tmp33 + tmp39
    tmp41 = tl.load(in_ptr0 + (tmp31 + (456*tmp14) + (157320*x2)), xmask)
    tmp42 = tl.load(in_ptr0 + (tmp29 + (456*tmp14) + (157320*x2)), xmask)
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp38
    tmp45 = tmp42 + tmp44
    tmp46 = tmp40 - tmp45
    tmp47 = tmp14.to(tl.float32)
    tmp48 = tmp11 - tmp47
    tmp49 = triton_helpers.maximum(tmp48, tmp4)
    tmp50 = triton_helpers.minimum(tmp49, tmp2)
    tmp51 = tmp46 * tmp50
    tmp52 = tmp45 + tmp51
    tl.store(in_out_ptr0 + (x4), tmp52, xmask)
```



## Performance benchmark

- `main`
```
Torch version: 2.1.0a0+git37359c3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[------------------------- Interpolate bilinear, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  1459.5  |   1223.4
      Input (3, 345, 456), torch.float32, torch.channels_last      |   989.5  |   1364.0

Times are in microseconds (us).

[------------------------ Interpolate bilinear, AA=false, cuda ------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   10.6  |    39.4
      Input (3, 345, 456), torch.float32, torch.channels_last      |   10.7  |    42.6

Times are in microseconds (us).
```

- PR
```
Torch version: 2.1.0a0+git37359c3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[------------------------- Interpolate bilinear, AA=false, cpu -------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  1432.1  |   942.3
      Input (3, 345, 456), torch.float32, torch.channels_last      |   987.7  |   946.6

Times are in microseconds (us).


[------------------------ Interpolate bilinear, AA=false, cuda ------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   11.9  |    43.1
      Input (3, 345, 456), torch.float32, torch.channels_last      |   11.2  |    43.2

Times are in microseconds (us).
```



- Using lerp
```
[--------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   1.7   |     1.7
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   1.1   |     1.5
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   7.1   |     6.4
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   4.5   |    11.7

[--------------------------------- Interpolate bilinear, AA=false, cuda --------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   12.8  |   106.5
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   13.0  |   107.6
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   12.8  |   115.9
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   12.8  |   116.9

Times are in microseconds (us).
```

- Without lerp
```
[--------------------------------- Interpolate bilinear, AA=false, cpu ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   1.7   |     1.6
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   1.1   |     1.5
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   7.0   |     6.3
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   4.3   |    10.7

[--------------------------------- Interpolate bilinear, AA=false, cuda --------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   13.2  |    82.9
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   13.4  |    83.4
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   13.0  |    91.9
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   13.4  |    91.8

Times are in microseconds (us).
```

## Test

```
pytest -vvv test/test_decomp.py -k "interpolate_bil or upsample_bil"
pytest -vvv test/inductor/test_torchinductor_opinfo.py -k "interp or upsampl"
pytest -vvv test/functorch/test_aotdispatch.py -k "interp or upsampl"
pytest -vvv test/functorch/test_ops.py -k "interp or upsampl"
pytest -vvv test/test_meta.py -k "interp or upsampl"
pytest -vvv test/test_proxy_tensor.py -k "interp or upsampl"
pytest -vvv test/test_decomp.py::HasDecompTest::test_has_decomposition
pytest -vvv test/functorch/test_vmap.py -k "interp or upsampl"


pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32
pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32

pytest -vvv test/functorch/test_vmap.py::TestVmapOperatorsOpInfoCUDA::test_op_has_batch_rule_nn_functional_interpolate_bicubic_cuda_float32
```


```
FAILED [1.1219s] test/test_decomp.py::TestDecompCPU::test_comprehensive_nn_functional_interpolate_bilinear_cpu_uint8

FAILED [0.1714s] test/test_decomp.py::TestDecompCPU::test_comprehensive_nn_functional_upsample_bilinear_cpu_uint8

FAILED [1.3664s] test/test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_interpolate_bilinear_cuda_bfloat16
```


## MPS failing tests

```
Job: macos-12-py3-arm64-mps / test (default, 1, 1)

Mismatched elements: 206447 / 437400 (47.2%)
Greatest absolute difference: 0.0009098052978515625 at index (1, 1, 255, 228) (up to 1e-05 allowed)
Greatest relative difference: 5.863010883331299 at index (1, 1, 72, 82) (up to 1.3e-06 allowed)

To execute this test, run the following from the base repo dir:
     python test/test_mps.py -k test_output_grad_match_nn_functional_interpolate_bilinear_cpu_float32


Mismatched elements: 219120 / 437400 (50.1%)
Greatest absolute difference: 1 at index (0, 0, 0, 4)
Greatest relative difference: 1.0 at index (0, 0, 203, 0)

To execute this test, run the following from the base repo dir:
     python test/test_mps.py -k test_output_match_nn_functional_upsample_bilinear_cpu_uint8



Mismatched elements: 52 / 437400 (0.0%)
Greatest absolute difference: 2 at index (0, 0, 25, 202) (up to 1 allowed)
Greatest relative difference: 0.0714285746216774 at index (1, 1, 168, 67) (up to 0 allowed)

To execute this test, run the following from the base repo dir:
     python test/test_mps.py -k test_output_match_nn_functional_interpolate_bilinear_cpu_uint8



Job: macos-13-py3-arm64-mps / test (default, 1, 1)

Mismatched elements: 52 / 437400 (0.0%)
Greatest absolute difference: 2 at index (0, 0, 25, 202) (up to 1 allowed)
Greatest relative difference: 0.0714285746216774 at index (1, 1, 168, 67) (up to 0 allowed)

To execute this test, run the following from the base repo dir:
     python test/test_mps.py -k test_output_match_nn_functional_interpolate_bilinear_cpu_uint8
```
