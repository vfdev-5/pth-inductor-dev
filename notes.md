# PyTorch Inductor dev notes


## Grid Sampler PR perfs


```
python -u perf_affine_grid_sampler.py
```

- Nightly
```
[------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------]
                                                                                                          |  Eager  |  Compiled
1 threads: --------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |    8.2  |    13.2
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |    8.4  |    15.6
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |    5.5  |     6.3
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |    4.4  |     6.5
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |   25.8  |    64.3
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |   26.2  |    73.1
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |    8.5  |    13.0
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |    8.4  |    15.5
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |    4.8  |     6.3
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |    4.4  |     6.6
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |   26.0  |    65.6
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |   26.5  |    72.2

Times are in milliseconds (ms).

[------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------]
                                                                                                          |  Eager  |  Compiled
1 threads: --------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |   86.1  |    137.4
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |   85.8  |    137.3
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |   76.5  |    142.8
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |   76.7  |    142.6
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |   92.4  |   1792.8
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |   92.2  |   1455.5
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |  112.1  |    221.9
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |  110.9  |    222.5
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |  111.7  |    178.4
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |  110.5  |    178.4
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |  110.7  |   1799.4
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |  109.6  |   1463.1

Times are in microseconds (us).
```

- PR
```
[------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------]
                                                                                                          |  Eager  |  Compiled
1 threads: --------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |    8.3  |    16.9
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |    8.4  |    12.6
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |    4.8  |     5.7
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |    4.5  |     6.0
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |   26.3  |    79.2
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |   26.2  |    83.8
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |    8.8  |    17.3
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |    8.9  |    12.2
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |    4.9  |     5.8
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |    4.3  |     6.1
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |   26.3  |    76.9
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |   26.4  |    84.5

Times are in milliseconds (ms).

[------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------]
                                                                                                          |  Eager  |  Compiled
1 threads: --------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |   87.7  |   126.2
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |   86.5  |   123.4
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |   77.6  |   111.1
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |   76.9  |   110.8
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |   92.4  |   336.8
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |   92.2  |   340.8
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |  112.6  |   111.2
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |  110.8  |   110.5
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |  112.4  |   111.2
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |  110.4  |   110.9
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |  112.5  |   335.6
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |  111.3  |   337.7

Times are in microseconds (us).
```

