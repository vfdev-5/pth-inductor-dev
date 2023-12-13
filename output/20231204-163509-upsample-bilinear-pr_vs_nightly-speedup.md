Description:

- 20231204-154746-upsample-nearest-PR
Torch version: 2.2.0a0+git0b5d9e3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.2.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+6e4932cda8

- 20231204-161440-upsample-nearest-Nightly
Torch version: 2.2.0a0+git9fcf1f9
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_89,code=sm_89
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.2.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+e6216047b8


[--------------------------------------------------------------------------------------------------------------------------------------------------------- Interpolate, cpu --------------------------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                                    |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git9fcf1f9) Nightly  |  speed-up PR vs Nightly  |  Eager (2.2.0a0+git9fcf1f9) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)      |        600.930 (+-5.727)        |        1688.647 (+-18.556)         |           1823.272 (+-53.709)           |     1.080 (+-0.000)      |          596.928 (+-5.818)         
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)          |        271.775 (+-2.468)        |        1710.734 (+-19.502)         |           2079.872 (+-22.611)           |     1.216 (+-0.000)      |          270.617 (+-2.973)         
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (256, 256)     |       1562.806 (+-18.960)       |        1210.417 (+-16.094)         |           1513.180 (+-19.226)           |     1.250 (+-0.000)      |         1565.835 (+-19.600)        
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)    |       1654.919 (+-23.918)       |        1316.265 (+-18.994)         |           2284.703 (+-32.394)           |     1.736 (+-0.000)      |          1658.480 (+-4.524)        
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (256, 256)         |        988.212 (+-13.855)       |        1262.167 (+-13.253)         |           1772.218 (+-22.834)           |     1.404 (+-0.000)      |          988.501 (+-13.112)        
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)        |       1075.613 (+-18.022)       |        1379.925 (+-13.903)         |           2569.187 (+-27.841)           |     1.862 (+-0.000)      |         1076.673 (+-13.870)        
      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)      |       2308.167 (+-33.868)       |        6611.365 (+-56.093)         |           7513.733 (+-126.097)          |     1.136 (+-0.000)      |         2285.215 (+-36.291)        
      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)          |        988.647 (+-14.488)       |        6960.652 (+-99.860)         |          10916.578 (+-194.577)          |     1.568 (+-0.000)      |          997.184 (+-12.480)        
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (256, 256)     |       6719.348 (+-217.282)      |        4652.211 (+-45.227)         |           6708.900 (+-120.480)          |     1.442 (+-0.000)      |         6757.597 (+-95.721)        
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)    |       7107.457 (+-110.361)      |        5323.247 (+-48.542)         |           9896.554 (+-255.446)          |     1.859 (+-0.000)      |         7061.767 (+-233.692)       
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (256, 256)         |       3968.517 (+-36.341)       |        5507.683 (+-43.747)         |           8117.150 (+-124.035)          |     1.474 (+-0.000)      |         3951.786 (+-43.609)        
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (256, 256)        |       4320.933 (+-25.906)       |        5456.215 (+-52.671)         |          11464.621 (+-147.097)          |     2.101 (+-0.000)      |         4309.560 (+-27.423)        
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)    |       2528.970 (+-29.124)       |        1832.809 (+-19.301)         |           2498.864 (+-29.544)           |     1.363 (+-0.000)      |         2539.851 (+-37.096)        
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)        |        631.739 (+-9.570)        |        1689.392 (+-18.695)         |           2659.343 (+-30.958)           |     1.574 (+-0.000)      |          631.428 (+-5.454)         
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (200, 300)   |       4432.819 (+-33.288)       |        1015.301 (+-22.837)         |           1607.306 (+-35.504)           |     1.583 (+-0.000)      |         4434.428 (+-43.498)        
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)  |       4511.564 (+-29.549)       |        1305.788 (+-19.721)         |           2435.790 (+-34.777)           |     1.865 (+-0.000)      |         4508.044 (+-30.940)        
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (200, 300)       |        910.415 (+-13.406)       |        1052.825 (+-19.124)         |           1952.845 (+-30.261)           |     1.855 (+-0.000)      |          909.118 (+-12.143)        
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)      |        985.584 (+-16.395)       |        1371.791 (+-18.229)         |           2731.866 (+-147.572)          |     1.991 (+-0.000)      |          989.971 (+-18.520)        
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)    |      10014.096 (+-211.019)      |        6403.334 (+-110.211)        |          10457.186 (+-145.870)          |     1.633 (+-0.000)      |        10005.541 (+-191.402)       
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)        |       2722.986 (+-31.608)       |        6746.002 (+-115.027)        |          11598.250 (+-158.962)          |     1.719 (+-0.000)      |         2782.645 (+-30.956)        
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (200, 300)   |      42970.713 (+-325.355)      |        4311.711 (+-35.670)         |           8077.226 (+-93.328)           |     1.873 (+-0.000)      |        42395.809 (+-286.415)       
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)  |      43243.100 (+-301.054)      |        5506.786 (+-40.888)         |          10724.618 (+-130.569)          |     1.948 (+-0.000)      |        42689.356 (+-307.561)       
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (200, 300)       |       3886.883 (+-83.797)       |        5023.875 (+-36.435)         |          10949.126 (+-115.774)          |     2.179 (+-0.000)      |         3868.654 (+-35.372)        
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (200, 300)      |       4157.685 (+-37.602)       |        6238.797 (+-79.831)         |          13391.204 (+-148.667)          |     2.146 (+-0.000)      |         4181.643 (+-34.084)        
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)      |       1473.258 (+-19.930)       |       10724.755 (+-130.449)        |          17162.168 (+-239.507)          |     1.600 (+-0.000)      |         1462.272 (+-24.479)        
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)          |        549.418 (+-4.553)        |       11210.479 (+-132.929)        |          18300.540 (+-202.650)          |     1.632 (+-0.000)      |          550.518 (+-4.184)         
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (600, 700)     |       8003.415 (+-135.592)      |        4837.020 (+-40.813)         |          10289.275 (+-133.659)          |     2.127 (+-0.000)      |         7941.016 (+-148.347)       
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)    |       8543.836 (+-164.685)      |        8239.059 (+-128.748)        |          16191.261 (+-188.822)          |     1.965 (+-0.000)      |         8498.577 (+-152.123)       
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (600, 700)         |       6271.643 (+-33.740)       |        4844.406 (+-42.042)         |          12079.816 (+-177.087)          |     2.494 (+-0.000)      |         6260.411 (+-47.081)        
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)        |       6817.547 (+-120.231)      |        8631.854 (+-124.947)        |          18290.525 (+-204.957)          |     2.119 (+-0.000)      |         6808.965 (+-109.987)       
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)      |       5744.059 (+-45.820)       |       42701.977 (+-398.044)        |          68946.146 (+-926.974)          |     1.615 (+-0.000)      |         5730.670 (+-42.898)        
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)          |       2044.269 (+-29.394)       |       44789.858 (+-377.806)        |          79103.804 (+-1059.116)         |     1.766 (+-0.000)      |         2046.521 (+-22.311)        
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (600, 700)     |      34075.011 (+-293.423)      |       19283.441 (+-256.727)        |          40479.417 (+-301.264)          |     2.099 (+-0.000)      |        34098.207 (+-768.475)       
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)    |      36337.585 (+-266.445)      |       32978.911 (+-327.800)        |          62777.811 (+-874.523)          |     1.904 (+-0.000)      |        36297.665 (+-265.161)       
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (600, 700)         |      25053.300 (+-244.243)      |       19350.561 (+-239.968)        |          52454.616 (+-285.841)          |     2.711 (+-0.000)      |        25052.150 (+-244.333)       
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (600, 700)        |      27287.270 (+-263.069)      |       35466.996 (+-308.674)        |          75949.054 (+-1598.156)         |     2.141 (+-0.000)      |        27222.554 (+-261.035)       

Times are in microseconds (us).

[--------------------------------------------------------------------------------------------------------------------------------------------------------- Interpolate, cuda ---------------------------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                                      |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git9fcf1f9) Nightly  |  speed-up PR vs Nightly  |  Eager (2.2.0a0+git9fcf1f9) Nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (1234, 1345)   |         98.275 (+-0.037)        |          97.474 (+-0.897)          |             97.866 (+-0.011)            |     1.004 (+-0.000)      |           98.254 (+-0.027)         
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)  |         97.049 (+-0.016)        |          96.621 (+-0.376)          |             96.693 (+-0.018)            |     1.001 (+-0.000)      |           97.038 (+-0.019)         
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (1234, 1345)       |         97.941 (+-0.017)        |          94.073 (+-0.862)          |             91.978 (+-0.938)            |     0.978 (+-0.000)      |           97.950 (+-0.023)         
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)      |         97.233 (+-0.017)        |          96.442 (+-0.513)          |             96.163 (+-0.591)            |     0.997 (+-0.000)      |           97.232 (+-0.020)         
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (1234, 1345)   |        384.720 (+-0.022)        |         382.909 (+-0.053)          |            382.871 (+-0.048)            |     1.000 (+-0.000)      |          384.707 (+-0.018)         
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)  |        384.409 (+-0.025)        |         382.412 (+-0.059)          |            382.530 (+-0.059)            |     1.000 (+-0.000)      |          384.390 (+-0.019)         
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (1234, 1345)       |        384.066 (+-0.025)        |         542.655 (+-0.065)          |            540.327 (+-0.067)            |     0.996 (+-0.000)      |          384.074 (+-0.020)         
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (1234, 1345)      |        384.094 (+-0.024)        |         548.691 (+-0.066)          |            553.311 (+-0.061)            |     1.008 (+-0.000)      |          384.083 (+-0.026)         
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (2345, 2456)   |        121.956 (+-0.016)        |         118.421 (+-1.001)          |            190.451 (+-0.036)            |     1.608 (+-0.000)      |          121.857 (+-0.020)         
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (2345, 2456)  |        121.942 (+-0.020)        |         119.290 (+-0.943)          |            161.366 (+-0.041)            |     1.353 (+-0.000)      |          121.906 (+-0.023)         
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (2345, 2456)       |        106.389 (+-0.365)        |         120.931 (+-0.396)          |            190.689 (+-0.041)            |     1.577 (+-0.000)      |          106.163 (+-0.278)         
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (2345, 2456)      |        106.620 (+-0.167)        |         121.848 (+-0.479)          |            162.951 (+-0.038)            |     1.337 (+-0.000)      |          106.619 (+-0.408)         
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (2345, 2456)   |        446.737 (+-0.036)        |         464.485 (+-1.700)          |            756.550 (+-0.340)            |     1.629 (+-0.000)      |          446.861 (+-0.028)         
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, antialias: False, osize: (2345, 2456)  |        446.542 (+-0.022)        |         471.969 (+-1.048)          |            639.564 (+-0.057)            |     1.355 (+-0.000)      |          446.678 (+-0.033)         
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, antialias: False, osize: (2345, 2456)       |        439.489 (+-0.035)        |         570.715 (+-1.702)          |            712.564 (+-3.255)            |     1.249 (+-0.000)      |          439.546 (+-0.039)         
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, antialias: False, osize: (2345, 2456)      |        439.457 (+-0.043)        |         573.842 (+-1.972)          |            660.118 (+-0.951)            |     1.150 (+-0.000)      |          439.468 (+-0.046)         

Times are in microseconds (us).