Description:

- 20231127-105525-upsample-nearest-PR
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

- 20231127-102511-upsample-nearest-Nightly
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


[------------------------------------------------------------------------------------------------------------------------------------------------ Interpolate, cpu -----------------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                  |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git9fcf1f9) Nightly  |  speed-up PR vs Nightly  |  Eager (2.2.0a0+git9fcf1f9) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |        621.163 (+-20.778)       |        1018.567 (+-34.227)         |           1206.860 (+-48.705)           |     1.185 (+-0.000)      |          607.580 (+-14.296)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |       2626.677 (+-97.980)       |         870.997 (+-28.067)         |            421.399 (+-15.743)           |     0.484 (+-0.000)      |         2862.472 (+-114.209)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       1488.885 (+-53.598)       |        4112.969 (+-181.760)        |           2464.728 (+-102.621)          |     0.599 (+-0.000)      |         1477.341 (+-40.279)

      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |        618.158 (+-22.540)       |        1088.801 (+-39.059)         |           2745.997 (+-101.949)          |     2.522 (+-0.000)      |          614.041 (+-19.291)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |       3434.688 (+-104.843)      |        1014.350 (+-37.017)         |           2539.435 (+-82.041)           |     2.504 (+-0.000)      |         2898.717 (+-142.784)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       1492.895 (+-62.081)       |        6619.341 (+-183.201)        |          22608.583 (+-719.916)          |     3.416 (+-0.000)      |         1471.411 (+-63.379)

      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |        280.997 (+-10.102)       |        1110.081 (+-32.484)         |            402.772 (+-13.182)           |     0.363 (+-0.000)      |          277.887 (+-8.256)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |        661.185 (+-23.689)       |        1033.600 (+-39.900)         |            379.413 (+-14.214)           |     0.367 (+-0.000)      |          650.561 (+-23.299)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |        567.445 (+-15.264)       |        5038.760 (+-157.398)        |           2207.147 (+-62.151)           |     0.438 (+-0.000)      |          565.148 (+-20.155)

      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |        280.388 (+-7.777)        |        1239.378 (+-53.034)         |           2940.184 (+-104.553)          |     2.372 (+-0.000)      |          280.397 (+-10.478)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |        658.109 (+-23.216)       |        1144.156 (+-35.842)         |           2706.679 (+-78.961)           |     2.366 (+-0.000)      |          662.249 (+-26.340)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |        568.326 (+-12.951)       |        7553.262 (+-294.715)        |          18626.812 (+-622.836)          |     2.466 (+-0.000)      |          571.578 (+-17.331)



      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       1589.841 (+-66.880)       |         813.405 (+-29.322)         |           1704.639 (+-55.427)           |     2.096 (+-0.000)      |         1596.030 (+-71.278)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |       4483.042 (+-105.695)      |         773.264 (+-36.603)         |           1659.660 (+-92.973)           |     2.146 (+-0.000)      |         4537.972 (+-156.981)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |       8268.333 (+-407.423)      |        3622.925 (+-118.475)        |          10477.658 (+-356.275)          |     2.892 (+-0.000)      |         8238.306 (+-296.254)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       1666.708 (+-59.962)       |         858.197 (+-26.878)         |           2652.062 (+-95.713)           |     3.090 (+-0.000)      |         1675.025 (+-57.954)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |       4580.614 (+-131.318)      |         812.441 (+-44.388)         |           2523.818 (+-117.798)          |     3.106 (+-0.000)      |         4598.754 (+-188.300)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |       8761.556 (+-317.827)      |        5075.582 (+-201.374)        |          16484.592 (+-546.955)          |     3.248 (+-0.000)      |         8824.734 (+-327.096)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |       1000.352 (+-34.707)       |        1027.102 (+-41.094)         |           1954.359 (+-86.728)           |     1.903 (+-0.000)      |         1002.925 (+-34.120)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |        923.474 (+-29.759)       |         983.726 (+-46.919)         |           2051.349 (+-121.731)          |     2.085 (+-0.000)      |          925.094 (+-29.004)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |       6359.233 (+-280.292)      |        4615.963 (+-218.098)        |          12281.204 (+-402.379)          |     2.661 (+-0.000)      |         6380.505 (+-178.647)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       1082.276 (+-38.363)       |        1055.561 (+-32.162)         |           2940.527 (+-105.388)          |     2.786 (+-0.000)      |         1084.082 (+-32.512)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |        999.049 (+-38.717)       |        1025.145 (+-46.369)         |           2841.317 (+-98.428)           |     2.772 (+-0.000)      |         1002.737 (+-41.365)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |       6917.582 (+-220.712)      |        6560.583 (+-388.675)        |          18595.083 (+-622.834)          |     2.834 (+-0.000)      |         6936.562 (+-266.225)



      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |       2381.551 (+-91.125)       |        3527.679 (+-129.768)        |           1595.700 (+-83.141)           |     0.452 (+-0.000)      |         2338.212 (+-90.898)
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |      10231.341 (+-364.954)      |        3476.134 (+-136.743)        |           1488.576 (+-62.251)           |     0.428 (+-0.000)      |        10232.945 (+-426.766)
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       5877.585 (+-211.677)      |       17196.536 (+-649.653)        |          10306.293 (+-382.213)          |     0.599 (+-0.000)      |         5777.394 (+-161.244)

      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |       2394.148 (+-82.276)       |        4160.593 (+-134.747)        |          10982.681 (+-357.571)          |     2.640 (+-0.000)      |         2351.181 (+-80.755)
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |      10328.772 (+-199.041)      |        4071.389 (+-178.694)        |          10709.686 (+-463.999)          |     2.630 (+-0.000)      |        10236.659 (+-303.688)
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       5840.145 (+-199.414)      |       27117.640 (+-800.862)        |          70215.095 (+-1936.827)         |     2.589 (+-0.000)      |         5775.434 (+-197.634)

      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |       1025.687 (+-42.874)       |        4221.793 (+-119.500)        |           1550.791 (+-65.064)           |     0.367 (+-0.000)      |         1021.655 (+-23.268)
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |       2911.186 (+-121.607)      |        4001.353 (+-153.769)        |           1444.523 (+-45.931)           |     0.361 (+-0.000)      |         2886.135 (+-105.449)
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |       2137.464 (+-85.569)       |       20105.699 (+-587.577)        |           9497.384 (+-387.722)          |     0.472 (+-0.000)      |         2110.963 (+-82.454)

      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |       1027.979 (+-27.730)       |        4756.465 (+-179.233)        |          12710.301 (+-363.026)          |     2.672 (+-0.000)      |         1020.801 (+-42.751)
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |       2890.935 (+-117.945)      |        4456.305 (+-178.267)        |          11826.128 (+-449.795)          |     2.654 (+-0.000)      |         2906.826 (+-112.624)
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |       2128.794 (+-62.481)       |       29869.686 (+-931.007)        |          80368.797 (+-2661.360)         |     2.691 (+-0.000)      |         2129.526 (+-63.025)



      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       6885.095 (+-310.462)      |        3079.039 (+-125.886)        |           7505.370 (+-315.638)          |     2.438 (+-0.000)      |         6893.997 (+-304.085)
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |      46494.418 (+-1317.236)     |        3442.323 (+-98.632)         |           8374.186 (+-253.425)          |     2.433 (+-0.000)      |        46450.924 (+-1303.529)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |      34484.676 (+-1158.333)     |       14371.033 (+-600.304)        |          41025.986 (+-1453.034)         |     2.855 (+-0.000)      |        34646.095 (+-1117.621)
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       7205.120 (+-299.519)      |        3450.204 (+-133.120)        |          10980.495 (+-426.576)          |     3.183 (+-0.000)      |         7216.354 (+-188.214)
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |      18345.160 (+-665.723)      |        3632.861 (+-109.751)        |          11098.904 (+-272.663)          |     3.055 (+-0.000)      |        46899.210 (+-1572.034)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |      36734.445 (+-1118.982)     |       20056.642 (+-622.401)        |          63776.093 (+-2414.481)         |     3.180 (+-0.000)      |        36821.510 (+-1190.497)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |       4032.140 (+-126.717)      |        4201.109 (+-181.064)        |           9116.638 (+-278.952)          |     2.170 (+-0.000)      |         4068.240 (+-143.099)
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |       3945.308 (+-133.246)      |        5017.782 (+-123.586)        |          11226.995 (+-372.639)          |     2.237 (+-0.000)      |         3953.694 (+-130.153)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |      25327.239 (+-660.285)      |       19824.602 (+-660.530)        |          53264.830 (+-2083.338)         |     2.687 (+-0.000)      |        25402.838 (+-749.554)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       4363.928 (+-167.479)      |        4555.112 (+-172.701)        |          12600.321 (+-440.424)          |     2.766 (+-0.000)      |         4362.200 (+-153.527)
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |       4247.567 (+-125.985)      |        5412.192 (+-115.925)        |          13806.700 (+-471.931)          |     2.551 (+-0.000)      |         4229.363 (+-94.781)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |      27516.200 (+-925.657)      |       27920.592 (+-856.020)        |          77145.788 (+-2212.289)         |     2.763 (+-0.000)      |        27621.721 (+-926.195)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------------------------------------------------------ Interpolate, cuda ------------------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                    |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git9fcf1f9) Nightly  |  speed-up PR vs Nightly  |  Eager (2.2.0a0+git9fcf1f9) Nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |         98.241 (+-0.022)        |         145.255 (+-4.630)          |             97.929 (+-1.843)            |     0.674 (+-0.000)      |           98.343 (+-0.024)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        122.829 (+-0.024)        |         146.415 (+-4.441)          |            190.124 (+-0.050)            |     1.299 (+-0.000)      |          122.167 (+-0.036)
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |         97.970 (+-0.024)        |         145.552 (+-3.464)          |             97.556 (+-2.176)            |     0.670 (+-0.000)      |           98.485 (+-0.022)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        122.854 (+-0.017)        |         146.803 (+-4.476)          |            161.156 (+-0.055)            |     1.098 (+-0.000)      |          121.900 (+-0.018)

      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |         97.893 (+-0.025)        |         145.896 (+-3.792)          |             95.373 (+-4.009)            |     0.654 (+-0.000)      |           97.252 (+-0.025)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        105.501 (+-0.125)        |         146.120 (+-4.675)          |            174.239 (+-0.089)            |     1.192 (+-0.000)      |          105.672 (+-0.138)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |         97.997 (+-0.027)        |         146.352 (+-4.428)          |             96.248 (+-2.816)            |     0.658 (+-0.000)      |           97.317 (+-0.026)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        105.912 (+-0.214)        |         146.762 (+-4.932)          |            162.812 (+-0.073)            |     1.109 (+-0.000)      |          105.884 (+-0.260)

      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |        384.605 (+-0.017)        |         385.824 (+-0.029)          |            383.070 (+-0.015)            |     0.993 (+-0.000)      |          384.799 (+-0.011)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        446.981 (+-0.034)        |         398.938 (+-0.110)          |            756.201 (+-0.767)            |     1.896 (+-0.000)      |          447.329 (+-0.022)
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |        384.590 (+-0.024)        |         385.360 (+-0.039)          |            382.705 (+-0.017)            |     0.993 (+-0.000)      |          384.605 (+-0.010)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        446.868 (+-0.028)        |         398.742 (+-0.121)          |            638.223 (+-0.024)            |     1.601 (+-0.000)      |          446.901 (+-0.026)

      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |        384.047 (+-0.013)        |         561.913 (+-0.026)          |            550.517 (+-0.024)            |     0.980 (+-0.000)      |          384.181 (+-0.014)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        439.885 (+-0.429)        |         416.036 (+-1.768)          |            710.031 (+-1.400)            |     1.707 (+-0.000)      |          439.463 (+-0.262)
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |        384.073 (+-0.012)        |         564.323 (+-0.052)          |            553.512 (+-0.036)            |     0.981 (+-0.000)      |          384.156 (+-0.022)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        439.802 (+-0.026)        |         422.735 (+-1.677)          |            659.088 (+-0.819)            |     1.559 (+-0.000)      |          439.437 (+-0.027)

Times are in microseconds (us).
