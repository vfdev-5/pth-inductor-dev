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



[---------------------------------------------------------------------------------------------------------------------------------- Interpolate, cpu ----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                  |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Eager (2.2.0a0+git9fcf1f9) Nightly  |  Compiled (2.2.0a0+git9fcf1f9) Nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |        621.163 (+-20.778)       |        1018.567 (+-34.227)         |          607.580 (+-14.296)          |           1206.860 (+-48.705)         
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |       2626.677 (+-97.980)       |         870.997 (+-28.067)         |         2862.472 (+-114.209)         |            421.399 (+-15.743)         
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       1488.885 (+-53.598)       |        4112.969 (+-181.760)        |         1477.341 (+-40.279)          |           2464.728 (+-102.621)        
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |        618.158 (+-22.540)       |        1088.801 (+-39.059)         |          614.041 (+-19.291)          |           2745.997 (+-101.949)        
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |       3434.688 (+-104.843)      |        1014.350 (+-37.017)         |         2898.717 (+-142.784)         |           2539.435 (+-82.041)         
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       1492.895 (+-62.081)       |        6619.341 (+-183.201)        |         1471.411 (+-63.379)          |          22608.583 (+-719.916)        
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |        280.997 (+-10.102)       |        1110.081 (+-32.484)         |          277.887 (+-8.256)           |            402.772 (+-13.182)         
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |        661.185 (+-23.689)       |        1033.600 (+-39.900)         |          650.561 (+-23.299)          |            379.413 (+-14.214)         
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |        567.445 (+-15.264)       |        5038.760 (+-157.398)        |          565.148 (+-20.155)          |           2207.147 (+-62.151)         
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |        280.388 (+-7.777)        |        1239.378 (+-53.034)         |          280.397 (+-10.478)          |           2940.184 (+-104.553)        
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |        658.109 (+-23.216)       |        1144.156 (+-35.842)         |          662.249 (+-26.340)          |           2706.679 (+-78.961)         
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |        568.326 (+-12.951)       |        7553.262 (+-294.715)        |          571.578 (+-17.331)          |          18626.812 (+-622.836)        
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       1589.841 (+-66.880)       |         813.405 (+-29.322)         |         1596.030 (+-71.278)          |           1704.639 (+-55.427)         
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |       4483.042 (+-105.695)      |         773.264 (+-36.603)         |         4537.972 (+-156.981)         |           1659.660 (+-92.973)         
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |       8268.333 (+-407.423)      |        3622.925 (+-118.475)        |         8238.306 (+-296.254)         |          10477.658 (+-356.275)        
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       1666.708 (+-59.962)       |         858.197 (+-26.878)         |         1675.025 (+-57.954)          |           2652.062 (+-95.713)         
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |       4580.614 (+-131.318)      |         812.441 (+-44.388)         |         4598.754 (+-188.300)         |           2523.818 (+-117.798)        
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |       8761.556 (+-317.827)      |        5075.582 (+-201.374)        |         8824.734 (+-327.096)         |          16484.592 (+-546.955)        
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |       1000.352 (+-34.707)       |        1027.102 (+-41.094)         |         1002.925 (+-34.120)          |           1954.359 (+-86.728)         
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |        923.474 (+-29.759)       |         983.726 (+-46.919)         |          925.094 (+-29.004)          |           2051.349 (+-121.731)        
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |       6359.233 (+-280.292)      |        4615.963 (+-218.098)        |         6380.505 (+-178.647)         |          12281.204 (+-402.379)        
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       1082.276 (+-38.363)       |        1055.561 (+-32.162)         |         1084.082 (+-32.512)          |           2940.527 (+-105.388)        
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |        999.049 (+-38.717)       |        1025.145 (+-46.369)         |         1002.737 (+-41.365)          |           2841.317 (+-98.428)         
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |       6917.582 (+-220.712)      |        6560.583 (+-388.675)        |         6936.562 (+-266.225)         |          18595.083 (+-622.834)        
      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |       2381.551 (+-91.125)       |        3527.679 (+-129.768)        |         2338.212 (+-90.898)          |           1595.700 (+-83.141)         
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |      10231.341 (+-364.954)      |        3476.134 (+-136.743)        |        10232.945 (+-426.766)         |           1488.576 (+-62.251)         
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       5877.585 (+-211.677)      |       17196.536 (+-649.653)        |         5777.394 (+-161.244)         |          10306.293 (+-382.213)        
      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |       2394.148 (+-82.276)       |        4160.593 (+-134.747)        |         2351.181 (+-80.755)          |          10982.681 (+-357.571)        
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |      10328.772 (+-199.041)      |        4071.389 (+-178.694)        |        10236.659 (+-303.688)         |          10709.686 (+-463.999)        
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       5840.145 (+-199.414)      |       27117.640 (+-800.862)        |         5775.434 (+-197.634)         |          70215.095 (+-1936.827)       
      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |       1025.687 (+-42.874)       |        4221.793 (+-119.500)        |         1021.655 (+-23.268)          |           1550.791 (+-65.064)         
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |       2911.186 (+-121.607)      |        4001.353 (+-153.769)        |         2886.135 (+-105.449)         |           1444.523 (+-45.931)         
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |       2137.464 (+-85.569)       |       20105.699 (+-587.577)        |         2110.963 (+-82.454)          |           9497.384 (+-387.722)        
      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |       1027.979 (+-27.730)       |        4756.465 (+-179.233)        |         1020.801 (+-42.751)          |          12710.301 (+-363.026)        
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |       2890.935 (+-117.945)      |        4456.305 (+-178.267)        |         2906.826 (+-112.624)         |          11826.128 (+-449.795)        
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |       2128.794 (+-62.481)       |       29869.686 (+-931.007)        |         2129.526 (+-63.025)          |          80368.797 (+-2661.360)       
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       6885.095 (+-310.462)      |        3079.039 (+-125.886)        |         6893.997 (+-304.085)         |           7505.370 (+-315.638)        
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |      46494.418 (+-1317.236)     |        3442.323 (+-98.632)         |        46450.924 (+-1303.529)        |           8374.186 (+-253.425)        
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |      34484.676 (+-1158.333)     |       14371.033 (+-600.304)        |        34646.095 (+-1117.621)        |          41025.986 (+-1453.034)       
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       7205.120 (+-299.519)      |        3450.204 (+-133.120)        |         7216.354 (+-188.214)         |          10980.495 (+-426.576)        
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |      18345.160 (+-665.723)      |        3632.861 (+-109.751)        |        46899.210 (+-1572.034)        |          11098.904 (+-272.663)        
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |      36734.445 (+-1118.982)     |       20056.642 (+-622.401)        |        36821.510 (+-1190.497)        |          63776.093 (+-2414.481)       
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |       4032.140 (+-126.717)      |        4201.109 (+-181.064)        |         4068.240 (+-143.099)         |           9116.638 (+-278.952)        
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |       3945.308 (+-133.246)      |        5017.782 (+-123.586)        |         3953.694 (+-130.153)         |          11226.995 (+-372.639)        
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |      25327.239 (+-660.285)      |       19824.602 (+-660.530)        |        25402.838 (+-749.554)         |          53264.830 (+-2083.338)       
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       4363.928 (+-167.479)      |        4555.112 (+-172.701)        |         4362.200 (+-153.527)         |          12600.321 (+-440.424)        
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |       4247.567 (+-125.985)      |        5412.192 (+-115.925)        |         4229.363 (+-94.781)          |          13806.700 (+-471.931)        
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |      27516.200 (+-925.657)      |       27920.592 (+-856.020)        |        27621.721 (+-926.195)         |          77145.788 (+-2212.289)       

Times are in microseconds (us).

[---------------------------------------------------------------------------------------------------------------------------------- Interpolate, cuda -----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                    |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Eager (2.2.0a0+git9fcf1f9) Nightly  |  Compiled (2.2.0a0+git9fcf1f9) Nightly
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |         98.241 (+-0.022)        |         145.255 (+-4.630)          |           98.343 (+-0.024)           |             97.929 (+-1.843)          
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        122.829 (+-0.024)        |         146.415 (+-4.441)          |          122.167 (+-0.036)           |            190.124 (+-0.050)          
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |         97.970 (+-0.024)        |         145.552 (+-3.464)          |           98.485 (+-0.022)           |             97.556 (+-2.176)          
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        122.854 (+-0.017)        |         146.803 (+-4.476)          |          121.900 (+-0.018)           |            161.156 (+-0.055)          
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |         97.893 (+-0.025)        |         145.896 (+-3.792)          |           97.252 (+-0.025)           |             95.373 (+-4.009)          
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        105.501 (+-0.125)        |         146.120 (+-4.675)          |          105.672 (+-0.138)           |            174.239 (+-0.089)          
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |         97.997 (+-0.027)        |         146.352 (+-4.428)          |           97.317 (+-0.026)           |             96.248 (+-2.816)          
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        105.912 (+-0.214)        |         146.762 (+-4.932)          |          105.884 (+-0.260)           |            162.812 (+-0.073)          
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |        384.605 (+-0.017)        |         385.824 (+-0.029)          |          384.799 (+-0.011)           |            383.070 (+-0.015)          
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        446.981 (+-0.034)        |         398.938 (+-0.110)          |          447.329 (+-0.022)           |            756.201 (+-0.767)          
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |        384.590 (+-0.024)        |         385.360 (+-0.039)          |          384.605 (+-0.010)           |            382.705 (+-0.017)          
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        446.868 (+-0.028)        |         398.742 (+-0.121)          |          446.901 (+-0.026)           |            638.223 (+-0.024)          
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |        384.047 (+-0.013)        |         561.913 (+-0.026)          |          384.181 (+-0.014)           |            550.517 (+-0.024)          
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        439.885 (+-0.429)        |         416.036 (+-1.768)          |          439.463 (+-0.262)           |            710.031 (+-1.400)          
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |        384.073 (+-0.012)        |         564.323 (+-0.052)          |          384.156 (+-0.022)           |            553.512 (+-0.036)          
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        439.802 (+-0.026)        |         422.735 (+-1.677)          |          439.437 (+-0.027)           |            659.088 (+-0.819)          

Times are in microseconds (us).