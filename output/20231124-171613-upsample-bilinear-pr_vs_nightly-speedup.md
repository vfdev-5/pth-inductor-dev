Description:

- 20231124-164848-upsample-nearest-PR
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

- 20231124-162203-upsample-nearest-Nightly
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
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |        602.616 (+-22.507)       |        1672.558 (+-61.445)         |           1192.776 (+-46.017)           |     0.713 (+-0.000)      |          609.795 (+-20.264)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |       2613.521 (+-122.279)      |        1798.871 (+-67.205)         |            416.776 (+-16.439)           |     0.232 (+-0.000)      |         2545.712 (+-81.266)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       1439.590 (+-53.201)       |       12177.282 (+-456.469)        |           2396.603 (+-100.388)          |     0.197 (+-0.000)      |         1444.820 (+-69.206)

      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |        595.688 (+-19.609)       |        1941.486 (+-67.515)         |           2698.828 (+-101.966)          |     1.390 (+-0.000)      |          595.998 (+-19.316)
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |       2543.121 (+-103.392)      |        1849.551 (+-49.013)         |           2497.978 (+-84.750)           |     1.351 (+-0.000)      |         3457.246 (+-161.574)
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       1441.896 (+-64.364)       |       12070.159 (+-457.697)        |          17205.256 (+-652.501)          |     1.425 (+-0.000)      |         1445.773 (+-72.047)

      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |        275.720 (+-12.007)       |        2006.832 (+-54.128)         |            392.934 (+-12.559)           |     0.196 (+-0.000)      |          275.736 (+-11.505)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |        640.172 (+-32.085)       |        1836.372 (+-62.403)         |            369.266 (+-15.056)           |     0.201 (+-0.000)      |          639.984 (+-22.521)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |        561.282 (+-25.214)       |       12424.702 (+-358.165)        |           2169.395 (+-88.032)           |     0.175 (+-0.000)      |          556.940 (+-23.101)

      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |        272.497 (+-9.831)        |        2221.923 (+-77.182)         |           2884.229 (+-106.684)          |     1.298 (+-0.000)      |          275.863 (+-10.525)
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |        635.990 (+-29.975)       |        2071.582 (+-56.936)         |           2658.581 (+-80.302)           |     1.283 (+-0.000)      |          640.408 (+-35.493)
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |        553.702 (+-22.087)       |       13845.586 (+-516.413)        |          18386.437 (+-662.175)          |     1.328 (+-0.000)      |          555.696 (+-21.636)



      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       1567.942 (+-70.205)       |        1658.238 (+-51.966)         |           1672.623 (+-92.869)           |     1.009 (+-0.000)      |         1569.049 (+-61.570)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |       4429.733 (+-118.069)      |        1526.548 (+-56.363)         |           1592.742 (+-95.980)           |     1.043 (+-0.000)      |         4470.015 (+-166.217)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |       7880.872 (+-411.135)      |       10180.096 (+-317.151)        |          10298.510 (+-434.265)          |     1.012 (+-0.000)      |         7872.057 (+-322.271)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       1644.797 (+-55.733)       |        1700.684 (+-62.882)         |           2599.895 (+-74.994)           |     1.529 (+-0.000)      |         1646.217 (+-77.783)
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |       4507.488 (+-167.130)      |        1623.189 (+-53.524)         |           2427.338 (+-125.345)          |     1.495 (+-0.000)      |         4512.917 (+-181.045)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |       8531.282 (+-420.985)      |       10765.625 (+-466.277)        |          16182.090 (+-555.195)          |     1.503 (+-0.000)      |         8516.455 (+-338.801)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |        987.015 (+-40.023)       |        1572.358 (+-50.432)         |           1915.885 (+-75.722)           |     1.218 (+-0.000)      |          988.420 (+-37.083)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |        909.427 (+-37.349)       |        1455.857 (+-55.554)         |           1970.702 (+-107.396)          |     1.354 (+-0.000)      |          909.668 (+-28.623)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |       6304.241 (+-235.947)      |        9606.154 (+-287.330)        |          12069.079 (+-363.372)          |     1.256 (+-0.000)      |         6268.382 (+-216.754)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       1066.768 (+-32.812)       |        1675.391 (+-59.169)         |           2868.948 (+-117.373)          |     1.712 (+-0.000)      |         1066.693 (+-46.852)
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |        987.753 (+-45.583)       |        1583.787 (+-53.015)         |           2738.527 (+-117.980)          |     1.729 (+-0.000)      |          984.206 (+-35.372)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |       6805.252 (+-287.116)      |       10161.756 (+-402.525)        |          18277.502 (+-608.970)          |     1.799 (+-0.000)      |         6795.273 (+-325.086)



      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)       |       2288.113 (+-100.293)      |        7649.243 (+-224.123)        |           1542.603 (+-69.541)           |     0.202 (+-0.000)      |         2288.374 (+-83.640)
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)     |      11015.588 (+-840.089)      |        7032.995 (+-352.263)        |           1457.239 (+-51.895)           |     0.207 (+-0.000)      |         9949.755 (+-520.631)
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)       |       5670.127 (+-243.651)      |       48400.507 (+-1226.293)       |          10180.785 (+-373.179)          |     0.210 (+-0.000)      |         5652.799 (+-253.295)

      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)      |       2282.076 (+-85.629)       |        7611.655 (+-324.243)        |          10743.412 (+-402.738)          |     1.411 (+-0.000)      |         2299.932 (+-126.082)
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)    |       9839.038 (+-466.248)      |        7572.693 (+-289.001)        |          10364.183 (+-464.048)          |     1.369 (+-0.000)      |         9965.452 (+-488.059)
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)      |       5649.836 (+-174.768)      |       50466.810 (+-1342.502)       |          68910.481 (+-2611.443)         |     1.365 (+-0.000)      |         5637.704 (+-213.980)

      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)           |       1003.204 (+-51.813)       |        8323.429 (+-300.154)        |           1519.664 (+-60.119)           |     0.183 (+-0.000)      |         1005.321 (+-49.493)
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)         |       2819.911 (+-109.809)      |        7614.242 (+-310.719)        |           1421.824 (+-59.943)           |     0.187 (+-0.000)      |         2780.579 (+-112.389)
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)           |       2074.881 (+-103.469)      |       52685.511 (+-1617.855)       |           9313.971 (+-364.692)          |     0.177 (+-0.000)      |         2070.967 (+-99.838)

      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)          |       1002.035 (+-45.740)       |       11455.418 (+-365.870)        |          12436.146 (+-502.899)          |     1.086 (+-0.000)      |         1002.460 (+-43.655)
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)        |       2805.970 (+-140.891)      |       10821.756 (+-476.958)        |          11589.904 (+-521.089)          |     1.071 (+-0.000)      |         2796.159 (+-109.443)
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)          |       2064.556 (+-74.383)       |       73569.394 (+-2712.546)       |          78970.826 (+-2265.202)         |     1.073 (+-0.000)      |         2068.290 (+-94.635)



      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (256, 256)     |       6735.109 (+-335.234)      |        6529.901 (+-331.401)        |           7173.570 (+-359.734)          |     1.099 (+-0.000)      |         6686.454 (+-244.620)
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)   |      45569.600 (+-1332.841)     |        6420.924 (+-186.626)        |           8136.140 (+-244.037)          |     1.267 (+-0.000)      |        45711.170 (+-1472.423)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (600, 700)     |      33970.534 (+-1227.498)     |       40777.666 (+-1370.909)       |          40496.394 (+-1407.569)         |     0.993 (+-0.000)      |        34051.758 (+-1229.656)
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (256, 256)    |       7019.610 (+-377.421)      |        6819.203 (+-241.450)        |          10650.626 (+-422.270)          |     1.562 (+-0.000)      |         6994.914 (+-358.409)
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (200, 300)  |      45702.060 (+-1453.013)     |        6805.777 (+-283.069)        |          10853.672 (+-380.916)          |     1.595 (+-0.000)      |        46289.645 (+-1556.288)
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (600, 700)    |      36225.604 (+-1293.904)     |       42056.000 (+-1076.126)       |          62782.248 (+-2365.931)         |     1.493 (+-0.000)      |        36160.560 (+-941.211)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (256, 256)         |       3953.718 (+-145.641)      |        6352.883 (+-315.787)        |           8857.842 (+-379.779)          |     1.394 (+-0.000)      |         3969.317 (+-175.424)
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (200, 300)       |       3887.818 (+-134.293)      |        6514.966 (+-196.642)        |          11057.670 (+-485.730)          |     1.697 (+-0.000)      |         3886.185 (+-109.819)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (600, 700)         |      24999.699 (+-899.604)      |       39654.558 (+-1286.393)       |          62371.204 (+-4647.319)         |     1.573 (+-0.000)      |        25032.083 (+-966.946)
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (256, 256)        |       4316.132 (+-175.782)      |        7379.662 (+-282.127)        |          12301.719 (+-492.169)          |     1.667 (+-0.000)      |         4289.724 (+-151.927)
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (200, 300)      |       4161.505 (+-115.859)      |        7508.279 (+-272.023)        |          13480.937 (+-363.634)          |     1.795 (+-0.000)      |         4192.151 (+-145.135)
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (600, 700)        |      27180.728 (+-1086.220)     |       46042.423 (+-1197.299)       |          75962.411 (+-2923.460)         |     1.650 (+-0.000)      |        27331.612 (+-970.517)

Times are in microseconds (us).

[------------------------------------------------------------------------------------------------------------------------------------------------ Interpolate, cuda ------------------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                    |  Eager (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git0b5d9e3) PR  |  Compiled (2.2.0a0+git9fcf1f9) Nightly  |  speed-up PR vs Nightly  |  Eager (2.2.0a0+git9fcf1f9) Nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |         98.090 (+-0.033)        |          97.579 (+-3.046)          |             97.899 (+-2.185)            |     1.003 (+-0.000)      |           98.112 (+-0.033)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        122.545 (+-0.035)        |         160.818 (+-0.073)          |            189.610 (+-0.060)            |     1.179 (+-0.000)      |          122.573 (+-0.041)
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |         98.132 (+-0.026)        |          97.590 (+-3.458)          |             97.501 (+-1.961)            |     0.999 (+-0.000)      |           98.146 (+-0.033)
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        122.543 (+-0.038)        |         160.267 (+-0.102)          |            160.935 (+-0.054)            |     1.004 (+-0.000)      |          122.525 (+-0.041)

      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |         97.828 (+-0.030)        |          96.049 (+-3.830)          |             92.882 (+-3.761)            |     0.967 (+-0.000)      |           97.853 (+-0.031)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        105.684 (+-0.243)        |         160.362 (+-0.061)          |            174.038 (+-0.071)            |     1.085 (+-0.000)      |          105.652 (+-0.198)
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |         97.927 (+-0.029)        |          96.271 (+-3.265)          |             92.784 (+-3.556)            |     0.964 (+-0.000)      |           97.950 (+-0.034)
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        106.191 (+-0.271)        |         162.036 (+-0.008)          |            162.621 (+-0.086)            |     1.004 (+-0.000)      |          106.042 (+-0.416)

      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (1234, 1345)   |        384.286 (+-0.038)        |         382.553 (+-0.065)          |            382.826 (+-0.080)            |     1.001 (+-0.000)      |          384.284 (+-0.039)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (2345, 2456)   |        445.951 (+-0.035)        |         639.190 (+-0.094)          |            754.859 (+-0.607)            |     1.181 (+-0.000)      |          445.823 (+-0.038)
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (1234, 1345)  |        384.169 (+-0.031)        |         384.236 (+-0.085)          |            382.627 (+-0.081)            |     0.996 (+-0.000)      |          384.205 (+-0.032)
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: bilinear, align_corners: False, osize: (2345, 2456)  |        446.807 (+-0.030)        |         772.890 (+-0.809)          |            638.060 (+-0.072)            |     0.826 (+-0.000)      |          446.806 (+-0.034)

      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (1234, 1345)       |        383.919 (+-0.036)        |         550.322 (+-0.077)          |            547.073 (+-0.091)            |     0.994 (+-0.000)      |          383.924 (+-0.036)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: True, osize: (2345, 2456)       |        439.236 (+-0.047)        |         656.493 (+-1.401)          |            711.314 (+-2.881)            |     1.084 (+-0.000)      |          439.267 (+-0.039)
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (1234, 1345)      |        383.948 (+-0.041)        |         554.558 (+-0.070)          |            549.668 (+-0.068)            |     0.991 (+-0.000)      |          383.934 (+-0.031)
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: bilinear, align_corners: False, osize: (2345, 2456)      |        439.262 (+-0.042)        |         659.183 (+-1.534)          |            662.636 (+-3.067)            |     1.005 (+-0.000)      |          439.287 (+-0.031)

Times are in microseconds (us).
