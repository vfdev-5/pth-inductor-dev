Description:
- 20240301-152339-upsample-nearest-PR
Torch version: 2.3.0a0+gitb4324ed
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 


- 20240301-151104-upsample-nearest-Nightly
Torch version: 2.3.0a0+git0d1e705
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 



[--------------------------------------------------------------------------------------------------------------------------------- Interpolate, cpu ---------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                |  Eager (2.3.0a0+gitb4324ed) PR  |  Compiled (2.3.0a0+gitb4324ed) PR  |  Eager (2.3.0a0+git0d1e705) Nightly  |  Compiled (2.3.0a0+git0d1e705) Nightly
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)      |        287.988 (+-10.399)       |         200.034 (+-8.630)          |          287.991 (+-11.302)          |            285.143 (+-8.412)          
      Input (1, 3, 500, 400), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)          |        697.206 (+-27.033)       |         171.650 (+-7.381)          |          701.642 (+-26.461)          |            193.280 (+-5.840)          
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)    |        149.149 (+-6.045)        |         222.780 (+-6.852)          |          145.055 (+-7.232)           |            299.968 (+-12.354)         
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)        |        596.741 (+-27.970)       |         205.923 (+-8.648)          |          598.000 (+-25.630)          |            233.912 (+-7.742)          
      Input (4, 3, 500, 400), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)      |       1095.734 (+-51.658)       |         700.850 (+-24.852)         |         1097.977 (+-35.521)          |           1044.255 (+-38.216)         
      Input (4, 3, 500, 400), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)          |       2741.813 (+-122.917)      |         583.073 (+-16.998)         |         2722.388 (+-116.263)         |            665.029 (+-36.331)         
      Input (4, 3, 500, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)    |        578.183 (+-37.266)       |         833.295 (+-42.264)         |          584.953 (+-45.549)          |           1131.341 (+-54.710)         
      Input (4, 3, 500, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)        |       2332.508 (+-103.556)      |         840.194 (+-47.664)         |         2334.314 (+-91.644)          |            935.625 (+-47.467)         
      Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (200, 300)    |        272.631 (+-11.348)       |         195.988 (+-5.748)          |          272.752 (+-12.716)          |            274.021 (+-9.475)          
      Input (1, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (200, 300)        |        640.409 (+-25.465)       |         164.773 (+-7.372)          |          639.390 (+-30.761)          |            185.018 (+-8.349)          
      Input (1, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (200, 300)  |        158.602 (+-6.593)        |         220.478 (+-6.809)          |          158.557 (+-6.143)           |            286.376 (+-8.981)          
      Input (1, 3, 1200, 1300), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (200, 300)      |        548.903 (+-22.889)       |         202.788 (+-9.158)          |          554.096 (+-21.330)          |            227.404 (+-8.995)          
      Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (200, 300)    |       1036.061 (+-35.285)       |         680.728 (+-30.925)         |         1038.718 (+-43.070)          |            986.254 (+-42.732)         
      Input (4, 3, 1200, 1300), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (200, 300)        |       2504.520 (+-125.805)      |         550.067 (+-21.383)         |         2523.134 (+-113.336)         |            628.000 (+-27.589)         
      Input (4, 3, 1200, 1300), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (200, 300)  |       1058.188 (+-57.853)       |        1216.427 (+-76.160)         |         1057.031 (+-66.075)          |           1380.231 (+-98.939)         
      Input (4, 3, 1200, 1300), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (200, 300)      |       2305.911 (+-116.864)      |        1080.189 (+-79.934)         |         2306.606 (+-121.544)         |           1141.561 (+-67.959)         
      Input (1, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)      |       1689.489 (+-60.579)       |        1077.401 (+-44.948)         |         1693.945 (+-67.998)          |           1634.264 (+-64.340)         
      Input (1, 3, 300, 400), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)          |       4198.368 (+-179.096)      |         886.656 (+-30.355)         |         4174.351 (+-141.020)         |           1028.568 (+-46.310)         
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)    |        716.572 (+-51.954)       |        1175.864 (+-52.191)         |          715.724 (+-41.104)          |           1674.373 (+-51.815)         
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)        |       3604.989 (+-132.489)      |        1096.933 (+-54.290)         |         3601.864 (+-140.218)         |           1270.347 (+-60.932)         
      Input (4, 3, 300, 400), torch.uint8, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)      |       6721.610 (+-355.997)      |        4203.213 (+-134.362)        |         6715.626 (+-288.233)         |           6423.763 (+-225.311)        
      Input (4, 3, 300, 400), torch.uint8, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)          |      16695.467 (+-709.620)      |        3460.013 (+-149.456)        |        16621.138 (+-713.320)         |           4001.810 (+-218.093)        
      Input (4, 3, 300, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)    |       3020.017 (+-147.314)      |        4743.164 (+-135.850)        |         3015.602 (+-105.852)         |           6709.494 (+-281.025)        
      Input (4, 3, 300, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)        |      14456.688 (+-752.839)      |        5150.893 (+-201.571)        |        14464.472 (+-720.027)         |           5737.315 (+-138.011)        

Times are in microseconds (us).

[--------------------------------------------------------------------------------------------------------------------------------- Interpolate, cuda ----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                                                  |  Eager (2.3.0a0+gitb4324ed) PR  |  Compiled (2.3.0a0+gitb4324ed) PR  |  Eager (2.3.0a0+git0d1e705) Nightly  |  Compiled (2.3.0a0+git0d1e705) Nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (1234, 1345)  |         17.574 (+-0.017)        |          61.821 (+-2.094)          |           17.613 (+-0.016)           |             62.611 (+-1.573)          
      Input (1, 3, 2345, 2456), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (1234, 1345)      |        369.728 (+-0.054)        |          62.267 (+-1.590)          |          369.766 (+-0.048)           |             65.444 (+-3.579)          
      Input (4, 3, 2345, 2456), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (1234, 1345)  |        246.265 (+-0.042)        |         246.589 (+-0.085)          |          246.276 (+-0.049)           |            246.585 (+-0.109)          
      Input (4, 3, 2345, 2456), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (1234, 1345)      |        1590.070 (+-0.454)       |         275.124 (+-0.066)          |          1590.059 (+-0.405)          |            275.129 (+-0.067)          
      Input (1, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (2345, 2456)  |        100.363 (+-0.040)        |          99.064 (+-0.065)          |          100.333 (+-0.048)           |             99.064 (+-0.038)          
      Input (1, 3, 1234, 1345), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (2345, 2456)      |        279.931 (+-0.059)        |          86.573 (+-0.134)          |          279.915 (+-0.038)           |             86.644 (+-0.063)          
      Input (4, 3, 1234, 1345), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (2345, 2456)  |        401.513 (+-0.042)        |         395.308 (+-0.065)          |          401.541 (+-0.043)           |            395.323 (+-0.054)          
      Input (4, 3, 1234, 1345), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (2345, 2456)      |        1180.081 (+-0.375)       |         416.067 (+-0.120)          |          1180.223 (+-0.472)          |            416.157 (+-0.198)          

Times are in microseconds (us).
