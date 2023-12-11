Description:

- 20230817-231218-affine-grid-sampler-PR
Torch version: 2.1.0a0+git1afae24
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_86,code=compute_86
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+440fd1bf20

- 20230817-170303-affine-grid-sampler-Nightly
Torch version: 2.1.0a0+git2932b0b
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_86,code=compute_86
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+440fd1bf20


[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.695 (+-0.056)         |          12.308 (+-0.074)          |             12.356 (+-0.054)            |     1.004 (+-0.000)      |           7.512 (+-0.325)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.001 (+-0.046)         |          11.562 (+-0.079)          |             14.186 (+-0.095)            |     1.227 (+-0.000)      |           7.511 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.215 (+-0.055)         |          12.919 (+-0.062)          |             11.570 (+-0.055)            |     0.896 (+-0.000)      |           7.843 (+-0.051)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         7.997 (+-0.048)         |          12.030 (+-0.070)          |             13.615 (+-0.135)            |     1.132 (+-0.000)      |           7.809 (+-0.049)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.580 (+-0.053)         |          4.507 (+-0.025)           |             4.479 (+-0.020)             |     0.994 (+-0.000)      |           4.621 (+-0.026)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.568 (+-0.014)         |          5.098 (+-0.073)           |             4.819 (+-0.030)             |     0.945 (+-0.000)      |           4.179 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.647 (+-0.032)         |          4.161 (+-0.023)           |             4.133 (+-0.062)             |     0.993 (+-0.000)      |           4.639 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.742 (+-0.017)         |          4.862 (+-0.018)           |             4.607 (+-0.015)             |     0.948 (+-0.000)      |           4.485 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.315 (+-0.180)        |          29.303 (+-0.087)          |             63.548 (+-0.410)            |     2.169 (+-0.000)      |           26.105 (+-0.115)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.490 (+-0.187)        |          33.719 (+-0.151)          |             70.633 (+-0.550)            |     2.095 (+-0.000)      |           26.554 (+-0.152)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.508 (+-0.122)        |          28.968 (+-0.078)          |             65.347 (+-0.085)            |     2.256 (+-0.000)      |           26.109 (+-0.177)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.606 (+-0.123)        |          32.502 (+-0.133)          |             75.345 (+-1.267)            |     2.318 (+-0.000)      |           26.464 (+-0.132)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         85.290 (+-0.388)        |          88.872 (+-0.325)          |             94.940 (+-0.366)            |     1.068 (+-0.000)      |           99.678 (+-0.518)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         85.677 (+-0.352)        |          89.251 (+-0.334)          |             94.639 (+-0.285)            |     1.060 (+-0.000)      |           98.618 (+-1.816)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        105.644 (+-0.371)        |          70.114 (+-0.364)          |            116.332 (+-0.420)            |     1.659 (+-0.000)      |          126.473 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        106.599 (+-0.359)        |          70.763 (+-0.407)          |            115.988 (+-0.397)            |     1.639 (+-0.000)      |          126.625 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         73.111 (+-0.752)        |          69.960 (+-0.123)          |             73.989 (+-0.263)            |     1.058 (+-0.000)      |           89.014 (+-0.605)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         73.360 (+-0.434)        |          70.120 (+-0.383)          |             74.751 (+-0.319)            |     1.066 (+-0.000)      |           88.534 (+-0.718)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        105.942 (+-0.645)        |          69.978 (+-0.283)          |             74.502 (+-0.273)            |     1.065 (+-0.000)      |          127.675 (+-0.801)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        106.891 (+-0.512)        |          70.542 (+-0.490)          |             74.760 (+-0.486)            |     1.060 (+-0.000)      |          126.778 (+-0.608)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         93.036 (+-0.067)        |          70.620 (+-0.330)          |            1740.788 (+-0.537)           |     24.650 (+-0.000)     |           92.959 (+-0.045)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.805 (+-0.053)        |         120.789 (+-0.035)          |            1401.371 (+-0.814)           |     11.602 (+-0.000)     |           92.844 (+-0.356)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        106.055 (+-0.469)        |          70.114 (+-0.310)          |            1741.294 (+-0.518)           |     24.835 (+-0.000)     |          127.256 (+-0.646)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        105.890 (+-1.088)        |         120.577 (+-0.053)          |            1400.857 (+-0.709)           |     11.618 (+-0.000)     |          126.624 (+-0.434)         

Times are in microseconds (us).
