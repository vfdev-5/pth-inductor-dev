Description:

- 20230818-092548-affine-grid-sampler-PR
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
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.841 (+-0.053)         |          12.576 (+-0.057)          |             12.356 (+-0.054)            |     0.983 (+-0.000)      |           7.512 (+-0.325)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.984 (+-0.043)         |          11.781 (+-0.051)          |             14.186 (+-0.095)            |     1.204 (+-0.000)      |           7.511 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         7.825 (+-0.057)         |          12.114 (+-0.069)          |             11.570 (+-0.055)            |     0.955 (+-0.000)      |           7.843 (+-0.051)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.108 (+-0.059)         |          11.784 (+-0.050)          |             13.615 (+-0.135)            |     1.155 (+-0.000)      |           7.809 (+-0.049)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.826 (+-0.025)         |          4.626 (+-0.020)           |             4.479 (+-0.020)             |     0.968 (+-0.000)      |           4.621 (+-0.026)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.568 (+-0.017)         |          5.053 (+-0.019)           |             4.819 (+-0.030)             |     0.954 (+-0.000)      |           4.179 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.222 (+-0.021)         |          4.737 (+-0.022)           |             4.133 (+-0.062)             |     0.873 (+-0.000)      |           4.639 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.714 (+-0.013)         |          4.905 (+-0.017)           |             4.607 (+-0.015)             |     0.939 (+-0.000)      |           4.485 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.708 (+-0.170)        |          29.308 (+-0.084)          |             63.548 (+-0.410)            |     2.168 (+-0.000)      |           26.105 (+-0.115)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.719 (+-0.138)        |          31.073 (+-0.071)          |             70.633 (+-0.550)            |     2.273 (+-0.000)      |           26.554 (+-0.152)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.354 (+-0.940)        |          29.062 (+-0.106)          |             65.347 (+-0.085)            |     2.249 (+-0.000)      |           26.109 (+-0.177)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.597 (+-0.118)        |          30.589 (+-0.123)          |             75.345 (+-1.267)            |     2.463 (+-0.000)      |           26.464 (+-0.132)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         90.241 (+-0.351)        |          89.540 (+-0.300)          |             94.940 (+-0.366)            |     1.060 (+-0.000)      |           99.678 (+-0.518)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         90.603 (+-0.468)        |          89.968 (+-0.315)          |             94.639 (+-0.285)            |     1.052 (+-0.000)      |           98.618 (+-1.816)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        113.329 (+-0.428)        |          68.345 (+-0.376)          |            116.332 (+-0.420)            |     1.702 (+-0.000)      |          126.473 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        112.630 (+-0.408)        |          69.527 (+-0.355)          |            115.988 (+-0.397)            |     1.668 (+-0.000)      |          126.625 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         79.712 (+-0.575)        |          69.109 (+-0.315)          |             73.989 (+-0.263)            |     1.071 (+-0.000)      |           89.014 (+-0.605)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         80.057 (+-0.345)        |          69.478 (+-0.288)          |             74.751 (+-0.319)            |     1.076 (+-0.000)      |           88.534 (+-0.718)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        112.984 (+-0.482)        |          68.781 (+-0.418)          |             74.502 (+-0.273)            |     1.083 (+-0.000)      |          127.675 (+-0.801)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        113.269 (+-0.553)        |          69.246 (+-0.372)          |             74.760 (+-0.486)            |     1.080 (+-0.000)      |          126.778 (+-0.608)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.955 (+-0.039)        |          69.319 (+-0.247)          |            1740.788 (+-0.537)           |     25.113 (+-0.000)     |           92.959 (+-0.045)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.718 (+-0.204)        |         120.858 (+-0.043)          |            1401.371 (+-0.814)           |     11.595 (+-0.000)     |           92.844 (+-0.356)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        113.298 (+-0.394)        |          69.377 (+-0.308)          |            1741.294 (+-0.518)           |     25.099 (+-0.000)     |          127.256 (+-0.646)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        113.072 (+-0.444)        |         120.555 (+-0.034)          |            1400.857 (+-0.709)           |     11.620 (+-0.000)     |          126.624 (+-0.434)         

Times are in microseconds (us).
