Description:

- 20230817-175337-affine-grid-sampler-PR
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
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.477 (+-0.151)         |          12.534 (+-0.068)          |             12.356 (+-0.054)            |     0.986 (+-0.000)      |           7.512 (+-0.325)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.559 (+-0.052)         |          11.432 (+-0.065)          |             14.186 (+-0.095)            |     1.241 (+-0.000)      |           7.511 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.645 (+-0.077)         |          11.730 (+-0.052)          |             11.570 (+-0.055)            |     0.986 (+-0.000)      |           7.843 (+-0.051)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.791 (+-0.105)         |          11.506 (+-0.071)          |             13.615 (+-0.135)            |     1.183 (+-0.000)      |           7.809 (+-0.049)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.724 (+-0.027)         |          4.460 (+-0.027)           |             4.479 (+-0.020)             |     1.004 (+-0.000)      |           4.621 (+-0.026)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.364 (+-0.013)         |          5.043 (+-0.020)           |             4.819 (+-0.030)             |     0.956 (+-0.000)      |           4.179 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.912 (+-0.036)         |          4.195 (+-0.026)           |             4.133 (+-0.062)             |     0.985 (+-0.000)      |           4.639 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.657 (+-0.017)         |          4.742 (+-0.017)           |             4.607 (+-0.015)             |     0.971 (+-0.000)      |           4.485 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.227 (+-0.167)        |          62.888 (+-0.318)          |             63.548 (+-0.410)            |     1.010 (+-0.000)      |           26.105 (+-0.115)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.581 (+-0.097)        |          31.033 (+-0.071)          |             70.633 (+-0.550)            |     2.276 (+-0.000)      |           26.554 (+-0.152)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.504 (+-0.140)        |          65.564 (+-0.344)          |             65.347 (+-0.085)            |     0.997 (+-0.000)      |           26.109 (+-0.177)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.647 (+-0.147)        |          30.911 (+-0.096)          |             75.345 (+-1.267)            |     2.437 (+-0.000)      |           26.464 (+-0.132)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         87.546 (+-1.444)        |          90.463 (+-0.330)          |             94.940 (+-0.366)            |     1.049 (+-0.000)      |           99.678 (+-0.518)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         87.217 (+-0.677)        |          89.641 (+-0.415)          |             94.639 (+-0.285)            |     1.056 (+-0.000)      |           98.618 (+-1.816)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        106.366 (+-0.684)        |          71.323 (+-0.489)          |            116.332 (+-0.420)            |     1.631 (+-0.000)      |          126.473 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        106.647 (+-0.338)        |          72.938 (+-0.300)          |            115.988 (+-0.397)            |     1.590 (+-0.000)      |          126.625 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         75.192 (+-0.316)        |          71.374 (+-0.303)          |             73.989 (+-0.263)            |     1.037 (+-0.000)      |           89.014 (+-0.605)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         75.266 (+-0.308)        |          72.630 (+-0.292)          |             74.751 (+-0.319)            |     1.029 (+-0.000)      |           88.534 (+-0.718)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        106.671 (+-0.503)        |          71.691 (+-0.266)          |             74.502 (+-0.273)            |     1.039 (+-0.000)      |          127.675 (+-0.801)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        107.302 (+-0.493)        |          72.265 (+-0.252)          |             74.760 (+-0.486)            |     1.035 (+-0.000)      |          126.778 (+-0.608)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.969 (+-0.026)        |          71.527 (+-0.351)          |            1740.788 (+-0.537)           |     24.337 (+-0.000)     |           92.959 (+-0.045)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.726 (+-0.290)        |         118.260 (+-0.026)          |            1401.371 (+-0.814)           |     11.850 (+-0.000)     |           92.844 (+-0.356)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        106.748 (+-0.573)        |          71.456 (+-0.299)          |            1741.294 (+-0.518)           |     24.369 (+-0.000)     |          127.256 (+-0.646)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        106.861 (+-1.025)        |         118.110 (+-0.030)          |            1400.857 (+-0.709)           |     11.861 (+-0.000)     |          126.624 (+-0.434)         

Times are in microseconds (us).
