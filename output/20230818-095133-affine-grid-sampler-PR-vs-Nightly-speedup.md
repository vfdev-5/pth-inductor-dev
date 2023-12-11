Description:

- 20230818-095133-affine-grid-sampler-PR
Torch version: 2.1.0a0+git0efd50c
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
                                                                                                          |  Eager (2.1.0a0+git0efd50c) PR  |  Compiled (2.1.0a0+git0efd50c) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.493 (+-0.356)         |          12.468 (+-0.129)          |             12.356 (+-0.054)            |     0.991 (+-0.000)      |           7.512 (+-0.325)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.658 (+-0.046)         |          11.514 (+-0.057)          |             14.186 (+-0.095)            |     1.232 (+-0.000)      |           7.511 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.467 (+-0.092)         |          11.384 (+-0.052)          |             11.570 (+-0.055)            |     1.016 (+-0.000)      |           7.843 (+-0.051)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.631 (+-0.066)         |          11.095 (+-0.088)          |             13.615 (+-0.135)            |     1.227 (+-0.000)      |           7.809 (+-0.049)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.331 (+-0.033)         |          4.479 (+-0.027)           |             4.479 (+-0.020)             |     1.000 (+-0.000)      |           4.621 (+-0.026)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.545 (+-0.016)         |          5.092 (+-0.019)           |             4.819 (+-0.030)             |     0.946 (+-0.000)      |           4.179 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.731 (+-0.030)         |          4.173 (+-0.022)           |             4.133 (+-0.062)             |     0.991 (+-0.000)      |           4.639 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.175 (+-0.010)         |          4.685 (+-0.017)           |             4.607 (+-0.015)             |     0.983 (+-0.000)      |           4.485 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.332 (+-0.139)        |          28.979 (+-0.084)          |             63.548 (+-0.410)            |     2.193 (+-0.000)      |           26.105 (+-0.115)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.446 (+-1.484)        |          29.402 (+-0.150)          |             70.633 (+-0.550)            |     2.402 (+-0.000)      |           26.554 (+-0.152)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.350 (+-0.145)        |          28.647 (+-0.093)          |             65.347 (+-0.085)            |     2.281 (+-0.000)      |           26.109 (+-0.177)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.612 (+-0.182)        |          28.074 (+-0.132)          |             75.345 (+-1.267)            |     2.684 (+-0.000)      |           26.464 (+-0.132)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git0efd50c) PR  |  Compiled (2.1.0a0+git0efd50c) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         88.298 (+-0.376)        |          90.564 (+-0.339)          |             94.940 (+-0.366)            |     1.048 (+-0.000)      |           99.678 (+-0.518)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         88.541 (+-0.369)        |          90.630 (+-0.418)          |             94.639 (+-0.285)            |     1.044 (+-0.000)      |           98.618 (+-1.816)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        112.919 (+-0.467)        |          70.385 (+-0.299)          |            116.332 (+-0.420)            |     1.653 (+-0.000)      |          126.473 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        112.744 (+-0.472)        |          70.659 (+-0.310)          |            115.988 (+-0.397)            |     1.642 (+-0.000)      |          126.625 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         79.507 (+-0.378)        |          70.232 (+-0.419)          |             73.989 (+-0.263)            |     1.054 (+-0.000)      |           89.014 (+-0.605)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         78.728 (+-0.311)        |          70.304 (+-0.291)          |             74.751 (+-0.319)            |     1.063 (+-0.000)      |           88.534 (+-0.718)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        112.984 (+-0.375)        |          70.225 (+-0.169)          |             74.502 (+-0.273)            |     1.061 (+-0.000)      |          127.675 (+-0.801)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        112.442 (+-0.458)        |          70.228 (+-0.311)          |             74.760 (+-0.486)            |     1.065 (+-0.000)      |          126.778 (+-0.608)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.969 (+-0.082)        |          70.693 (+-0.321)          |            1740.788 (+-0.537)           |     24.625 (+-0.000)     |           92.959 (+-0.045)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.722 (+-0.259)        |         120.756 (+-0.055)          |            1401.371 (+-0.814)           |     11.605 (+-0.000)     |           92.844 (+-0.356)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        112.789 (+-0.882)        |          70.186 (+-0.246)          |            1741.294 (+-0.518)           |     24.810 (+-0.000)     |          127.256 (+-0.646)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        112.482 (+-0.431)        |         120.487 (+-0.033)          |            1400.857 (+-0.709)           |     11.627 (+-0.000)     |          126.624 (+-0.434)         

Times are in microseconds (us).
