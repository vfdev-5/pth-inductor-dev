Description:

- 20230801-220216-affine-grid-sampler-PR-afgg
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

- 20230801-203836-affine-grid-sampler-Nightly
Torch version: 2.1.0a0+git16df542
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


[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cpu ------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR-afgg  |  Compiled (2.1.0a0+git1afae24) PR-afgg  |  Compiled (2.1.0a0+git16df542) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git16df542) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           7.467 (+-0.036)            |             11.905 (+-0.276)            |             13.391 (+-0.051)            |     1.125 (+-0.000)      |           7.343 (+-0.036)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           7.722 (+-0.168)            |             14.371 (+-0.035)            |             15.899 (+-0.038)            |     1.106 (+-0.000)      |           7.870 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |           7.710 (+-0.051)            |             11.354 (+-0.053)            |             13.376 (+-0.045)            |     1.178 (+-0.000)      |           7.698 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |           7.870 (+-0.050)            |             13.744 (+-0.237)            |             15.206 (+-0.102)            |     1.106 (+-0.000)      |           7.912 (+-0.039)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           4.738 (+-0.015)            |             4.508 (+-0.005)             |             6.566 (+-0.027)             |     1.456 (+-0.000)      |           4.630 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           4.391 (+-0.010)            |             4.860 (+-0.390)             |             6.438 (+-0.047)             |     1.325 (+-0.000)      |           4.458 (+-0.010)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |           4.279 (+-0.008)            |             4.127 (+-0.010)             |             6.598 (+-0.709)             |     1.599 (+-0.000)      |           5.064 (+-0.025)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |           4.537 (+-0.010)            |             4.593 (+-0.006)             |             6.365 (+-0.104)             |     1.386 (+-0.000)      |           4.480 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           26.411 (+-0.066)           |             62.275 (+-0.436)            |             64.486 (+-0.353)            |     1.035 (+-0.000)      |           26.210 (+-0.110)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           26.457 (+-0.096)           |             72.887 (+-0.247)            |             74.207 (+-0.337)            |     1.018 (+-0.000)      |           25.995 (+-0.120)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |           26.457 (+-0.086)           |             64.110 (+-0.233)            |             66.340 (+-0.406)            |     1.035 (+-0.000)      |           26.145 (+-0.085)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |           26.536 (+-0.094)           |             73.742 (+-0.483)            |             71.946 (+-0.460)            |     0.976 (+-0.000)      |           26.457 (+-0.166)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cuda -----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR-afgg  |  Compiled (2.1.0a0+git1afae24) PR-afgg  |  Compiled (2.1.0a0+git16df542) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git16df542) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           91.971 (+-0.253)           |             90.570 (+-0.193)            |            137.206 (+-0.214)            |     1.515 (+-0.000)      |           84.280 (+-0.241)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           91.893 (+-0.361)           |             89.866 (+-0.170)            |            136.678 (+-0.471)            |     1.521 (+-0.000)      |           84.573 (+-0.214)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |          116.967 (+-0.481)           |            110.468 (+-0.326)            |            223.770 (+-0.334)            |     2.026 (+-0.000)      |          108.098 (+-0.392)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |          117.563 (+-0.546)           |            111.438 (+-0.212)            |            223.101 (+-0.350)            |     2.002 (+-0.000)      |          108.225 (+-0.395)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           80.706 (+-0.289)           |             70.525 (+-0.204)            |            143.697 (+-0.311)            |     2.038 (+-0.000)      |           74.485 (+-0.258)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           80.955 (+-0.208)           |             69.986 (+-0.250)            |            143.658 (+-0.244)            |     2.053 (+-0.000)      |           74.163 (+-0.238)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |          117.576 (+-0.435)           |             71.179 (+-0.412)            |            178.515 (+-0.539)            |     2.508 (+-0.000)      |          108.394 (+-0.473)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |          117.441 (+-0.205)           |             70.313 (+-0.170)            |            178.664 (+-0.555)            |     2.541 (+-0.000)      |          108.098 (+-0.416)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           92.962 (+-0.509)           |            1740.964 (+-0.597)           |            1785.401 (+-0.369)           |     1.026 (+-0.000)      |           92.638 (+-0.539)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           92.928 (+-0.493)           |            1401.146 (+-0.732)           |            1453.229 (+-0.628)           |     1.037 (+-0.000)      |           92.458 (+-0.428)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |          118.152 (+-0.442)           |            1740.644 (+-0.480)           |            1793.475 (+-0.458)           |     1.030 (+-0.000)      |          107.962 (+-0.548)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |          118.182 (+-0.425)           |            1400.621 (+-0.624)           |            1461.796 (+-0.630)           |     1.044 (+-0.000)      |          107.894 (+-0.994)         

Times are in microseconds (us).
