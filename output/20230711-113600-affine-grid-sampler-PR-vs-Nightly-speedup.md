Description:

- 20230711-113600-affine-grid-sampler-PR
Torch version: 2.1.0a0+git1c48419
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_86,code=compute_86
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+440fd1bf20

- 20230711-002217-affine-grid-sampler-Nightly
Torch version: 2.1.0a0+gitbcdd413
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_86,code=compute_86
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+440fd1bf20


[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+gitbcdd413) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitbcdd413) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.466 (+-0.072)         |          15.557 (+-0.054)          |             13.292 (+-0.113)            |     0.854 (+-0.000)      |           7.567 (+-0.037)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.685 (+-0.035)         |          11.384 (+-0.024)          |             15.798 (+-0.036)            |     1.388 (+-0.000)      |           7.489 (+-0.114)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.572 (+-0.085)         |          15.867 (+-0.046)          |             12.964 (+-0.050)            |     0.817 (+-0.000)      |           7.623 (+-0.126)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.834 (+-0.169)         |          11.447 (+-0.030)          |             15.386 (+-0.061)            |     1.344 (+-0.000)      |           7.647 (+-0.030)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         5.039 (+-0.011)         |          4.569 (+-0.016)           |             6.383 (+-0.038)             |     1.397 (+-0.000)      |           4.504 (+-0.028)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.326 (+-0.008)         |          4.867 (+-0.013)           |             6.393 (+-0.067)             |     1.314 (+-0.000)      |           4.270 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.085 (+-0.031)         |          4.220 (+-0.006)           |             6.426 (+-0.126)             |     1.523 (+-0.000)      |           4.780 (+-0.204)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.411 (+-0.004)         |          4.619 (+-0.005)           |             6.283 (+-0.114)             |     1.360 (+-0.000)      |           4.315 (+-0.028)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.061 (+-0.083)        |          28.477 (+-0.026)          |             63.423 (+-0.464)            |     2.227 (+-0.000)      |           25.943 (+-0.299)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.358 (+-0.086)        |          30.660 (+-0.328)          |             71.692 (+-0.282)            |     2.338 (+-0.000)      |           26.143 (+-0.299)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.172 (+-0.124)        |          28.072 (+-0.039)          |             65.312 (+-0.478)            |     2.327 (+-0.000)      |           25.810 (+-0.344)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.522 (+-0.065)        |          30.480 (+-0.060)          |             71.560 (+-0.606)            |     2.348 (+-0.000)      |           26.105 (+-1.344)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+gitbcdd413) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitbcdd413) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         88.726 (+-0.344)        |          88.732 (+-0.194)          |            141.983 (+-0.551)            |     1.600 (+-0.000)      |           89.228 (+-0.300)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         88.873 (+-0.366)        |          88.690 (+-0.196)          |            141.351 (+-0.456)            |     1.594 (+-0.000)      |           89.257 (+-0.326)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        110.747 (+-0.742)        |          69.262 (+-0.174)          |            228.701 (+-8.460)            |     3.302 (+-0.000)      |          112.709 (+-0.746)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        110.729 (+-0.421)        |          68.543 (+-0.096)          |            230.542 (+-0.656)            |     3.363 (+-0.000)      |          112.994 (+-0.644)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         78.248 (+-0.323)        |          68.913 (+-0.227)          |            148.836 (+-0.244)            |     2.160 (+-0.000)      |           79.004 (+-0.973)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         77.898 (+-0.362)        |          68.819 (+-0.218)          |            149.036 (+-0.566)            |     2.166 (+-0.000)      |           78.681 (+-0.309)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        111.041 (+-0.404)        |          69.329 (+-0.100)          |            184.097 (+-0.673)            |     2.655 (+-0.000)      |          113.252 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        110.903 (+-0.391)        |          70.003 (+-0.271)          |            183.848 (+-1.566)            |     2.626 (+-0.000)      |          113.787 (+-0.943)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.796 (+-0.536)        |          69.966 (+-0.218)          |            1793.246 (+-0.481)           |     25.630 (+-0.000)     |           92.416 (+-0.072)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.744 (+-0.439)        |          87.140 (+-0.101)          |            1457.581 (+-0.599)           |     16.727 (+-0.000)     |           92.510 (+-0.557)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        110.599 (+-0.331)        |          70.036 (+-0.280)          |            1800.172 (+-0.422)           |     25.704 (+-0.000)     |          112.876 (+-0.498)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        111.289 (+-0.346)        |          86.682 (+-0.020)          |            1463.788 (+-0.566)           |     16.887 (+-0.000)     |          112.987 (+-0.358)         

Times are in microseconds (us).
