Description:

- 20230706-161935-affine-grid-sampler-PR-afgg
Torch version: 2.1.0a0+gita92954b
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

- 20230706-140020-affine-grid-sampler-Nightly
Torch version: 2.1.0a0+gitd3ba890
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


[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cpu ------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gita92954b) PR-afgg  |  Compiled (2.1.0a0+gita92954b) PR-afgg  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  Speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           7.396 (+-0.044)            |             13.152 (+-0.127)            |             13.330 (+-0.028)            |     1.014 (+-0.000)      |           7.522 (+-0.044)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           7.629 (+-0.035)            |             15.760 (+-0.197)            |             16.024 (+-0.128)            |     1.017 (+-0.000)      |           7.530 (+-0.086)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |           7.607 (+-0.097)            |             13.016 (+-0.025)            |             13.149 (+-0.200)            |     1.010 (+-0.000)      |           7.672 (+-0.124)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |           7.928 (+-0.031)            |             15.124 (+-0.048)            |             15.698 (+-0.118)            |     1.038 (+-0.000)      |           7.745 (+-0.078)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           4.376 (+-0.013)            |             5.718 (+-0.012)             |             6.471 (+-0.087)             |     1.132 (+-0.000)      |           4.388 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           4.355 (+-0.005)            |             6.102 (+-0.016)             |             6.490 (+-0.069)             |     1.064 (+-0.000)      |           4.397 (+-0.021)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |           4.847 (+-0.037)            |             5.341 (+-0.012)             |             6.464 (+-0.193)             |     1.210 (+-0.000)      |           4.793 (+-0.024)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |           4.459 (+-0.011)            |             5.888 (+-0.010)             |             6.688 (+-0.196)             |     1.136 (+-0.000)      |           4.370 (+-0.013)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           26.197 (+-0.120)           |             77.394 (+-1.476)            |             64.199 (+-0.402)            |     0.830 (+-0.000)      |           26.645 (+-0.173)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           26.350 (+-0.084)           |             89.640 (+-1.248)            |             71.674 (+-0.679)            |     0.800 (+-0.000)      |           26.498 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |           26.205 (+-0.119)           |             83.627 (+-0.598)            |             66.274 (+-0.172)            |     0.792 (+-0.000)      |           26.758 (+-0.081)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |           26.573 (+-0.172)           |             88.107 (+-0.229)            |             72.297 (+-0.398)            |     0.821 (+-0.000)      |           26.535 (+-0.145)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cuda -----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gita92954b) PR-afgg  |  Compiled (2.1.0a0+gita92954b) PR-afgg  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  Speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           85.616 (+-0.433)           |            120.988 (+-0.409)            |            136.807 (+-0.636)            |     1.131 (+-0.000)      |           86.542 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           85.723 (+-0.443)           |            120.728 (+-0.276)            |            136.184 (+-0.356)            |     1.128 (+-0.000)      |           86.255 (+-0.361)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |          107.531 (+-0.641)           |            146.030 (+-0.369)            |            221.696 (+-0.820)            |     1.518 (+-0.000)      |          110.533 (+-0.721)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |          107.751 (+-0.571)           |            145.755 (+-0.368)            |            222.547 (+-0.218)            |     1.527 (+-0.000)      |          110.922 (+-0.567)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           74.330 (+-0.335)           |            106.024 (+-0.185)            |            142.400 (+-0.258)            |     1.343 (+-0.000)      |           76.351 (+-0.443)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           74.095 (+-0.252)           |            105.237 (+-0.305)            |            142.147 (+-0.709)            |     1.351 (+-0.000)      |           76.435 (+-0.390)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |          107.494 (+-0.367)           |            106.647 (+-0.205)            |            178.589 (+-0.474)            |     1.675 (+-0.000)      |          110.797 (+-0.527)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |          107.748 (+-0.412)           |            105.086 (+-1.852)            |            178.601 (+-0.382)            |     1.700 (+-0.000)      |          109.887 (+-0.417)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           93.169 (+-0.368)           |            1791.232 (+-0.549)           |            1800.770 (+-0.552)           |     1.005 (+-0.000)      |           92.892 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           92.844 (+-0.411)           |            1456.503 (+-1.217)           |            1463.572 (+-0.644)           |     1.005 (+-0.000)      |           92.596 (+-0.489)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |          107.779 (+-0.460)           |            1791.217 (+-0.469)           |            1806.567 (+-0.456)           |     1.009 (+-0.000)      |          110.970 (+-0.444)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |          108.004 (+-0.487)           |            1456.907 (+-0.848)           |            1470.514 (+-1.586)           |     1.009 (+-0.000)      |          110.857 (+-0.506)         

Times are in microseconds (us).
