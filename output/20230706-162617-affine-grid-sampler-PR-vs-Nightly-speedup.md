Description:

- 20230706-162617-affine-grid-sampler-PR
Torch version: 2.1.0a0+git421fe7b
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


[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  Speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.489 (+-0.287)         |          16.801 (+-0.138)          |             13.330 (+-0.028)            |     0.793 (+-0.000)      |           7.522 (+-0.044)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.587 (+-0.031)         |          12.494 (+-0.066)          |             16.024 (+-0.128)            |     1.283 (+-0.000)      |           7.530 (+-0.086)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         7.808 (+-0.038)         |          20.410 (+-1.616)          |             13.149 (+-0.200)            |     0.644 (+-0.000)      |           7.672 (+-0.124)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         7.989 (+-0.034)         |          12.130 (+-0.033)          |             15.698 (+-0.118)            |     1.294 (+-0.000)      |           7.745 (+-0.078)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.593 (+-0.030)         |          5.848 (+-0.012)           |             6.471 (+-0.087)             |     1.106 (+-0.000)      |           4.388 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.190 (+-0.008)         |          5.979 (+-0.008)           |             6.490 (+-0.069)             |     1.085 (+-0.000)      |           4.397 (+-0.021)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.582 (+-0.011)         |          5.465 (+-0.024)           |             6.464 (+-0.193)             |     1.183 (+-0.000)      |           4.793 (+-0.024)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.403 (+-0.004)         |          5.866 (+-0.007)           |             6.688 (+-0.196)             |     1.140 (+-0.000)      |           4.370 (+-0.013)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.106 (+-0.138)        |         104.156 (+-3.881)          |             64.199 (+-0.402)            |     0.616 (+-0.000)      |           26.645 (+-0.173)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.192 (+-0.141)        |         102.890 (+-1.249)          |             71.674 (+-0.679)            |     0.697 (+-0.000)      |           26.498 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         25.752 (+-0.133)        |          99.068 (+-3.399)          |             66.274 (+-0.172)            |     0.669 (+-0.000)      |           26.758 (+-0.081)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.366 (+-0.082)        |         103.052 (+-1.758)          |             72.297 (+-0.398)            |     0.702 (+-0.000)      |           26.535 (+-0.145)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  Speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         88.257 (+-0.462)        |         125.216 (+-0.401)          |            136.807 (+-0.636)            |     1.093 (+-0.000)      |           86.542 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         87.649 (+-0.440)        |         125.382 (+-3.365)          |            136.184 (+-0.356)            |     1.086 (+-0.000)      |           86.255 (+-0.361)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        111.428 (+-0.511)        |         108.644 (+-0.338)          |            221.696 (+-0.820)            |     2.041 (+-0.000)      |          110.533 (+-0.721)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        110.717 (+-0.458)        |         108.719 (+-0.427)          |            222.547 (+-0.218)            |     2.047 (+-0.000)      |          110.922 (+-0.567)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         77.541 (+-0.317)        |         108.937 (+-0.301)          |            142.400 (+-0.258)            |     1.307 (+-0.000)      |           76.351 (+-0.443)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         77.313 (+-0.341)        |         108.872 (+-0.421)          |            142.147 (+-0.709)            |     1.306 (+-0.000)      |           76.435 (+-0.390)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        110.669 (+-0.475)        |         109.328 (+-0.345)          |            178.589 (+-0.474)            |     1.634 (+-0.000)      |          110.797 (+-0.527)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        110.605 (+-0.521)        |         109.049 (+-0.401)          |            178.601 (+-0.382)            |     1.638 (+-0.000)      |          109.887 (+-0.417)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.652 (+-0.097)        |         333.377 (+-0.011)          |            1800.770 (+-0.552)           |     5.402 (+-0.000)      |           92.892 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.280 (+-0.373)        |         334.606 (+-0.026)          |            1463.572 (+-0.644)           |     4.374 (+-0.000)      |           92.596 (+-0.489)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        110.864 (+-0.533)        |         333.195 (+-0.016)          |            1806.567 (+-0.456)           |     5.422 (+-0.000)      |          110.970 (+-0.444)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        112.047 (+-0.700)        |         334.676 (+-0.028)          |            1470.514 (+-1.586)           |     4.394 (+-0.000)      |          110.857 (+-0.506)         

Times are in microseconds (us).
