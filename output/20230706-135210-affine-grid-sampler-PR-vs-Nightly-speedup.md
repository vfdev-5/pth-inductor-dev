Description:

- 20230706-135210-affine-grid-sampler-PR
Torch version: 2.1.0a0+gitd20adf4
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
                                                                                                          |  Eager (2.1.0a0+gitd20adf4) PR  |  Compiled (2.1.0a0+gitd20adf4) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.247 (+-0.059)         |          16.735 (+-0.035)          |             13.330 (+-0.028)            |     0.797 (+-0.000)      |           7.522 (+-0.044)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.287 (+-0.029)         |          12.613 (+-0.144)          |             16.024 (+-0.128)            |     1.270 (+-0.000)      |           7.530 (+-0.086)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.740 (+-0.076)         |          19.568 (+-0.551)          |             13.149 (+-0.200)            |     0.672 (+-0.000)      |           7.672 (+-0.124)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.575 (+-0.037)         |          12.099 (+-0.058)          |             15.698 (+-0.118)            |     1.297 (+-0.000)      |           7.745 (+-0.078)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.870 (+-0.019)         |          5.861 (+-0.013)           |             6.471 (+-0.087)             |     1.104 (+-0.000)      |           4.388 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.347 (+-0.008)         |          6.160 (+-0.017)           |             6.490 (+-0.069)             |     1.054 (+-0.000)      |           4.397 (+-0.021)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.227 (+-0.014)         |          5.382 (+-0.012)           |             6.464 (+-0.193)             |     1.201 (+-0.000)      |           4.793 (+-0.024)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.446 (+-0.011)         |          5.998 (+-0.012)           |             6.688 (+-0.196)             |     1.115 (+-0.000)      |           4.370 (+-0.013)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.087 (+-0.146)        |         99.752 (+-10.110)          |             64.199 (+-0.402)            |     0.644 (+-0.000)      |           26.645 (+-0.173)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.594 (+-0.086)        |         103.234 (+-1.612)          |             71.674 (+-0.679)            |     0.694 (+-0.000)      |           26.498 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.356 (+-0.064)        |         102.336 (+-1.817)          |             66.274 (+-0.172)            |     0.648 (+-0.000)      |           26.758 (+-0.081)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.754 (+-0.099)        |         108.095 (+-1.821)          |             72.297 (+-0.398)            |     0.669 (+-0.000)      |           26.535 (+-0.145)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gitd20adf4) PR  |  Compiled (2.1.0a0+gitd20adf4) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         83.708 (+-0.416)        |         125.210 (+-0.336)          |            136.807 (+-0.636)            |     1.093 (+-0.000)      |           86.542 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         84.536 (+-0.471)        |         125.382 (+-0.512)          |            136.184 (+-0.356)            |     1.086 (+-0.000)      |           86.255 (+-0.361)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        108.149 (+-2.579)        |         110.681 (+-0.399)          |            221.696 (+-0.820)            |     2.003 (+-0.000)      |          110.533 (+-0.721)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        108.137 (+-0.539)        |         110.331 (+-0.212)          |            222.547 (+-0.218)            |     2.017 (+-0.000)      |          110.922 (+-0.567)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         73.548 (+-0.264)        |         110.255 (+-0.252)          |            142.400 (+-0.258)            |     1.292 (+-0.000)      |           76.351 (+-0.443)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         73.379 (+-0.376)        |         110.883 (+-0.448)          |            142.147 (+-0.709)            |     1.282 (+-0.000)      |           76.435 (+-0.390)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        107.504 (+-0.631)        |         110.808 (+-0.460)          |            178.589 (+-0.474)            |     1.612 (+-0.000)      |          110.797 (+-0.527)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        108.152 (+-0.523)        |         110.037 (+-0.320)          |            178.601 (+-0.382)            |     1.623 (+-0.000)      |          109.887 (+-0.417)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.679 (+-0.011)        |         335.387 (+-0.017)          |            1800.770 (+-0.552)           |     5.369 (+-0.000)      |           92.892 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.817 (+-0.498)        |         334.844 (+-0.022)          |            1463.572 (+-0.644)           |     4.371 (+-0.000)      |           92.596 (+-0.489)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        107.543 (+-0.603)        |         333.839 (+-0.017)          |            1806.567 (+-0.456)           |     5.412 (+-0.000)      |          110.970 (+-0.444)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        108.494 (+-0.497)        |         334.746 (+-0.023)          |            1470.514 (+-1.586)           |     4.393 (+-0.000)      |          110.857 (+-0.506)         

Times are in microseconds (us).
