Description:

- 20230710-205742-affine-grid-sampler-PR
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
                                                                                                          |  Eager (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.626 (+-0.043)         |          15.824 (+-0.036)          |             13.330 (+-0.028)            |     0.842 (+-0.000)      |           7.522 (+-0.044)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.720 (+-0.030)         |          11.376 (+-0.044)          |             16.024 (+-0.128)            |     1.409 (+-0.000)      |           7.530 (+-0.086)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.096 (+-0.050)         |          16.750 (+-0.062)          |             13.149 (+-0.200)            |     0.785 (+-0.000)      |           7.672 (+-0.124)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         7.915 (+-0.073)         |          11.355 (+-0.025)          |             15.698 (+-0.118)            |     1.382 (+-0.000)      |           7.745 (+-0.078)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.760 (+-0.011)         |          4.570 (+-0.007)           |             6.471 (+-0.087)             |     1.416 (+-0.000)      |           4.388 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.516 (+-0.011)         |          4.910 (+-0.006)           |             6.490 (+-0.069)             |     1.322 (+-0.000)      |           4.397 (+-0.021)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.088 (+-0.044)         |          4.210 (+-0.026)           |             6.464 (+-0.193)             |     1.535 (+-0.000)      |           4.793 (+-0.024)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.414 (+-0.196)         |          4.608 (+-0.006)           |             6.688 (+-0.196)             |     1.451 (+-0.000)      |           4.370 (+-0.013)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.502 (+-0.559)        |          29.301 (+-0.042)          |             64.199 (+-0.402)            |     2.191 (+-0.000)      |           26.645 (+-0.173)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.185 (+-0.095)        |          32.850 (+-0.053)          |             71.674 (+-0.679)            |     2.182 (+-0.000)      |           26.498 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.445 (+-0.248)        |          28.069 (+-0.125)          |             66.274 (+-0.172)            |     2.361 (+-0.000)      |           26.758 (+-0.081)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.338 (+-0.186)        |          33.278 (+-0.066)          |             72.297 (+-0.398)            |     2.172 (+-0.000)      |           26.535 (+-0.145)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+git421fe7b) PR  |  Compiled (2.1.0a0+gitd3ba890) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitd3ba890) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         91.857 (+-0.281)        |          90.289 (+-0.222)          |            136.807 (+-0.636)            |     1.515 (+-0.000)      |           86.542 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         92.283 (+-0.363)        |          90.315 (+-0.791)          |            136.184 (+-0.356)            |     1.508 (+-0.000)      |           86.255 (+-0.361)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        120.941 (+-0.681)        |          74.034 (+-0.787)          |            221.696 (+-0.820)            |     2.995 (+-0.000)      |          110.533 (+-0.721)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        120.673 (+-0.628)        |          74.398 (+-0.226)          |            222.547 (+-0.218)            |     2.991 (+-0.000)      |          110.922 (+-0.567)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         81.171 (+-0.269)        |          73.679 (+-0.251)          |            142.400 (+-0.258)            |     1.933 (+-0.000)      |           76.351 (+-0.443)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         80.391 (+-0.336)        |          74.320 (+-0.171)          |            142.147 (+-0.709)            |     1.913 (+-0.000)      |           76.435 (+-0.390)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        121.020 (+-0.612)        |          73.723 (+-0.300)          |            178.589 (+-0.474)            |     2.422 (+-0.000)      |          110.797 (+-0.527)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        120.961 (+-0.460)        |          73.953 (+-0.363)          |            178.601 (+-0.382)            |     2.415 (+-0.000)      |          109.887 (+-0.417)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.540 (+-0.337)        |          74.419 (+-0.190)          |            1800.770 (+-0.552)           |     24.198 (+-0.000)     |           92.892 (+-0.220)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.401 (+-0.055)        |         1233.865 (+-0.348)         |            1463.572 (+-0.644)           |     1.186 (+-0.000)      |           92.596 (+-0.489)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        121.794 (+-0.353)        |          74.391 (+-0.260)          |            1806.567 (+-0.456)           |     24.285 (+-0.000)     |          110.970 (+-0.444)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        120.597 (+-0.850)        |         1233.630 (+-0.278)         |            1470.514 (+-1.586)           |     1.192 (+-0.000)      |          110.857 (+-0.506)         

Times are in microseconds (us).
