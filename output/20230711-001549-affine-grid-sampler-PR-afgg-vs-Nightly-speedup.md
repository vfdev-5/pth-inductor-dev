Description:

- 20230711-001549-affine-grid-sampler-PR-afgg
Torch version: 2.1.0a0+git3ed904e
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


[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cpu ------------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git3ed904e) PR-afgg  |  Compiled (2.1.0a0+git3ed904e) PR-afgg  |  Compiled (2.1.0a0+gitbcdd413) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitbcdd413) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           8.225 (+-0.067)            |             11.879 (+-0.035)            |             13.292 (+-0.113)            |     1.119 (+-0.000)      |           7.567 (+-0.037)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           8.400 (+-0.031)            |             14.443 (+-0.048)            |             15.798 (+-0.036)            |     1.094 (+-0.000)      |           7.489 (+-0.114)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |           8.346 (+-0.040)            |             11.773 (+-0.092)            |             12.964 (+-0.050)            |     1.101 (+-0.000)      |           7.623 (+-0.126)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |           8.482 (+-0.039)            |             13.598 (+-0.065)            |             15.386 (+-0.061)            |     1.131 (+-0.000)      |           7.647 (+-0.030)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           4.727 (+-0.014)            |             4.501 (+-0.009)             |             6.383 (+-0.038)             |     1.418 (+-0.000)      |           4.504 (+-0.028)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           4.336 (+-0.009)            |             4.847 (+-0.005)             |             6.393 (+-0.067)             |     1.319 (+-0.000)      |           4.270 (+-0.066)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |           4.432 (+-0.024)            |             4.182 (+-0.016)             |             6.426 (+-0.126)             |     1.537 (+-0.000)      |           4.780 (+-0.204)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |           4.361 (+-0.006)            |             4.593 (+-0.063)             |             6.283 (+-0.114)             |     1.368 (+-0.000)      |           4.315 (+-0.028)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           26.330 (+-0.101)           |             62.365 (+-0.240)            |             63.423 (+-0.464)            |     1.017 (+-0.000)      |           25.943 (+-0.299)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           26.562 (+-0.171)           |             70.276 (+-0.399)            |             71.692 (+-0.282)            |     1.020 (+-0.000)      |           26.143 (+-0.299)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |           26.608 (+-0.117)           |             65.074 (+-0.244)            |             65.312 (+-0.478)            |     1.004 (+-0.000)      |           25.810 (+-0.344)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |           26.544 (+-0.089)           |             72.216 (+-0.439)            |             71.560 (+-0.606)            |     0.991 (+-0.000)      |           26.105 (+-1.344)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------------ Affine grid sampling, cuda -----------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git3ed904e) PR-afgg  |  Compiled (2.1.0a0+git3ed904e) PR-afgg  |  Compiled (2.1.0a0+gitbcdd413) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitbcdd413) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           86.228 (+-0.318)           |             89.540 (+-0.196)            |            141.983 (+-0.551)            |     1.586 (+-0.000)      |           89.228 (+-0.300)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           86.324 (+-0.316)           |             89.952 (+-0.293)            |            141.351 (+-0.456)            |     1.571 (+-0.000)      |           89.257 (+-0.326)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |          108.712 (+-0.428)           |            110.125 (+-0.243)            |            228.701 (+-8.460)            |     2.077 (+-0.000)      |          112.709 (+-0.746)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |          109.276 (+-0.535)           |            110.025 (+-0.175)            |            230.542 (+-0.656)            |     2.095 (+-0.000)      |          112.994 (+-0.644)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           75.942 (+-0.773)           |             71.561 (+-0.197)            |            148.836 (+-0.244)            |     2.080 (+-0.000)      |           79.004 (+-0.973)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           75.206 (+-0.292)           |             71.511 (+-0.555)            |            149.036 (+-0.566)            |     2.084 (+-0.000)      |           78.681 (+-0.309)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |          110.005 (+-0.231)           |             70.790 (+-0.318)            |            184.097 (+-0.673)            |     2.601 (+-0.000)      |          113.252 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |          109.298 (+-0.226)           |             70.846 (+-0.202)            |            183.848 (+-1.566)            |     2.595 (+-0.000)      |          113.787 (+-0.943)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           92.575 (+-0.179)           |            1740.517 (+-0.727)           |            1793.246 (+-0.481)           |     1.030 (+-0.000)      |           92.416 (+-0.072)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           92.342 (+-0.021)           |            1400.480 (+-0.649)           |            1457.581 (+-0.599)           |     1.041 (+-0.000)      |           92.510 (+-0.557)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |          109.505 (+-0.427)           |            1739.787 (+-0.470)           |            1800.172 (+-0.422)           |     1.035 (+-0.000)      |          112.876 (+-0.498)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |          109.795 (+-1.309)           |            1400.236 (+-0.748)           |            1463.788 (+-0.566)           |     1.045 (+-0.000)      |          112.987 (+-0.358)         

Times are in microseconds (us).
