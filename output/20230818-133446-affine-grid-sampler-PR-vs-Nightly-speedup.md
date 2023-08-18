Description:

- 20230818-133446-affine-grid-sampler-PR
Torch version: 2.1.0a0+git0a6cfd9
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
                                                                                                          |  Eager (2.1.0a0+git0a6cfd9) PR  |  Compiled (2.1.0a0+git0a6cfd9) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.802 (+-0.067)         |          12.602 (+-0.077)          |             12.356 (+-0.054)            |     0.980 (+-0.000)      |           7.512 (+-0.325)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.763 (+-0.056)         |          11.709 (+-0.074)          |             14.186 (+-0.095)            |     1.212 (+-0.000)      |           7.511 (+-0.043)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.797 (+-0.064)         |          11.735 (+-0.057)          |             11.570 (+-0.055)            |     0.986 (+-0.000)      |           7.843 (+-0.051)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.706 (+-0.065)         |          11.367 (+-0.050)          |             13.615 (+-0.135)            |     1.198 (+-0.000)      |           7.809 (+-0.049)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.949 (+-0.228)         |          4.652 (+-0.023)           |             4.479 (+-0.020)             |     0.963 (+-0.000)      |           4.621 (+-0.026)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.569 (+-0.019)         |          5.077 (+-0.017)           |             4.819 (+-0.030)             |     0.949 (+-0.000)      |           4.179 (+-0.022)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.284 (+-0.023)         |          4.873 (+-0.023)           |             4.133 (+-0.062)             |     0.848 (+-0.000)      |           4.639 (+-0.061)          
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.558 (+-0.024)         |          4.629 (+-0.016)           |             4.607 (+-0.015)             |     0.995 (+-0.000)      |           4.485 (+-0.011)          
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.410 (+-0.121)        |          29.380 (+-0.086)          |             63.548 (+-0.410)            |     2.163 (+-0.000)      |           26.105 (+-0.115)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.527 (+-0.957)        |          30.471 (+-0.131)          |             70.633 (+-0.550)            |     2.318 (+-0.000)      |           26.554 (+-0.152)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.479 (+-0.122)        |          27.955 (+-0.085)          |             65.347 (+-0.085)            |     2.338 (+-0.000)      |           26.109 (+-0.177)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.668 (+-0.139)        |          30.261 (+-0.105)          |             75.345 (+-1.267)            |     2.490 (+-0.000)      |           26.464 (+-0.132)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git0a6cfd9) PR  |  Compiled (2.1.0a0+git0a6cfd9) PR  |  Compiled (2.1.0a0+git2932b0b) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+git2932b0b) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         94.407 (+-0.309)        |          91.874 (+-0.315)          |             94.940 (+-0.366)            |     1.033 (+-0.000)      |           99.678 (+-0.518)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         95.142 (+-0.561)        |          92.871 (+-0.639)          |             94.639 (+-0.285)            |     1.019 (+-0.000)      |           98.618 (+-1.816)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        115.714 (+-0.663)        |          73.662 (+-0.253)          |            116.332 (+-0.420)            |     1.579 (+-0.000)      |          126.473 (+-0.393)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        116.097 (+-0.567)        |          73.083 (+-0.269)          |            115.988 (+-0.397)            |     1.587 (+-0.000)      |          126.625 (+-0.585)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         81.825 (+-0.385)        |          75.039 (+-9.486)          |             73.989 (+-0.263)            |     0.986 (+-0.000)      |           89.014 (+-0.605)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         82.828 (+-1.691)        |          74.268 (+-6.061)          |             74.751 (+-0.319)            |     1.006 (+-0.000)      |           88.534 (+-0.718)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        116.550 (+-3.346)        |          74.788 (+-0.354)          |             74.502 (+-0.273)            |     0.996 (+-0.000)      |          127.675 (+-0.801)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        116.702 (+-1.107)        |          73.777 (+-0.226)          |             74.760 (+-0.486)            |     1.013 (+-0.000)      |          126.778 (+-0.608)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         93.024 (+-0.115)        |          73.374 (+-0.308)          |            1740.788 (+-0.537)           |     23.725 (+-0.000)     |           92.959 (+-0.045)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.755 (+-0.295)        |         120.955 (+-0.029)          |            1401.371 (+-0.814)           |     11.586 (+-0.000)     |           92.844 (+-0.356)         
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        116.953 (+-0.645)        |          72.971 (+-0.335)          |            1741.294 (+-0.518)           |     23.863 (+-0.000)     |          127.256 (+-0.646)         
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        116.987 (+-0.495)        |         120.725 (+-0.035)          |            1400.857 (+-0.709)           |     11.604 (+-0.000)     |          126.624 (+-0.434)         

Times are in microseconds (us).
