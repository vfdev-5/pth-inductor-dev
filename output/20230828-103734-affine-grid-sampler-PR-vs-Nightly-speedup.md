Description:

- 20230828-103734-affine-grid-sampler-PR
Torch version: 2.1.0a0+gite91e17c
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+e6216047b8

- 20230825-171504-affine-grid-sampler-Nightly
Torch version: 2.1.0a0+gitcf76938
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 

Triton version: 2.1.0+e6216047b8


[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gitcf76938) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitcf76938) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         37.965 (+-0.158)        |          47.825 (+-0.112)          |             47.867 (+-0.124)            |     1.001 (+-0.000)      |           33.654 (+-0.411)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         35.496 (+-0.136)        |          59.323 (+-6.264)          |             58.979 (+-0.206)            |     0.994 (+-0.000)      |           32.543 (+-0.198)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         38.051 (+-0.823)        |          45.873 (+-1.999)          |             45.833 (+-0.081)            |     0.999 (+-0.000)      |           33.752 (+-0.116)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         35.823 (+-0.737)        |          57.696 (+-1.408)          |             58.360 (+-0.108)            |     1.012 (+-0.000)      |           32.576 (+-0.751)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         24.286 (+-0.180)        |          27.516 (+-0.119)          |             27.937 (+-0.081)            |     1.015 (+-0.000)      |           24.367 (+-0.074)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         19.290 (+-0.075)        |          26.053 (+-0.075)          |             26.092 (+-0.054)            |     1.002 (+-0.000)      |           20.144 (+-0.064)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         24.321 (+-0.324)        |          26.119 (+-0.062)          |             26.575 (+-0.061)            |     1.017 (+-0.000)      |           24.515 (+-0.095)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         19.326 (+-0.169)        |          25.175 (+-0.060)          |             25.259 (+-0.051)            |     1.003 (+-0.000)      |           19.770 (+-0.070)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        114.167 (+-0.748)        |         112.998 (+-1.102)          |            248.679 (+-1.431)            |     2.201 (+-0.000)      |          114.609 (+-0.515)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        115.429 (+-0.680)        |         124.432 (+-3.060)          |            282.187 (+-2.418)            |     2.268 (+-0.000)      |          115.368 (+-0.652)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        111.310 (+-0.942)        |         111.081 (+-0.382)          |            253.899 (+-2.226)            |     2.286 (+-0.000)      |          111.285 (+-1.226)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        111.065 (+-0.974)        |         126.718 (+-0.787)          |            294.124 (+-1.963)            |     2.321 (+-0.000)      |          110.910 (+-0.969)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gitcf76938) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitcf76938) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |        228.923 (+-0.072)        |          92.343 (+-0.416)          |             92.648 (+-0.286)            |     1.003 (+-0.000)      |          228.274 (+-0.067)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |        221.860 (+-0.017)        |          91.910 (+-0.364)          |             92.528 (+-0.423)            |     1.007 (+-0.000)      |          221.922 (+-0.297)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        235.933 (+-0.012)        |          74.382 (+-0.306)          |            115.865 (+-0.419)            |     1.558 (+-0.000)      |          236.032 (+-0.111)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        226.925 (+-0.012)        |          73.764 (+-0.352)          |            116.468 (+-0.477)            |     1.579 (+-0.000)      |          226.950 (+-0.027)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |        225.716 (+-0.162)        |          74.075 (+-0.272)          |             72.621 (+-0.292)            |     0.980 (+-0.000)      |          225.937 (+-0.017)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |        217.719 (+-0.009)        |          74.269 (+-0.234)          |             73.518 (+-0.296)            |     0.990 (+-0.000)      |          217.793 (+-0.008)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        231.900 (+-0.017)        |          74.459 (+-0.259)          |             73.030 (+-0.387)            |     0.981 (+-0.000)      |          231.991 (+-0.184)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        223.935 (+-0.170)        |          74.114 (+-0.294)          |             73.542 (+-0.336)            |     0.992 (+-0.000)      |          223.893 (+-0.021)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        319.715 (+-0.275)        |         148.777 (+-0.036)          |            772.116 (+-0.266)            |     5.190 (+-0.000)      |          320.549 (+-0.387)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        320.073 (+-0.212)        |         154.391 (+-0.014)          |            797.651 (+-0.232)            |     5.166 (+-0.000)      |          320.665 (+-0.397)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        325.649 (+-0.184)        |         148.817 (+-0.030)          |            772.508 (+-0.259)            |     5.191 (+-0.000)      |          325.751 (+-0.398)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        326.335 (+-0.193)        |         154.444 (+-0.010)          |            797.756 (+-0.229)            |     5.165 (+-0.000)      |          326.870 (+-0.372)         

Times are in microseconds (us).
