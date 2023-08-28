Description:

- 20230828-134459-affine-grid-sampler-PR
Torch version: 2.1.0a0+git52598e9
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
                                                                                                          |  Eager (2.1.0a0+git52598e9) PR  |  Compiled (2.1.0a0+git52598e9) PR  |  Compiled (2.1.0a0+gitcf76938) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitcf76938) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         38.010 (+-0.118)        |          51.466 (+-1.257)          |             47.867 (+-0.124)            |     0.930 (+-0.000)      |           33.654 (+-0.411)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         35.532 (+-0.236)        |          52.189 (+-0.093)          |             58.979 (+-0.206)            |     1.130 (+-0.000)      |           32.543 (+-0.198)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         38.187 (+-0.112)        |          47.892 (+-0.117)          |             45.833 (+-0.081)            |     0.957 (+-0.000)      |           33.752 (+-0.116)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         36.708 (+-0.244)        |          51.680 (+-0.104)          |             58.360 (+-0.108)            |     1.129 (+-0.000)      |           32.576 (+-0.751)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         24.201 (+-0.088)        |          27.451 (+-0.059)          |             27.937 (+-0.081)            |     1.018 (+-0.000)      |           24.367 (+-0.074)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         19.266 (+-0.105)        |          26.070 (+-0.085)          |             26.092 (+-0.054)            |     1.001 (+-0.000)      |           20.144 (+-0.064)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         24.293 (+-0.125)        |          26.085 (+-0.064)          |             26.575 (+-0.061)            |     1.019 (+-0.000)      |           24.515 (+-0.095)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         19.440 (+-0.075)        |          25.252 (+-0.059)          |             25.259 (+-0.051)            |     1.000 (+-0.000)      |           19.770 (+-0.070)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        114.900 (+-0.508)        |         113.416 (+-1.271)          |            248.679 (+-1.431)            |     2.193 (+-0.000)      |          114.609 (+-0.515)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        115.973 (+-0.555)        |         124.711 (+-1.596)          |            282.187 (+-2.418)            |     2.263 (+-0.000)      |          115.368 (+-0.652)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        111.730 (+-0.562)        |         110.914 (+-0.865)          |            253.899 (+-2.226)            |     2.289 (+-0.000)      |          111.285 (+-1.226)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        112.859 (+-0.487)        |         131.696 (+-1.298)          |            294.124 (+-1.963)            |     2.233 (+-0.000)      |          110.910 (+-0.969)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git52598e9) PR  |  Compiled (2.1.0a0+git52598e9) PR  |  Compiled (2.1.0a0+gitcf76938) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitcf76938) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |        228.811 (+-0.037)        |          92.990 (+-0.446)          |             92.648 (+-0.286)            |     0.996 (+-0.000)      |          228.274 (+-0.067)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |        222.107 (+-0.076)        |          93.247 (+-0.387)          |             92.528 (+-0.423)            |     0.992 (+-0.000)      |          221.922 (+-0.297)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        235.654 (+-0.055)        |          75.781 (+-0.566)          |            115.865 (+-0.419)            |     1.529 (+-0.000)      |          236.032 (+-0.111)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        226.752 (+-0.088)        |          76.312 (+-0.328)          |            116.468 (+-0.477)            |     1.526 (+-0.000)      |          226.950 (+-0.027)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |        225.540 (+-0.013)        |          75.638 (+-0.341)          |             72.621 (+-0.292)            |     0.960 (+-0.000)      |          225.937 (+-0.017)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |        217.425 (+-0.024)        |          75.484 (+-0.545)          |             73.518 (+-0.296)            |     0.974 (+-0.000)      |          217.793 (+-0.008)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        231.474 (+-0.020)        |          75.972 (+-0.339)          |             73.030 (+-0.387)            |     0.961 (+-0.000)      |          231.991 (+-0.184)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        223.408 (+-0.016)        |          75.622 (+-0.279)          |             73.542 (+-0.336)            |     0.973 (+-0.000)      |          223.893 (+-0.021)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        319.382 (+-0.023)        |         149.060 (+-0.190)          |            772.116 (+-0.266)            |     5.180 (+-0.000)      |          320.549 (+-0.387)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        319.987 (+-0.134)        |         154.443 (+-0.014)          |            797.651 (+-0.232)            |     5.165 (+-0.000)      |          320.665 (+-0.397)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        326.138 (+-0.439)        |         149.092 (+-0.036)          |            772.508 (+-0.259)            |     5.181 (+-0.000)      |          325.751 (+-0.398)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        326.024 (+-0.118)        |         154.452 (+-0.209)          |            797.756 (+-0.229)            |     5.165 (+-0.000)      |          326.870 (+-0.372)         

Times are in microseconds (us).
