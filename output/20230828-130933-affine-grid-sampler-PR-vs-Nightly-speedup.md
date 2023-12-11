Description:

- 20230828-130933-affine-grid-sampler-PR
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
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         37.987 (+-0.105)        |          47.825 (+-0.075)          |             47.867 (+-0.124)            |     1.001 (+-0.000)      |           33.654 (+-0.411)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         35.563 (+-0.103)        |          52.158 (+-0.104)          |             58.979 (+-0.206)            |     1.131 (+-0.000)      |           32.543 (+-0.198)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         38.219 (+-0.506)        |          46.043 (+-0.971)          |             45.833 (+-0.081)            |     0.995 (+-0.000)      |           33.752 (+-0.116)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         35.347 (+-0.116)        |          52.029 (+-0.125)          |             58.360 (+-0.108)            |     1.122 (+-0.000)      |           32.576 (+-0.751)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         24.304 (+-0.070)        |          27.507 (+-0.057)          |             27.937 (+-0.081)            |     1.016 (+-0.000)      |           24.367 (+-0.074)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         19.306 (+-0.072)        |          26.129 (+-0.085)          |             26.092 (+-0.054)            |     0.999 (+-0.000)      |           20.144 (+-0.064)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         24.262 (+-0.103)        |          26.130 (+-0.063)          |             26.575 (+-0.061)            |     1.017 (+-0.000)      |           24.515 (+-0.095)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         19.406 (+-0.139)        |          25.253 (+-0.096)          |             25.259 (+-0.051)            |     1.000 (+-0.000)      |           19.770 (+-0.070)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        114.398 (+-1.256)        |         113.026 (+-0.663)          |            248.679 (+-1.431)            |     2.200 (+-0.000)      |          114.609 (+-0.515)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        115.565 (+-0.520)        |         124.275 (+-0.512)          |            282.187 (+-2.418)            |     2.271 (+-0.000)      |          115.368 (+-0.652)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        111.645 (+-2.343)        |         111.035 (+-0.483)          |            253.899 (+-2.226)            |     2.287 (+-0.000)      |          111.285 (+-1.226)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        112.581 (+-0.818)        |         127.205 (+-0.961)          |            294.124 (+-1.963)            |     2.312 (+-0.000)      |          110.910 (+-0.969)         

Times are in milliseconds (ms).

[------------------------------------------------------------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gite91e17c) PR  |  Compiled (2.1.0a0+gitcf76938) Nightly  |  speed-up PR vs Nightly  |  Eager (2.1.0a0+gitcf76938) Nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |        228.888 (+-0.015)        |          94.357 (+-0.336)          |             92.648 (+-0.286)            |     0.982 (+-0.000)      |          228.274 (+-0.067)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |        222.054 (+-0.019)        |          94.575 (+-0.386)          |             92.528 (+-0.423)            |     0.978 (+-0.000)      |          221.922 (+-0.297)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        235.818 (+-0.111)        |          75.655 (+-0.349)          |            115.865 (+-0.419)            |     1.531 (+-0.000)      |          236.032 (+-0.111)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        227.102 (+-0.013)        |          76.492 (+-1.292)          |            116.468 (+-0.477)            |     1.523 (+-0.000)      |          226.950 (+-0.027)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |        225.699 (+-0.016)        |          75.978 (+-0.349)          |             72.621 (+-0.292)            |     0.956 (+-0.000)      |          225.937 (+-0.017)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |        217.889 (+-0.120)        |          75.592 (+-0.303)          |             73.518 (+-0.296)            |     0.973 (+-0.000)      |          217.793 (+-0.008)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        232.202 (+-0.272)        |          76.273 (+-0.283)          |             73.030 (+-0.387)            |     0.957 (+-0.000)      |          231.991 (+-0.184)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        224.150 (+-0.194)        |          76.652 (+-0.281)          |             73.542 (+-0.336)            |     0.959 (+-0.000)      |          223.893 (+-0.021)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |        319.565 (+-0.188)        |         148.866 (+-0.026)          |            772.116 (+-0.266)            |     5.187 (+-0.000)      |          320.549 (+-0.387)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |        319.981 (+-0.117)        |         154.760 (+-0.147)          |            797.651 (+-0.232)            |     5.154 (+-0.000)      |          320.665 (+-0.397)         
      Input: (8, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        325.612 (+-0.060)        |         149.086 (+-0.019)          |            772.508 (+-0.259)            |     5.182 (+-0.000)      |          325.751 (+-0.398)         
      Input: (8, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        326.192 (+-0.126)        |         154.744 (+-0.169)          |            797.756 (+-0.229)            |     5.155 (+-0.000)      |          326.870 (+-0.372)         

Times are in microseconds (us).
