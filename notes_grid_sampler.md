## Grid Sampler PR perfs

- https://github.com/pytorch/pytorch/pull/104709


- Nightly
```
python -u perf_affine_grid_sampler.py output --tag=Nightly
```

- PR
```
python -u perf_affine_grid_sampler.py output --tag=PR

python -u perf_results_compute_speedup_v2.py output/20230706-135210-affine-grid-sampler-PR-vs-Nightly-speedup.md 'output/20230706-135210-affine-grid-sampler-PR.pkl' 'output/20230706-135210-affine-grid-sampler-Nightly.pkl' --compare "Compiled (2.1.0a0+gitd20adf4) PR;Compiled (2.1.0a0+gitd3ba890) Nightly;speed-up PR vs Nightly"
```


## Notes

```
Output filepath: output/20230711-104016-affine-grid-sampler-PR-case_cuda-bicubic_CL_bs_4.pkl
Torch version: 2.1.0a0+git3cfe3f5
Triton: 2.1.0+440fd1bf20
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_86,code=compute_86
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

[---------------------------------------------------------------------- Affine grid sampling, cuda ----------------------------------------------------------------------]
                                                                                                     |  Eager (2.1.0a0+git3cfe3f5) PR  |  Compiled (2.1.0a0+git3cfe3f5) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (4, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic   |        168.292 (+-0.086)        |         2714.932 (+-0.716)
      Input: (4, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic  |        174.364 (+-0.219)        |         2712.738 (+-0.780)

      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic   |         91.919 (+-0.025)        |         1241.695 (+-0.347)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic  |        104.637 (+-0.358)        |         1241.315 (+-0.307)

      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic   |         77.581 (+-0.238)        |          69.880 (+-0.201)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic  |        110.850 (+-0.441)        |          70.559 (+-0.219)

Times are in microseconds (us).
```


```
[------------------------------------------ Affine Grid Sampler 2d -------------------------------------]
                             |  Eager (Torch 2.1.0a0+git3cfe3f5)  |  Inductor (Torch 2.1.0a0+git3cfe3f5)
6 threads: ---------------------------------------------------------------------------------------------
      bicubic f32, CL, BS=2  |               179.6                |                  1233.0
      bicubic f32, CL, BS=1  |               137.4                |                  34.9
      bicubic f32, CF, BS=2  |               182.9                |                  51.2

Times are in microseconds (us).
```

- Using a hack for CL,BS>1 "convert to CF->compute->convert to CL":
```
[------------------------------------------------------------------------ Affine grid sampling, cuda ------------------------------------------------------------------------]
                                                                                                         |  Eager (2.1.0a0+git3cfe3f5) PR  |  Compiled (2.1.0a0+git3cfe3f5) PR
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (1, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic   |         78.622 (+-0.433)        |          69.296 (+-0.148)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic       |         77.182 (+-0.396)        |          69.791 (+-0.202)
      Input: (1, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic  |        111.933 (+-0.723)        |          69.114 (+-0.323)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic      |        112.676 (+-0.574)        |          69.705 (+-0.427)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic   |         92.645 (+-0.077)        |          70.014 (+-0.325)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic       |         92.504 (+-0.109)        |          87.434 (+-0.018)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic  |        113.532 (+-0.727)        |          70.176 (+-0.242)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic      |        117.368 (+-4.523)        |          87.290 (+-0.064)

Times are in microseconds (us).
```


- Compare with latest suggestions: https://github.com/pytorch/pytorch/pull/104709#discussion_r1259670263, https://github.com/pytorch/pytorch/pull/104709#discussion_r1259670673

- with suggestions (git commit is the same)
```
[------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.449 (+-0.028)         |          15.056 (+-0.022)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.492 (+-0.037)         |          10.868 (+-0.058)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         7.599 (+-0.051)         |          22.159 (+-0.237)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         7.537 (+-0.029)         |          10.463 (+-0.019)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.699 (+-0.012)         |          3.978 (+-0.013)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.195 (+-0.006)         |          4.299 (+-0.004)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.791 (+-0.017)         |          3.616 (+-0.016)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.227 (+-0.008)         |          4.046 (+-0.005)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.152 (+-0.098)        |          27.818 (+-0.049)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.436 (+-0.070)        |          28.402 (+-0.038)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.021 (+-0.224)        |          27.308 (+-0.058)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.284 (+-0.089)        |          27.995 (+-0.051)

Times are in milliseconds (ms).

[------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         90.083 (+-0.400)        |          83.094 (+-0.188)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         90.375 (+-0.279)        |          82.192 (+-0.237)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        112.404 (+-0.613)        |          71.363 (+-0.261)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        112.481 (+-0.487)        |          72.148 (+-0.279)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         78.045 (+-0.296)        |          72.013 (+-0.147)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         78.071 (+-0.495)        |          72.213 (+-1.946)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        112.790 (+-0.540)        |          71.584 (+-0.194)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        112.895 (+-0.598)        |          72.569 (+-0.188)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.627 (+-0.393)        |          72.257 (+-0.201)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.628 (+-0.553)        |          86.470 (+-0.033)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        112.755 (+-0.301)        |          72.379 (+-0.244)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        112.991 (+-0.421)        |          86.430 (+-0.022)
```

- without (git commit is the same)
```
[------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.374 (+-0.062)         |          15.610 (+-0.071)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.503 (+-0.029)         |          11.180 (+-0.028)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         7.532 (+-0.050)         |          15.756 (+-0.440)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         7.628 (+-0.027)         |          10.977 (+-0.064)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.602 (+-0.013)         |          4.489 (+-0.007)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.217 (+-0.008)         |          4.840 (+-0.006)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         4.994 (+-0.016)         |          4.154 (+-0.008)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.223 (+-0.011)         |          4.571 (+-0.005)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.243 (+-0.186)        |          28.218 (+-0.066)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.368 (+-0.084)        |          28.610 (+-0.115)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.213 (+-0.090)        |          27.737 (+-0.033)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.190 (+-0.122)        |          28.106 (+-0.043)

Times are in milliseconds (ms).

[------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1c48419) PR  |  Compiled (2.1.0a0+git1c48419) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         86.539 (+-0.331)        |          90.460 (+-0.368)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         86.822 (+-0.359)        |          89.588 (+-0.190)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        114.045 (+-0.523)        |          70.287 (+-0.097)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        114.186 (+-0.424)        |          70.044 (+-0.236)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         77.120 (+-0.382)        |          70.745 (+-0.219)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         76.958 (+-0.307)        |          70.229 (+-0.227)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        114.029 (+-0.424)        |          70.369 (+-0.280)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        114.433 (+-0.402)        |          70.751 (+-0.462)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.454 (+-0.347)        |          70.902 (+-0.379)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.412 (+-0.509)        |          86.872 (+-0.034)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        114.764 (+-0.461)        |          71.134 (+-0.248)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        115.110 (+-0.368)        |          86.609 (+-0.025)
```



- Using torch inductor decomposition for grid_sampler_2d and respecting memory_format

```
Num threads: 1

[-------------------------------------------------------------------- Affine grid sampling, cuda --------------------------------------------------------------------]
                                                                                                     |  Eager (2.1.0a0+git1afae24)   |  Compiled (2.1.0a0+git1afae24)
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic   |        81.432 (+-0.370)       |         76.534 (+-1.620)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic  |       114.701 (+-0.500)       |         76.528 (+-0.417)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic   |        92.963 (+-0.008)       |        119.427 (+-0.029)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic  |       117.821 (+-0.646)       |        119.463 (+-0.037)

Times are in microseconds (us).


[--------------------------------------------------------------------- Affine grid sampling, cuda --------------------------------------------------------------------]
                                                                                                      |  Eager (2.1.0a0+git1afae24)   |  Compiled (2.1.0a0+git1afae24)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest    |        75.615 (+-0.444)       |         74.913 (+-0.308)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest   |       108.713 (+-0.733)       |         75.046 (+-1.018)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear   |        85.254 (+-0.355)       |         92.434 (+-0.282)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear  |       109.187 (+-0.413)       |         76.057 (+-0.366)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic    |        76.196 (+-0.418)       |         75.984 (+-0.349)
      Input: (1, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic   |       109.573 (+-0.628)       |         76.043 (+-0.485)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest    |        78.769 (+-0.236)       |         83.504 (+-0.663)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest   |       112.718 (+-1.198)       |         81.733 (+-0.473)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear   |        89.260 (+-3.003)       |        102.118 (+-0.309)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear  |       111.613 (+-0.486)       |         80.993 (+-0.312)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic    |        92.742 (+-0.373)       |        120.507 (+-0.033)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic   |       111.569 (+-0.599)       |        120.261 (+-0.023)

Times are in microseconds (us).
```

- 15/08/2023 using _inductor/decomposition.py to keep memory format consistency:
```
Num threads: 1

[------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         7.556 (+-0.051)         |          15.812 (+-0.124)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         7.930 (+-0.064)         |          11.608 (+-0.071)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.011 (+-0.137)         |          16.390 (+-0.396)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.219 (+-0.048)         |          11.612 (+-0.063)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         4.755 (+-0.027)         |          4.733 (+-0.020)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.408 (+-0.014)         |          5.039 (+-0.019)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.189 (+-0.022)         |          4.577 (+-0.047)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.636 (+-0.017)         |          4.999 (+-0.018)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.259 (+-0.174)        |          28.555 (+-0.057)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.561 (+-0.133)        |          31.004 (+-0.074)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.220 (+-0.196)        |          28.134 (+-0.076)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.731 (+-0.156)        |          30.910 (+-0.096)

Times are in milliseconds (ms).

[------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         88.442 (+-0.297)        |          90.337 (+-0.284)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         89.268 (+-0.340)        |          90.220 (+-0.271)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        111.380 (+-0.377)        |          69.506 (+-0.510)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        111.241 (+-0.741)        |          69.978 (+-0.256)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         76.633 (+-0.335)        |          69.653 (+-0.242)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         76.809 (+-0.267)        |          69.450 (+-0.189)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        111.359 (+-0.524)        |          68.858 (+-0.231)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        111.302 (+-0.741)        |          69.647 (+-0.308)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.912 (+-0.188)        |          70.138 (+-0.759)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.652 (+-0.296)        |         118.250 (+-0.024)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        111.380 (+-0.519)        |          69.161 (+-0.393)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        111.876 (+-0.476)        |         118.075 (+-0.025)

Times are in microseconds (us).
```


- using `_expand_grid` arg for cpu input
```

Num threads: 1

[------------------------------------------------------------------------- Affine grid sampling, cpu -------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         8.296 (+-0.062)         |          12.264 (+-0.083)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         8.728 (+-0.055)         |          14.185 (+-0.060)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |         8.704 (+-0.060)         |          12.111 (+-0.074)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |         8.817 (+-0.056)         |          13.779 (+-0.082)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         5.014 (+-0.026)         |          4.607 (+-0.023)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         4.613 (+-0.014)         |          5.287 (+-0.028)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |         5.358 (+-0.015)         |          4.677 (+-0.025)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |         4.559 (+-0.015)         |          4.750 (+-0.018)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         26.318 (+-0.097)        |          64.344 (+-0.354)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         26.800 (+-0.104)        |          71.970 (+-0.631)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |         26.407 (+-1.060)        |          66.251 (+-0.322)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |         26.681 (+-0.113)        |          70.286 (+-0.576)

Times are in milliseconds (ms).

[------------------------------------------------------------------------- Affine grid sampling, cuda ------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git1afae24) PR  |  Compiled (2.1.0a0+git1afae24) PR
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |         88.251 (+-0.380)        |          90.057 (+-0.417)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |         87.581 (+-0.407)        |          89.818 (+-0.235)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |        107.544 (+-0.455)        |          70.633 (+-0.288)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |        108.117 (+-0.396)        |          71.327 (+-0.313)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |         75.589 (+-0.295)        |          70.709 (+-0.261)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |         75.671 (+-0.342)        |          71.604 (+-0.368)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |        107.936 (+-0.603)        |          71.359 (+-0.279)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |        108.138 (+-0.497)        |          71.197 (+-0.484)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |         92.869 (+-0.080)        |          71.131 (+-0.385)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |         92.625 (+-0.412)        |         120.649 (+-0.029)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |        107.871 (+-0.502)        |          70.607 (+-0.265)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |        107.647 (+-0.489)        |         120.449 (+-0.028)

Times are in microseconds (us).
```


- vs Nightly
```
[------------------------------------------------------------------------------ Affine grid sampling, cpu ------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git2932b0b) Nightly  |  Compiled (2.1.0a0+git2932b0b) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           7.442 (+-0.240)            |             11.853 (+-0.112)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           7.830 (+-0.084)            |             14.106 (+-0.902)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |           7.461 (+-0.469)            |             11.834 (+-0.059)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |           7.858 (+-0.045)            |             13.757 (+-0.088)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           4.844 (+-0.035)            |             4.565 (+-0.027)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           4.274 (+-0.014)            |             4.831 (+-0.015)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |           4.696 (+-0.044)            |             4.162 (+-0.032)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |           4.403 (+-0.020)            |             4.595 (+-0.020)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           26.057 (+-0.235)           |             63.771 (+-0.849)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           27.271 (+-0.606)           |             70.484 (+-1.120)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |           26.683 (+-0.858)           |             64.298 (+-0.942)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |           26.144 (+-0.694)           |             72.671 (+-0.659)

Times are in milliseconds (ms).

[------------------------------------------------------------------------------ Affine grid sampling, cuda -----------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git2932b0b) Nightly  |  Compiled (2.1.0a0+git2932b0b) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           90.914 (+-0.623)           |             88.847 (+-0.319)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           90.617 (+-0.789)           |             89.108 (+-0.330)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |          112.833 (+-0.356)           |            111.381 (+-0.512)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |          112.630 (+-0.475)           |            112.378 (+-0.415)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           79.251 (+-0.341)           |             70.719 (+-0.285)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           78.923 (+-0.391)           |             70.289 (+-0.250)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |          112.920 (+-0.573)           |             71.820 (+-0.344)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |          112.647 (+-0.546)           |             71.762 (+-1.035)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           92.608 (+-0.461)           |            1744.730 (+-0.649)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           92.585 (+-0.434)           |            1408.995 (+-0.797)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |          112.881 (+-0.677)           |            1743.939 (+-0.588)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |          112.789 (+-0.543)           |            1407.821 (+-0.713)

Times are in microseconds (us).

---


Num threads: 1

[------------------------------------------------------------------------------ Affine grid sampling, cpu ------------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git2932b0b) Nightly  |  Compiled (2.1.0a0+git2932b0b) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           7.512 (+-0.325)            |             12.356 (+-0.054)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           7.511 (+-0.043)            |             14.186 (+-0.095)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |           7.843 (+-0.051)            |             11.570 (+-0.055)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |           7.809 (+-0.049)            |             13.615 (+-0.135)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           4.621 (+-0.026)            |             4.479 (+-0.020)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           4.179 (+-0.022)            |             4.819 (+-0.030)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |           4.639 (+-0.061)            |             4.133 (+-0.062)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |           4.485 (+-0.011)            |             4.607 (+-0.015)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           26.105 (+-0.115)           |             63.548 (+-0.410)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           26.554 (+-0.152)           |             70.633 (+-0.550)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |           26.109 (+-0.177)           |             65.347 (+-0.085)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |           26.464 (+-0.132)           |             75.345 (+-1.267)

Times are in milliseconds (ms).

[------------------------------------------------------------------------------ Affine grid sampling, cuda -----------------------------------------------------------------------------]
                                                                                                          |  Eager (2.1.0a0+git2932b0b) Nightly  |  Compiled (2.1.0a0+git2932b0b) Nightly
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bilinear   |           99.678 (+-0.518)           |             94.940 (+-0.366)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bilinear       |           98.618 (+-1.816)           |             94.639 (+-0.285)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bilinear  |          126.473 (+-0.393)           |            116.332 (+-0.420)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bilinear      |          126.625 (+-0.585)           |            115.988 (+-0.397)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=nearest    |           89.014 (+-0.605)           |             73.989 (+-0.263)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=nearest        |           88.534 (+-0.718)           |             74.751 (+-0.319)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=nearest   |          127.675 (+-0.801)           |             74.502 (+-0.273)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=nearest       |          126.778 (+-0.608)           |             74.760 (+-0.486)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=True, mode=bicubic    |           92.959 (+-0.045)           |            1740.788 (+-0.537)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=True, mode=bicubic        |           92.844 (+-0.356)           |            1401.371 (+-0.814)
      Input: (2, 3, 345, 456) torch.float32, torch.contiguous_format, align_corners=False, mode=bicubic   |          127.256 (+-0.646)           |            1741.294 (+-0.518)
      Input: (2, 3, 345, 456) torch.float32, torch.channels_last, align_corners=False, mode=bicubic       |          126.624 (+-0.434)           |            1400.857 (+-0.709)

Times are in microseconds (us).
```

### Run Test checking strides between eager vs meta

```
pytest -vvv test/test_meta.py::TestMetaCUDA::test_dispatch_symbolic_meta_outplace_all_strides_grid_sampler_2d_cuda_float32
```


### Triton code dynamic vs static bs=2

```
root@server:/tmp/pth/inductor# python -u /tmp/pth/inductor/torch_compile_debug/grid_sampler_cuda_CL_bicubic_bs_1_2_run_2023_08_24_12_18_19_449318-pid_2816/torchinductor/model__14_inference_40.1/output_code_updated.py
0.001443
root@server:/tmp/pth/inductor# python -u /tmp/pth/inductor/torch_compile_debug/grid_sampler_cuda_CL_bicubic_bs_2_1_run_2023_08_24_12_10_00_590042-pid_2685/torchinductor/model___39.0/output_code_updated.py
0.001367
```

- Install the latest triton from pytorch requirements:
```
# setup.py:

        triton_pin_file = os.path.join(
            cwd, ".ci", "docker", "ci_commit_pins", triton_text_file
        )
        triton_version_file = os.path.join(cwd, ".ci", "docker", "triton_version.txt")

```

```
pip install 'pytorch-triton==2.1.0+440fd1bf20' --index-url https://download.pytorch.org/whl/nightly/cu117
pip install 'pytorch-triton==2.1.0+e6216047b8' --index-url https://download.pytorch.org/whl/nightly/cu117
```