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