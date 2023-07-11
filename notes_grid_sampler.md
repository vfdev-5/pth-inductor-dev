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