## Benchmarks

- `transform = ImageClassification(crop_size=224, antialias=False)`

```
python -u imagenet_inference_tf_perfs.py


Torch version: 2.1.0a0+git37359c3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DL$BKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno$type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-er$
or=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe$
uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.$
.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Torchvision version: 0.16.0a0


[------------ Imagenet inference image transformation -------------]
                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------
      Input (3, 345, 456), torch.uint8, CL    |   789.1  |   1938.7
      Input (3, 345, 456), torch.float32, CL  |  1674.1  |   2400.3
4 threads: ---------------------------------------------------------
      Input (3, 345, 456), torch.uint8, CL    |   590.4  |   1941.6
      Input (3, 345, 456), torch.float32, CL  |   786.3  |   1635.1

Times are in microseconds (us).
```

```
# With dynamic shapes, AA=False
#         label, sub_label, description, " : ", median, "µs (", min, max, ")", mean

Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Eager  :  792.0450439453125 µs ( 737.041015625 14587.9951171875 ) 833.041748046875
Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Compiled  :  2065.076171875 µs ( 1825.4140625 419261.5625 ) 2190.918212890625
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Eager  :  1653.6600341796875 µs ( 1444.0150146484375 2687.630126953125 ) 1781.6495361328125
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Compiled  :  2051.981201171875 µs ( 1713.984130859375 388151.53125 ) 2216.003662109375
```



- `transform = ImageClassification(crop_size=224, antialias=True)`

```
python -u imagenet_inference_tf_perfs.py


Torch version: 2.1.0a0+git37359c3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DL$BKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno$type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-er$
or=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe$
uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.$
.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Torchvision version: 0.16.0a0


[------------ Imagenet inference image transformation -------------]
                                              |  Eager   |  Compiled
1 threads: ---------------------------------------------------------
      Input (3, 345, 456), torch.uint8, CL    |   866.1  |    558.5
      Input (3, 345, 456), torch.float32, CL  |  2390.8  |   1328.3
4 threads: ---------------------------------------------------------
      Input (3, 345, 456), torch.uint8, CL    |   758.4  |    662.7
      Input (3, 345, 456), torch.float32, CL  |  1256.6  |    760.1

Times are in microseconds (us).
```

```
[-------------- Imagenet inference image transformation --------------]
                                                  |  Eager  |  Compiled
1 threads: ------------------------------------------------------------
      Input (3, randH, randW), torch.uint8, CL    |   4.4   |    4.3
      Input (3, randH, randW), torch.float32, CL  |   5.9   |    5.7

Times are in milliseconds (ms).
```

```
# Without dynamic shapes, AA=True
#         label, sub_label, description, " : ", median, "µs (", min, max, ")", mean

Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Eager  :  852.72802734375 µs ( 759.4120483398438 4161.91943359375 ) 853.50439453125
Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Compiled  :  2236013.5 µs ( 2140806.75 2424274.75 ) 2253452.5
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Eager  :  2310.567138671875 µs ( 1831.7840576171875 10400.8720703125 ) 2292.603759765625
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Compiled  :  2219111.0 µs ( 2151038.25 2299802.5 ) 2216932.0
```

```
# With dynamic shapes, AA=True
#         label, sub_label, description, " : ", median, "µs (", min, max, ")", mean

Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Eager  :  853.51904296875 µs ( 761.9970092773438 4995.13037109375 ) 856.0342407226562
Imagenet inference image transformation Input (3, randH, randW), torch.uint8, CL Compiled  :  736.8090209960938 µs ( 612.9760131835938 419997.8125 ) 771.1366577148438
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Eager  :  2309.97607421875 µs ( 1858.6141357421875 3924.150146484375 ) 2293.649169921875
Imagenet inference image transformation Input (3, randH, randW), torch.float32, CL Compiled  :  1851.7120361328125 µs ( 1350.3970947265625 344464.28125 ) 1911.6771240234375
```





- Bicubic op

```
python perf_interp_bicubic.py

Torch version: 2.1.0a0+git37359c3
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[--------------------------------- Interpolate bicubic ---------------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   712.1  |   3727.2
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  2138.1  |   2234.8
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   426.4  |   3921.5
      Input (3, 345, 456), torch.float32, torch.channels_last      |  2148.4  |   2270.9
4 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.uint8, torch.contiguous_format    |   738.5  |   3745.1
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   635.4  |   2242.6
      Input (3, 345, 456), torch.uint8, torch.channels_last        |   423.6  |   3938.1
      Input (3, 345, 456), torch.float32, torch.channels_last      |   653.0  |   3138.8

Times are in microseconds (us).
```


- Improved bilinear decomposition:
```
device = CPU, outsize = 270
[---------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  1450.1  |   935.8
      Input (3, 345, 456), torch.float32, torch.channels_last      |  1205.8  |   949.6


device = CPU, outsize = 224
[---------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  1143.8  |   639.7
      Input (3, 345, 456), torch.float32, torch.channels_last      |   904.7  |   655.5

Times are in microseconds (us).


device = CUDA, outsize = 270
[--------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   14.0  |    38.3
      Input (3, 345, 456), torch.float32, torch.channels_last      |   15.5  |    42.0

Times are in microseconds (us).
```

vs old bilinear decomposition
```
device = CPU, outsize = 270
[--------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   1.5   |    1.5
      Input (3, 345, 456), torch.float32, torch.channels_last      |   1.6   |    1.2

Times are in milliseconds (ms).

device = CPU, outsize = 224
[---------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |  1079.1  |   848.1
      Input (3, 345, 456), torch.float32, torch.channels_last      |   808.0  |   867.7

Times are in microseconds (us).

device = CUDA, outsize = 270
[--------------------------- Interpolate bilinear, AA=false ---------------------------]
                                                                   |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------------
      Input (3, 345, 456), torch.float32, torch.contiguous_format  |   14.1  |    38.2
      Input (3, 345, 456), torch.float32, torch.channels_last      |   15.5  |    40.8

Times are in microseconds (us).
```



```
[----------------------------------------------------------------------------------------- Bilinear upsampling 2d -------------------------------]
                                     |  Eager (Torch 2.1.0a0+git37359c3)  |  Inductor (3 blocks)  |  Inductor (1 block)  |  Inductor (1 block v2)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------
      3, (345, 456) -> 224, f32, CL  |               819.9                |    850.3              |     614.5            |        636.2


Times are in microseconds (us).
```