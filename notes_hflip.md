## Debugging

```
TORCH_COMPILE_DEBUG=1 python -m debugpy --wait-for-client --listen 5678 check_hflip.py
```

Put breakpoints into `cpp.py`, `LoopLevel.lines()`


- Why vectorized `CppVecKernel` is not taken to perform load and store?
  - `scheduler.py:codegen()`
    - `CppKernelProxy.codegen_nodes`






## Some benchmarks


- 02/06/2023
```

Torch version: 2.1.0a0+git72cdbf6
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKIN
ETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-lim
its -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style
-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -f
no-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_
CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

[------------------------------------------- HFlip measurements -------------------------------------------]
                                           |  Torch 2.1.0a0+git72cdbf6  |  Torch inductor 2.1.0a0+git72cdbf6
1 threads: -------------------------------------------------------------------------------------------------
      3, 256, torch.uint8, channels_last   |            22.4            |                 85.0
      3, 256, torch.uint8, channels_first  |            11.4            |                 26.8

Times are in microseconds (us).


Num threads: 1

[------------------------------------------- HFlip measurements -------------------------------------------]
                                           |  Torch 2.1.0a0+git72cdbf6  |  Torch inductor 2.1.0a0+git72cdbf6
1 threads: -------------------------------------------------------------------------------------------------
      3, 224, torch.uint8, channels_last   |            17.2            |                 67.6
      3, 256, torch.uint8, channels_last   |            21.3            |                 97.0
      3, 224, torch.uint8, channels_first  |             9.3            |                 22.8
      3, 256, torch.uint8, channels_first  |            13.2            |                 24.4

Times are in microseconds (us).
```



- 12/06/2023
```
$ python check_perfs_flip.py

/pytorch/torch/utils/benchmark/utils/timer.py:16: UserWarning: 'has_cuda' is deprecated, please use 'torch.backends.cuda.is_built()'
  if torch.has_cuda and torch.cuda.is_available():

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

Num threads: 1


[------------------------------------------- HFlip measurements -------------------------------------------]
                                           |  Torch 2.1.0a0+git37359c3  |  Torch inductor 2.1.0a0+git37359c3
1 threads: -------------------------------------------------------------------------------------------------
      3, 224, torch.uint8, channels_last   |            17.2            |                 68.1
      3, 224, torch.uint8, channels_first  |             9.2            |                 23.0

Times are in microseconds (us).

[------------------------------------------- VFlip measurements -------------------------------------------]
                                           |  Torch 2.1.0a0+git37359c3  |  Torch inductor 2.1.0a0+git37359c3
1 threads: -------------------------------------------------------------------------------------------------
      3, 224, torch.uint8, channels_last   |            9.8             |                 73.3
      3, 224, torch.uint8, channels_first  |            9.3             |                 25.7

Times are in microseconds (us).
```


- 13/06/2023

```
python -u /tmp/pth/inductor/torch_compile_debug/run_2023_06_13_01_21_29_423950-pid_63896/torchinductor/model___26.0/output_code.py

flip -1 measurements 3, 224, uint8, CF Torch 2.1.0a0+git37359c3  :  16.842 µs
flip -1 measurements 3, 224, uint8, CF Torch inductor 2.1.0a0+git37359c3  :  9.378 µs
```

```
python -u /tmp/pth/inductor/torch_compile_debug/run_2023_06_12_22_53_07_612601-pid_53061/torchinductor/model___26.0/output_code.py

flip -2 measurements 3, 224, uint8, CF Torch 2.1.0a0+git37359c3  :  19.377 µs
flip -2 measurements 3, 224, uint8, CF Torch inductor 2.1.0a0+git37359c3  :  9.017 µs
```

```
python -u check_perfs_flip2.py

HFlip measurements 3, 224, torch.uint8, channels_first Torch 2.1.0a0+git37359c3  :  11.501 µs
HFlip measurements 3, 224, torch.uint8, channels_first Torch inductor 2.1.0a0+git37359c3  :  26.139 µs

VFlip measurements 3, 224, torch.uint8, channels_first Torch 2.1.0a0+git37359c3  :  9.648 µs
VFlip measurements 3, 224, torch.uint8, channels_first Torch inductor 2.1.0a0+git37359c3  :  22.963 µs
```






## Various logs

```
TORCH_COMPILE_DEBUG=1 python check_hflip.py
```

- Contiguous CF
```
# IR post fusion:

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 150528})]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep('arg0_1', 224*c0 - c1 + 223, {c0: 672, c1: 224})]
buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
buf0.group.device = cpu
buf0.group.iteration = ((672, 224), ())
buf0.sizes = ([672, 224], [])
class buf0_loop_body:
    var_ranges = {z0: 672, z1: 224}
    index0 = 224*z0 - z1 + 223
    index1 = 224*z0 + z1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf0', get_index_1, load, None)
        return store
```

- Contiguous CL
```
# IR post fusion

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', 672*c0 + c1 + 3*c2, {c0: 224, c1: 3, c2: 224})]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep('arg0_1', 672*c0 + c1 - 3*c2 + 669, {c0: 224, c1: 3, c2: 224})]
buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
buf0.group.device = cpu
buf0.group.iteration = ((224, 3, 224), ())
buf0.sizes = ([224, 3, 224], [])
class buf0_loop_body:
    var_ranges = {z0: 224, z1: 3, z2: 224}
    index0 = 672*z0 + z1 - 3*z2 + 669
    index1 = 672*z0 + z1 + 3*z2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf0', get_index_1, load, None)
        return store
```

- Sliced CF
```
x = torch.randint(0, 256, size=(1, 3, 224 + 21, 224 + 22), dtype=torch.uint8)
x = x[:, :, 9:-12, 12:-10]
```

```
# IR post fusion

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 150528})]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep('arg0_1', 60270*c0 + 246*c1 - c2 + 223, {c0: 3, c1: 224, c2: 224})]
buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
buf0.group.device = cpu
buf0.group.iteration = ((3, 224, 224), ())
buf0.sizes = ([3, 224, 224], [])
class buf0_loop_body:
    var_ranges = {z0: 3, z1: 224, z2: 224}
    index0 = 60270*z0 + 246*z1 - z2 + 223
    index1 = 50176*z0 + 224*z1 + z2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf0', get_index_1, load, None)
        return store
```