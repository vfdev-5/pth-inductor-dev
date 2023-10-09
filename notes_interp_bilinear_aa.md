## Notes on upsample_bilinear_aa decomposition

### Debug channels last buffers

- Why intermediate buffer is CF even if input is CL -> indexing outputs CF
- Manually, put channels loop to the inner and set manually the buffer as CL can gain 2x speed-up
  - Why compiler can't see that?

```bash
python torch_compile_debug/upsample_bilinear_aa_cpu_CL_bs_1_f32_updated_run_2023_10_06_16_06_25_416667-pid_102451/torchinductor/model___9.0/output_code.py
```


Debug code:
```bash
TORCH_COMPILE_DEBUG=1 python -m debugpy --wait-for-client --listen 5678 check_interpolate_bilinear_aa.py
```

```
    def realize(self):
        if isinstance(
            self.data,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        ):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction)), type(self.data)
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),                            # <======== PROBLEM IS HERE THAT COMPUTED BUFFER IS CF AND NOT CL
            data=self.data,
        )
```

```
__init__ (/home/vfdev-5/pytorch/torch/_inductor/ir.py:2241)
realize (/home/vfdev-5/pytorch/torch/_inductor/ir.py:5442)
realize_hint (/home/vfdev-5/pytorch/torch/_inductor/ir.py:5464)
run_node (/home/vfdev-5/pytorch/torch/_inductor/graph.py:830)
run (/home/vfdev-5/pytorch/torch/fx/interpreter.py:138)
run (/home/vfdev-5/pytorch/torch/_inductor/graph.py:464)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:190)
fx_codegen_and_compile (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:535)
compile_fx_inner (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:340)
inner (/home/vfdev-5/usr/lib/python3.8/contextlib.py:75)
inner (/home/vfdev-5/pytorch/torch/_inductor/debug.py:297)
debug_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/repro/after_aot.py:80)
fw_compiler_base (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:1096)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:190)
aot_dispatch_base (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:1576)
aot_wrapper_synthetic_base (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:2395)
aot_wrapper_dedupe (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:2215)
create_aot_dispatcher_function (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:3432)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:190)
aot_module_simplified (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:3894)
compiler_fn (/home/vfdev-5/pytorch/torch/_dynamo/backends/common.py:55)
compile_fx (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:1159)
__call__ (/home/vfdev-5/pytorch/torch/__init__.py:1604)
debug_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/repro/after_dynamo.py:117)
call_user_compiler (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:1039)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:190)
compile_and_call_fx_graph (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:987)
inner (/home/vfdev-5/usr/lib/python3.8/contextlib.py:75)
compile_subgraph (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:859)
RETURN_VALUE (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:2213)
step (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:706)
run (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:743)
run (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:2103)
transform (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:451)
transform_code_object (/home/vfdev-5/pytorch/torch/_dynamo/bytecode_transformation.py:1028)
compile_inner (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:481)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:190)
_compile (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:559)
_convert_frame_assert (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:380)
```




### Debug cache_size perf issue

=> when cache_size is hit, no compilation is done anymore

```
torch._dynamo.config.cache_size_limit = 0

[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: -----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   615.3  |    603.9
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1088.0  |   1098.0
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   258.6  |    267.8
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1602.0  |   1616.5
```

```
torch._dynamo.config.cache_size_limit = 4

[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: -----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   685.3  |   2297.1
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1097.9  |   1678.5
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   255.5  |   2870.4
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1618.3  |   1941.1

Times are in microseconds (us).
```


### Perf results

```bash
python -u perf_interp_bilinear_aa.py
```

- 09/10/2023
```
[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: 1->4 ------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   602.4  |   2282.6
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1084.8  |   1665.9
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   254.0  |   2926.0
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1616.5  |   1971.3

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2607.5  |   9105.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4399.8  |   6787.1
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   925.2  |  12410.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6443.0  |   8522.9

1 threads: 4->1 -----------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   738.8  |   2347.7
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1102.8  |   1702.8
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   260.1  |   2988.9
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1626.1  |   1981.3

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2337.6  |   9176.5
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4366.1  |   6794.0
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   925.3  |  12592.9
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6548.2  |   8741.5

Times are in microseconds (us).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: 1->4 -----------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.8  |   150.2
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   28.6  |   150.9

      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.6  |   142.9
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   84.7  |   145.7

1 threads: 4->1 -----------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.9  |   139.1
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.8  |   153.3

      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.6  |   150.0
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   84.6  |   140.9


Times are in microseconds (us).
```



- 06/10/2023 - fixed results without cache size hit
```
[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: 4->1 ------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2684.1  |  11642.3
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  5142.0  |   9673.6
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   921.1  |  14819.6
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  7561.5  |  12628.3

      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   619.5  |   2330.9
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1110.9  |   1699.3
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   259.5  |   2905.2
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1631.5  |   1971.1

1 threads: 1->4 -----------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2279.0  |   9247.1
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4412.6  |   7156.9
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |  1164.1  |  14993.4
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6482.0  |   9506.0

      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   626.3  |   2338.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1104.7  |   1699.4
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   259.4  |   2909.1
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1628.6  |   1980.2


Times are in microseconds (us).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: 4->1 -----------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.5  |   150.9
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   84.6  |   154.1

      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.9  |   151.9
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.4  |   149.9

1 threads: 1->4 ----------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.8  |   147.2
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.0  |   158.3

      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.8  |   140.8
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   33.0  |   153.0


Times are in microseconds (us).
```


- 02/10/2023 - incorrect results due to cache size hit
```
[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: -----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2752.5  |   9436.2
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4468.5  |   6943.9
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   938.7  |  12452.1
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6534.3  |   9496.2

      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   590.6  |   2307.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1111.8  |   1706.1
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   263.6  |    274.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1638.1  |   1657.8

1 threads: -----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   680.0  |   2298.6
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1089.0  |   1670.6
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   260.3  |   2916.8
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1607.4  |   1949.2

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2747.8  |   9412.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4456.3  |   6914.5
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   944.1  |    963.6
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6602.6  |   6577.3

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.4  |   141.6
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.0  |   156.9

      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.9  |    24.5
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   33.6  |    49.4

1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.8  |   139.5
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   30.2  |   140.7

      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.6  |    43.0
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.1  |    85.4

Times are in microseconds (us).
```


- 28/09/2023
```
[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: -----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   696.9  |   2285.9
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   254.6  |   2851.5
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1097.5  |   1666.2
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1571.6  |   1938.2

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2878.1  |  10878.4
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   918.0  |    930.2
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4482.9  |   6808.3
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6339.2  |   6391.5

1 threads: -----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  3027.9  |  11655.1
      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   910.6  |  14586.6
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4534.1  |   6801.3
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6346.8  |  10062.5

      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   639.4  |   2299.0
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   256.7  |    268.4
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1136.7  |   1672.4
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1607.1  |   1593.1

Times are in microseconds (us).


[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   12.2  |   149.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   31.9  |   148.4

      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.6  |    42.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.0  |    85.3

1 threads: ----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.4  |   140.3
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.0  |   140.5

      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.9  |    24.2
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.3  |    44.3

Times are in microseconds (us).
```

Reported warnings:
```
[2023-09-28 14:50:06,548] torch._dynamo.convert_frame: [WARNING] torch._dynamo hit config.cache_size_limit (8)
[2023-09-28 14:50:06,548] torch._dynamo.convert_frame: [WARNING]    function: 'transform' (perf_interp_bilinear_aa.py:5)
[2023-09-28 14:50:06,548] torch._dynamo.convert_frame: [WARNING] to diagnose recompilation issues, set env variable TORCHDYNAMO_REPORT_GUARD_FAILURES=1 and also see https://pytorch.org/docs/master/compile/troubleshooting.html.
```


- 27/09/2023
```
Torch version: 2.2.0a0+gitea20db8
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.2.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[---------------------------------- Interpolate bilinear, AA=true, cpu ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   1.2   |     2.2
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   1.8   |     2.5
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   4.5   |     8.7
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   7.0   |    17.6

Times are in milliseconds (ms).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   12.9  |   162.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.7  |   162.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   14.8  |   169.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.3  |   168.9

Times are in microseconds (us).
```

### Run tests for upsampling AA=True

```
pytest -vvv test/test_decomp.py -k "interp or upsampl"
pytest -vvv test/inductor/test_torchinductor_opinfo.py -k "interp or upsampl"
pytest -vvv test/functorch/test_aotdispatch.py -k "interp or upsampl"
pytest -vvv test/functorch/test_ops.py -k "interp or upsampl"
pytest -vvv test/test_meta.py -k "interp or upsampl"
pytest -vvv test/test_proxy_tensor.py -k "interp or upsampl"
pytest -vvv test/test_decomp.py::HasDecompTest::test_has_decomposition
pytest -vvv test/functorch/test_vmap.py -k "interp or upsampl"


pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32
pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32

pytest -vvv test/functorch/test_vmap.py::TestVmapOperatorsOpInfoCUDA::test_op_has_batch_rule_nn_functional_interpolate_bicubic_cuda_float32
```



```
FAILED [1.4041s] test/functorch/test_vmap.py::TestVmapOperatorsOpInfoCUDA::test_op_has_batch_rule_nn_functional_interpolate_bicubic_cuda_float32 - RuntimeError: aten::_upsample_bicubic2d_aa.vec hit the vmap fallback which is currently disabled

FAILED [6.6944s] test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive__upsample_bilinear2d_aa_cpu_float32 - Failed: Unexpected success
FAILED [10.3795s] test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32 - Failed: Unexpected success
FAILED [7.5643s] test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32 - RuntimeError: Cannot call sizes() on tensor with symbolic sizes/strides
```


```
FAILED [17.0285s] test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32 - RuntimeError: Cannot call sizes() on tensor with symbolic sizes/strides
FAILED [2.5616s] test/functorch/test_aotdispatch.py::TestEagerFusionOpInfoCPU::test_aot_autograd_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32 - AssertionError: (ValueRanges(lower=0.500000000000000, upper=4.61168601842739e+18, is_bool=False), 0.5*shape_0)


FAILED [0.0315s] test/functorch/test_ops.py::TestOperatorsCPU::test_vmapjvpall_has_batch_rule_nn_functional_interpolate_bicubic_cpu_float32 - RuntimeError: aten::_upsample_bicubic2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0299s] test/functorch/test_ops.py::TestOperatorsCPU::test_vmapjvpall_has_batch_rule_nn_functional_interpolate_bilinear_cpu_float32 - RuntimeError: aten::_upsample_bilinear2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0348s] test/functorch/test_ops.py::TestOperatorsCPU::test_vmapvjp_has_batch_rule_nn_functional_interpolate_bicubic_cpu_float32 - RuntimeError: aten::_upsample_bicubic2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0339s] test/functorch/test_ops.py::TestOperatorsCPU::test_vmapvjp_has_batch_rule_nn_functional_interpolate_bilinear_cpu_float32 - RuntimeError: aten::_upsample_bilinear2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0328s] test/functorch/test_ops.py::TestOperatorsCUDA::test_vmapjvpall_has_batch_rule_nn_functional_interpolate_bicubic_cuda_float32 - RuntimeError: aten::_upsample_bicubic2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0326s] test/functorch/test_ops.py::TestOperatorsCUDA::test_vmapjvpall_has_batch_rule_nn_functional_interpolate_bilinear_cuda_float32 - RuntimeError: aten::_upsample_bilinear2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0591s] test/functorch/test_ops.py::TestOperatorsCUDA::test_vmapvjp_has_batch_rule_nn_functional_interpolate_bicubic_cuda_float32 - RuntimeError: aten::_upsample_bicubic2d_aa hit the vmap fallback which is currently disabled
FAILED [0.0558s] test/functorch/test_ops.py::TestOperatorsCUDA::test_vmapvjp_has_batch_rule_nn_functional_interpolate_bilinear_cuda_float32 - RuntimeError: aten::_upsample_bilinear2d_aa hit the vmap fallback which is currently disabled
```
