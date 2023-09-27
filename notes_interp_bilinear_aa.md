## Notes on upsample_bilinear_aa decomposition


### Perf results

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

pytest -vvv test/test_decomp.py::HasDecompTest::test_has_decomposition


pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bilinear_cpu_float32
pytest -vvv test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive_nn_functional_interpolate_bicubic_cpu_float32
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
