## Notes on upsample_bilinear_aa decomposition

### Questions and feedback from Mario

```
python -u perf_interp_bilinear_aa_custom.py

[---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   11.0  |    63.2
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   42.4  |    74.4

Times are in microseconds (us).
```

- Check contig load
- Check if RBLOCK reduction is contig
- Why reduction hint is DEFAULT and not INNER or OUTER... ?
- Do bisecting to see where the slowest part of the code vs cuda impl
- Write manual triton kernel equivalent to cuda impl


Benchmarking triton compiled code vs Aten
```
python /tmp/pth/inductor/torch_compile_debug/upsample_aa_bs1_single_kernel_and_WIP_run_2023_10_23_13_33_12_638970-pid_93744/torchinductor/model___9.0/output_code_bench.py

[----------------------------- Interpolate bilinear, AA=true, cuda ------------------------------]
                                                             |  Eager  |  Compiled  |  Compiled v2
1 threads: ---------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> 256, 256, torch.float32, CF  |   11.8  |    22.9    |      22.6


python -u /tmp/pth/inductor/torch_compile_debug/upsample_aa_bs4_single_kernel_and_WIP_run_2023_10_20_10_20_26_241925-pid_74696/torchinductor/model___9.0/output_code_bench.py

[----------------------------- Interpolate bilinear, AA=true, cuda ------------------------------]
                                                             |  Eager  |  Compiled  |  Compiled v2
1 threads: ---------------------------------------------------------------------------------------
      Input (4, 3, 500, 400) -> 256, 256, torch.float32, CF  |   42.7  |    35.8    |      33.9


Times are in microseconds (us).
```


Compared to nearest
```
python -u /tmp/pth/inductor/torch_compile_debug/upsample_nearest_bs1_and_bench_run_2023_10_23_16_27_11_509611-pid_106293/torchinductor/model___9.0/output_code_bench.py

[-------------------------- Interpolate nearest, cuda ---------------------------]
                                                             |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------
      Input (1, 3, 500, 400) -> 256, 256, torch.float32, CF  |   10.4  |    21.2


python -u /tmp/pth/inductor/torch_compile_debug/upsample_nearest_bs1_and_bench_run_2023_10_23_16_27_11_509611-pid_106293/torchinductor/model___9.0/output_code_bench_bs4.py
[-------------------------- Interpolate nearest, cuda ---------------------------]
                                                             |  Eager  |  Compiled
1 threads: -----------------------------------------------------------------------
      Input (4, 3, 500, 400) -> 256, 256, torch.float32, CF  |   15.4  |    22.7

Times are in microseconds (us).
```

or
```
python -u perf_interp_nearest_custom.py

[--------------------------------------- Interpolate nearest, cuda ---------------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |    9.4  |    61.9
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   15.4  |    70.1

Times are in microseconds (us).
```


### Check

> Most of the speed-up is already there with 512, so let's add 512. Put up a PR and trigger a perf run from https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml
> Click on run workflow -> your branch with the default settings
> once it finishes, you'll be able to see the results in https://hud.pytorch.org/benchmark/compilers

- https://github.com/pytorch/pytorch/pull/111656
- https://github.com/pytorch/pytorch/actions/runs/6589467661
- https://hud.pytorch.org/benchmark/compilers


### Other decomp versions


```python
@register_decomposition(aten._upsample_bilinear2d_aa.default)
@aten._upsample_bilinear2d_aa.default.py_impl(DispatchKey.Autograd)
@pw_cast_for_opmath
def _upsample_bilinear2d_aa(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    _upsample_2d_common_check(input, output_size)

    in_h, in_w = input.shape[-2:]
    interp_size = 2  # bilinear

    memory_format = utils.suggest_memory_format(input)

    src_x_min, x_weights = _compute_indices_weights_aa(
        output_size[1],
        in_w,
        scales_w,
        interp_size,
        align_corners,
        device=input.device,
    )

    src_y_min, y_weights = _compute_indices_weights_aa(
        output_size[0],
        in_h,
        scales_h,
        interp_size,
        align_corners,
        device=input.device,
    )

    x_max_interp_size = x_weights.shape[-1]
    y_max_interp_size = y_weights.shape[-1]

    kx = torch.arange(x_max_interp_size, device=input.device)
    ky = torch.arange(y_max_interp_size, device=input.device)

    src_x_min = src_x_min.unsqueeze(dim=-1)
    src_y_min = src_y_min.unsqueeze(dim=-1)

    x_indices = torch.clamp(src_x_min + kx, max=in_w - 1)
    y_indices = torch.clamp(src_y_min + ky, max=in_h - 1)

    y_indices = y_indices.view(*y_indices.shape, 1, 1)
    input_selected = input[:, :, y_indices, x_indices]

    # # This is slow:
    # two kernels: 1) create yx_weights, 2) matmul yx_weights, input_selected
    # [---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
    #                                                                                     |  Eager  |  Compiled
    # 1 threads: ------------------------------------------------------------------------------------------------
    #     Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   29.3  |   165.3
    #     Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   12.1  |   152.1
    # y_weights = y_weights.view(*y_weights.shape, 1, 1)
    # yx_weights = y_weights * x_weights.unsqueeze(dim=0)
    # output = (yx_weights * input_selected).sum(dim=(-3, -1))

    # input_selected.shape: (N, C, oH, y_max_interp_size, oW, x_max_interp_size) -> (N, C, oH, oW, y_max_interp_size, x_max_interp_size)
    # # y_weights.shape: (oH, y_max_interp_size) -> (oH, 1, y_max_interp_size)
    # # x_weights.shape: (oW, x_max_interp_size) -> (oW, 1, x_max_interp_size)
    # [---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
    #                                                                                       |  Eager  |  Compiled
    # 1 threads: ------------------------------------------------------------------------------------------------
    #       Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   84.5  |   167.5
    #       Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   43.1  |   160.6
    #       Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   30.2  |    81.4
    #       Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   12.1  |    79.5
    # input_selected = input_selected.transpose(-3, -2)
    # x_weights = x_weights.view(output_size[1], 1, x_max_interp_size)
    # y_weights = y_weights.view(output_size[0], 1, y_max_interp_size)

    # output = (x_weights * input_selected).sum(dim=-1)
    # output = (y_weights * output).sum(dim=-1)

    # # Two kernels:
    # # 1) to compute (x_weights.unsqueeze(0) * input_selected).sum(dim=-1)
    # # 2) to compute final output
    # output = (x_weights.unsqueeze(0) * input_selected).sum(dim=-1)
    # output = (y_weights.unsqueeze(-1) * output).sum(dim=-2)

    # # One kernel faster on N=1 and slower with N=4 on cuda
    # [---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
    #                                                                                       |  Eager  |  Compiled
    # 1 threads: ------------------------------------------------------------------------------------------------
    #       Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   84.4  |   216.9
    #       Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   43.0  |   209.5
    #       Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   29.3  |    60.4
    #       Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   12.0  |    60.9
    # y_weights = y_weights.view(output_size[0], y_max_interp_size, 1, 1)
    # x_weights = x_weights.view(1, output_size[1], x_max_interp_size)
    # output = (y_weights * (x_weights * input_selected)).sum(dim=(-1, -3))

    # SAME AS PREVIOUS
    # y_weights = y_weights.view(output_size[0], y_max_interp_size, 1, 1).expand(
    #     output_size[0], y_max_interp_size, output_size[1], 1
    # )
    # x_weights = x_weights.view(1, 1, output_size[1], x_max_interp_size).expand(
    #     output_size[0], 1, output_size[1], x_max_interp_size
    # )
    # output = (y_weights * (x_weights * input_selected)).sum(dim=(-1, -3))

    # SAME AS PREVIOUS
    # # input_selected.shape: (N, C, oH, y_max_interp_size, oW, x_max_interp_size) -> (N, C, oH, oW, y_max_interp_size, x_max_interp_size)
    # # y_weights.shape: (oH, y_max_interp_size) -> (oH, 1, y_max_interp_size, 1)
    # # x_weights.shape: (oW, x_max_interp_size) -> (1, oW, 1, x_max_interp_size)
    # input_selected = input_selected.transpose(-3, -2)
    # y_weights = y_weights.view(y_weights.shape[0], 1, y_weights.shape[1], 1).expand(
    #     y_weights.shape[0], output_size[1], y_weights.shape[1], x_max_interp_size
    # )
    # x_weights = x_weights.view(1, x_weights.shape[0], 1, x_weights.shape[1]).expand(
    #     output_size[0], x_weights.shape[0], y_max_interp_size, x_weights.shape[1]
    # )
    # output = (y_weights * (x_weights * input_selected)).sum(dim=(-1, -2))


    # SAME AS SEPARABLE VERSION
    x_weights = x_weights.view(1, output_size[1], x_max_interp_size)
    y_weights = y_weights.view(output_size[0], y_max_interp_size, 1)
    output = (x_weights * input_selected).sum(dim=-1)
    output = (y_weights * output).sum(dim=-2)

    output = output.contiguous(memory_format=memory_format)

    if not input.is_floating_point():
        output = output.round()

    return output
```

### Debug Vertical Pass

```
TORCH_COMPILE_DEBUG=1 python -m debugpy --wait-for-client --listen 5678 check_interpolate_bilinear_aa.py
```

- TritonScheduling can fuse buf3 and buf4, but CppScheduling can't. In `_can_fuse_horizontal_impl`
```python
#     vars1, vars2: ((4, 3, 271, 456), (4, 3, 123576))
#     reduce1, reduce2: ((), ())

    def _can_fuse_horizontal_impl(self, node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return False
```
- Loop order in C++ is better to be `(4, 271, 456, 3)`


- `group_fn = CppScheduling.group_fn`


```
                    in_suffix = True
                    if node.group[1] == (group, ()):
                        # we can fuse in some extra pointwise into the suffix
                        with kernel.write_to_suffix():
                            node.run(vars, ())
                    else:
                        from torch._inductor.ir import LoopBody
                        node.group = (node.group[0], (group, ()))
                        node._sizes = ([4, 3, 271, 456], [])
                        node._body = LoopBody(node._body, ((4, 3, 271 * 456), ), {"z0": 4, "z1": 3, "z2": 271, "z3": 456})

                        with kernel.write_to_suffix():
                            node.run(vars, ())

                        # assert False, f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
```



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

- output -> finalize -> get_fill_order
```
get_fill_order (/home/vfdev-5/pytorch/torch/_inductor/ir.py:2688)
decide_layout (/home/vfdev-5/pytorch/torch/_inductor/ir.py:2721)
finalize (/home/vfdev-5/pytorch/torch/_inductor/graph.py:724)
output (/home/vfdev-5/pytorch/torch/_inductor/graph.py:715)
run_node (/home/vfdev-5/pytorch/torch/fx/interpreter.py:195)
run_node (/home/vfdev-5/pytorch/torch/_inductor/graph.py:757)
run (/home/vfdev-5/pytorch/torch/fx/interpreter.py:138)
run (/home/vfdev-5/pytorch/torch/_inductor/graph.py:464)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:221)
fx_codegen_and_compile (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:547)
compile_fx_inner (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:350)
inner (/home/vfdev-5/usr/lib/python3.8/contextlib.py:75)
inner (/home/vfdev-5/pytorch/torch/_inductor/debug.py:297)
debug_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/repro/after_aot.py:80)
fw_compiler_base (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:1108)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:221)
aot_dispatch_base (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:1604)
aot_wrapper_synthetic_base (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:2423)
aot_wrapper_dedupe (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:2243)
create_aot_dispatcher_function (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:3460)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:221)
aot_module_simplified (/home/vfdev-5/pytorch/torch/_functorch/aot_autograd.py:3922)
compiler_fn (/home/vfdev-5/pytorch/torch/_dynamo/backends/common.py:55)
compile_fx (/home/vfdev-5/pytorch/torch/_inductor/compile_fx.py:1171)
__call__ (/home/vfdev-5/pytorch/torch/__init__.py:1604)
debug_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/repro/after_dynamo.py:117)
call_user_compiler (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:1039)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:221)
compile_and_call_fx_graph (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:987)
inner (/home/vfdev-5/usr/lib/python3.8/contextlib.py:75)
compile_subgraph (/home/vfdev-5/pytorch/torch/_dynamo/output_graph.py:859)
RETURN_VALUE (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:2217)
step (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:710)
run (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:747)
run (/home/vfdev-5/pytorch/torch/_dynamo/symbolic_convert.py:2107)
transform (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:462)
transform_code_object (/home/vfdev-5/pytorch/torch/_dynamo/bytecode_transformation.py:1028)
compile_inner (/home/vfdev-5/pytorch/torch/_dynamo/convert_frame.py:492)
time_wrapper (/home/vfdev-5/pytorch/torch/_dynamo/utils.py:221)
```
for buf3
```

```



- buf3 is chosen as a CF tensor even if it is hinted in the decomp to be CL
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


- 31/10/2023

```
[-------------------------------------------- Interpolate bilinear, AA=true, cpu -------------------------------------------]
                                                                                      |   Eager   |  Compiled  |  Just C++
1 threads: ------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format      |    766.0  |   20620.2  |     20677.3
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format    |   1719.6  |   19530.7  |     19497.4
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last          |    333.1  |   20856.8  |     21364.1
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last        |   2199.4  |   19880.4  |     19793.3
      Input (4, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format      |   3217.5  |   83345.9  |     83517.6
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format    |   7111.5  |   75855.2  |     75608.2
      Input (4, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last          |   1300.4  |   83555.3  |     83619.2
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last        |   8881.5  |   86082.9  |     85837.3
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format    |   3577.0  |   15391.7  |     15341.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.float32, torch.contiguous_format  |   9438.9  |   12377.3  |     12249.9
      Input (1, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last        |   1130.9  |   12413.4  |     12326.8
      Input (1, 3, 1200, 1300) -> (200, 300), torch.float32, torch.channels_last      |  10247.6  |    8201.1  |      8060.1
      Input (4, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.contiguous_format    |  13913.0  |   62827.1  |     62963.2
      Input (4, 3, 1200, 1300) -> (200, 300), torch.float32, torch.contiguous_format  |  37462.7  |   48888.1  |     48815.6
      Input (4, 3, 1200, 1300) -> (200, 300), torch.uint8, torch.channels_last        |   4723.9  |   51666.5  |     51516.6
      Input (4, 3, 1200, 1300) -> (200, 300), torch.float32, torch.channels_last      |  41864.6  |   34116.5  |     33966.8
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format      |   1864.5  |   35455.5  |     35566.7
      Input (1, 3, 300, 400) -> (600, 700), torch.float32, torch.contiguous_format    |   2653.1  |   35303.6  |     35455.9
      Input (1, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last          |    510.8  |   35767.9  |     35901.2
      Input (1, 3, 300, 400) -> (600, 700), torch.float32, torch.channels_last        |   5365.5  |   33153.8  |     33548.8
      Input (4, 3, 300, 400) -> (600, 700), torch.uint8, torch.contiguous_format      |   8824.6  |  141773.7  |    139218.2
      Input (4, 3, 300, 400) -> (600, 700), torch.float32, torch.contiguous_format    |  10564.3  |  140867.2  |    140051.1
      Input (4, 3, 300, 400) -> (600, 700), torch.uint8, torch.channels_last          |   2093.3  |  153663.1  |    154263.3
      Input (4, 3, 300, 400) -> (600, 700), torch.float32, torch.channels_last        |  21512.9  |  132679.4  |    132839.2

Times are in microseconds (us).


[------------------------------------------- Interpolate bilinear, AA=true, cuda --------------------------------------------]
                                                                                        |  Eager   |  Compiled  |  Just Triton
1 threads: -------------------------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format      |    11.3  |     65.3   |       22.3
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last          |    29.3  |     70.5   |       23.8
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format      |    42.1  |     70.1   |       33.8
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last          |    91.8  |     64.9   |       47.9
      Input (1, 3, 1200, 1300) -> (200, 300), torch.float32, torch.contiguous_format    |    85.9  |    117.2   |      117.9
      Input (1, 3, 1200, 1300) -> (200, 300), torch.float32, torch.channels_last        |   216.9  |    163.9   |      166.0
      Input (4, 3, 1200, 1300) -> (200, 300), torch.float32, torch.contiguous_format    |   331.8  |    471.0   |      472.0
      Input (4, 3, 1200, 1300) -> (200, 300), torch.float32, torch.channels_last        |   841.7  |    647.4   |      650.8
      Input (1, 3, 300, 400) -> (600, 700), torch.float32, torch.contiguous_format      |    30.1  |     64.7   |       28.4
      Input (1, 3, 300, 400) -> (600, 700), torch.float32, torch.channels_last          |    54.2  |     65.7   |       39.3
      Input (4, 3, 300, 400) -> (600, 700), torch.float32, torch.contiguous_format      |    95.6  |    109.8   |      109.5
      Input (4, 3, 300, 400) -> (600, 700), torch.float32, torch.channels_last          |   187.8  |    232.8   |      232.8
      Input (1, 3, 2345, 2456) -> (1234, 1345), torch.float32, torch.contiguous_format  |   246.0  |    225.0   |      229.2
      Input (1, 3, 2345, 2456) -> (1234, 1345), torch.float32, torch.channels_last      |   778.7  |    575.0   |      574.9
      Input (4, 3, 2345, 2456) -> (1234, 1345), torch.float32, torch.contiguous_format  |   872.5  |    898.3   |      898.7
      Input (4, 3, 2345, 2456) -> (1234, 1345), torch.float32, torch.channels_last      |  2996.3  |   2289.4   |     2288.8
      Input (1, 3, 1234, 1345) -> (2345, 2456), torch.float32, torch.contiguous_format  |   431.3  |    381.1   |      381.2
      Input (1, 3, 1234, 1345) -> (2345, 2456), torch.float32, torch.channels_last      |   796.4  |   1042.8   |     1044.2
      Input (4, 3, 1234, 1345) -> (2345, 2456), torch.float32, torch.contiguous_format  |  1323.6  |   1518.6   |     1515.9
      Input (4, 3, 1234, 1345) -> (2345, 2456), torch.float32, torch.channels_last      |  2797.8  |   3728.6   |     3728.0
      Input (1, 3, 2345, 2456) -> (120, 200), torch.float32, torch.contiguous_format    |   461.7  |    435.9   |      436.3
      Input (1, 3, 2345, 2456) -> (120, 200), torch.float32, torch.channels_last        |   941.6  |    521.6   |      521.3
      Input (4, 3, 2345, 2456) -> (120, 200), torch.float32, torch.contiguous_format    |  1832.3  |   1724.6   |     1725.6
      Input (4, 3, 2345, 2456) -> (120, 200), torch.float32, torch.channels_last        |  3740.3  |   2040.9   |     2041.0

Times are in microseconds (us).
```



```bash
root@qgpu1:/tmp/pth/inductor# python -u profile_interp_bilinear_aa.py
torch.Size([4, 3, 2345, 3456]) torch.float32 True
STAGE:2023-10-30 16:06:28 16591:16591 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2023-10-30 16:07:29 16591:16591 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2023-10-30 16:07:29 16591:16591 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
        loop_interp_bilinear_aa_cuda         0.29%     177.805ms        89.80%       55.193s       55.193s       0.000us         0.00%       66.164s       66.164s             1
               Torch-Compiled Region         0.75%     458.558ms        89.43%       54.961s       5.496ms       0.000us         0.00%       66.164s       6.616ms         10000
    triton_per_fused_index_mul_sum_0         0.19%     116.583ms        88.56%       54.429s       5.443ms       61.440s        99.99%       66.146s       6.615ms         10000
                    triton__0d1d2de3         0.00%       0.000us         0.00%       0.000us       0.000us       61.446s       100.00%       61.446s       6.145ms         10000
                      cuLaunchKernel        88.37%       54.313s        88.37%       54.313s       5.431ms        4.724s         7.69%        4.724s     472.427us         10000
            TorchDynamo Cache Lookup         0.09%      54.039ms         0.09%      54.039ms       5.404us       0.000us         0.00%       0.000us       0.000us         10000
                         aten::empty         0.12%      73.508ms         0.12%      73.508ms       7.351us       0.000us         0.00%       0.000us       0.000us         10000
               cudaDeviceSynchronize        10.20%        6.266s        10.20%        6.266s        6.266s       0.000us         0.00%       0.000us       0.000us             1
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 61.459s
Self CUDA time total: 61.446s

STAGE:2023-10-30 16:07:33 16591:16591 ActivityProfilerController.cpp:312] Completed Stage: Warm Up
STAGE:2023-10-30 16:08:35 16591:16591 ActivityProfilerController.cpp:318] Completed Stage: Collection
STAGE:2023-10-30 16:08:35 16591:16591 ActivityProfilerController.cpp:322] Completed Stage: Post Processing
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         loop_triton         0.38%     235.331ms        89.69%       55.138s       55.138s       0.000us         0.00%       61.397s       61.397s             1
                    triton__0d1d2de3         0.00%       0.000us         0.00%       0.000us       0.000us       61.403s       100.00%       61.403s       6.140ms         10000
    triton_per_fused_index_mul_sum_0         0.17%     107.492ms        89.20%       54.837s       5.484ms       61.397s        99.99%       61.397s       6.140ms         10000
                         aten::empty         0.11%      66.013ms         0.11%      66.013ms       6.601us       0.000us         0.00%       0.000us       0.000us         10000
                      cuLaunchKernel        89.03%       54.729s        89.03%       54.729s       5.473ms       0.000us         0.00%       0.000us       0.000us         10000
               cudaDeviceSynchronize        10.31%        6.338s        10.31%        6.338s        6.338s       0.000us         0.00%       0.000us       0.000us             1
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 61.477s
Self CUDA time total: 61.403s
```


- 25/10/2023
```
root@qgpu1:/tmp/pth/inductor# python -u /tmp/pth/inductor/torch_compile_debug/upsample_aa_LS_bs1_single_kernel_run_2023_10_25_09_41_08_923946-pid_116566/torchinductor/model___9.0/output_code_bench.py
Compile triton_per_fused_index_mul_sum_0
Compile triton_per_fused_index_mul_sum_0_v2
Compile triton_per_fused_index_mul_sum_0
Compile triton_per_fused_index_mul_sum_0_v2
- Check consistency v0
- Check consistency v2
- Start benchmarks
[----------------------------------------------------------- Interpolate bilinear, AA=true, cuda -----------------------------------------------------------]
                                                                 |  Eager F.interpolate  |  Compiled F.interpolate  |  Compiled Triton  |  Compiled Triton v2
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------
      Input (4, 3, 3456, 4567) -> 2345, 3456, torch.float32, CF  |          3.2          |           6.0            |        4.3        |         3.5

Times are in milliseconds (ms).

Triton Testing do_bench: rep=2000, return_mode=median
Interpolate bilinear, AA=true, CUDA
Input (4, 3, 3456, 4567) -> 2345, 3456, torch.float32, CF
- Eager F.interpolate 3.2055039405822754
- Compiled F.interpolate 5.966815948486328
- Compiled Triton 4.24729585647583
- Compiled Triton v2 3.433568000793457
```

Why "Compiled F.interpolate" is 6.0 vs "Compiled Triton" is 4.3 ms?

Profiling shows:
```
186002 function calls (183002 primitive calls) in 0.116 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
2000/1000    0.014    0.000    0.115    0.000 /pytorch/torch/_dynamo/eval_frame.py:378(_fn)
     1000    0.009    0.000    0.009    0.000 {built-in method torch.empty}
     1000    0.007    0.000    0.007    0.000 {built-in method __triton_launcher.launch}
     1000    0.004    0.000    0.042    0.000 /tmp/torchinductor_root/rh/crhpk4gxlssj4zvvpvul7lvvldvun5bs6bkrtikcu5fjhavxivxb.py:258(call)
     1000    0.004    0.000    0.023    0.000 /pytorch/torch/_inductor/triton_heuristics.py:490(run)
     1000    0.004    0.000    0.061    0.000 /pytorch/torch/_functorch/aot_autograd.py:2883(runtime_wrapper)
9000/7000    0.003    0.000    0.004    0.000 /usr/local/lib/python3.8/dist-packages/triton/compiler/compiler.py:702(__getattribute__)
     4000    0.003    0.000    0.004    0.000 /usr/lib/python3.8/contextlib.py:82(__init__)
     1000    0.003    0.000    0.008    0.000 <string>:2(guard)
     1000    0.003    0.000    0.018    0.000 <string>:1(launcher)
     4000    0.003    0.000    0.010    0.000 /pytorch/torch/_dynamo/eval_frame.py:113(backend_cache_wrapper)
     4000    0.003    0.000    0.003    0.000 /pytorch/torch/_dynamo/eval_frame.py:131(_set_current_backend)
     8000    0.002    0.000    0.013    0.000 {built-in method builtins.next}
     1000    0.002    0.000    0.050    0.000 /pytorch/torch/_functorch/aot_autograd.py:1802(call_func_with_args)
```





- 24/10/2023 - single kernel and added (256, 512, 1024) to reduction_hint configs
```
[----------------------------------- Interpolate bilinear, AA=true, cuda ------------------------------------]
                                                                                        |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------
      Input (1, 3, 3456, 4567) -> (2345, 3456), torch.float32, torch.contiguous_format  |   922.5  |   1034.7
      Input (4, 3, 3456, 4567) -> (2345, 3456), torch.float32, torch.contiguous_format  |  3201.2  |   5961.9

Without adding to reduction_hint configs
[----------------------------------- Interpolate bilinear, AA=true, cuda ------------------------------------]
                                                                                        |  Eager   |  Compiled
1 threads: ---------------------------------------------------------------------------------------------------
      Input (1, 3, 3456, 4567) -> (2345, 3456), torch.float32, torch.contiguous_format  |   927.1  |   2896.0
      Input (4, 3, 3456, 4567) -> (2345, 3456), torch.float32, torch.contiguous_format  |  3185.2  |  14785.8


[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   11.2  |    63.5
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   42.8  |    72.0

Times are in microseconds (us).

[--------------------------------------- Interpolate nearest, cuda ---------------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |    9.1  |    67.0
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   15.4  |    70.5

Times are in microseconds (us).
```


- 20/10/2023 - single kernel

```
- persistent_reduction, xblock in (1, 8, 32, 128, 512)
[---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   28.4  |   102.2
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   12.7  |   100.8
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   82.1  |   284.6
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   49.8  |   268.3

      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   28.6  |    70.1
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   11.4  |    64.1
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   91.9  |    70.8
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   43.4  |    70.8

- persistent_reduction, xblock in (1, 8, 32, 128)

[---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   28.1  |   103.0
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   12.8  |   110.2
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   82.4  |   288.4
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   49.6  |   270.6

      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   28.3  |    61.6
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   11.4  |    62.3
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   92.0  |   102.4
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   43.4  |   100.5

- persistent_reduction, xblock in (1, 8, 32, 128, 256, 512)

[---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   28.3  |   100.5
      Input (1, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   12.7  |    99.7
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.channels_last      |   83.0  |   289.0
      Input (4, 3, (345, 456)) -> (123, 124), torch.float32, torch.contiguous_format  |   50.1  |   271.5

      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   29.9  |    69.3
      Input (1, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   11.4  |    68.0
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.channels_last      |   91.9  |    73.9
      Input (4, 3, (500, 400)) -> (256, 256), torch.float32, torch.contiguous_format  |   43.3  |    69.8


```


- 17/10/2023 - v2
```
[----------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |   Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format    |    830.6  |   5549.4
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   1732.6  |   5068.2
      Input (1, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last        |    339.4  |   5541.5
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   2196.4  |   5900.1
      Input (4, 3, 500, 400) -> (256, 256), torch.uint8, torch.contiguous_format    |   3262.9  |  21497.6
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   7372.4  |  20353.8
      Input (4, 3, 500, 400) -> (256, 256), torch.uint8, torch.channels_last        |   1362.3  |  22442.0
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   8943.1  |  21495.9
      Input (1, 3, 345, 456) -> (123, 124), torch.uint8, torch.contiguous_format    |    406.0  |   1078.6
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   1049.4  |    958.4
      Input (1, 3, 345, 456) -> (123, 124), torch.uint8, torch.channels_last        |    151.7  |   1574.9
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   1145.9  |   1423.8
      Input (4, 3, 345, 456) -> (123, 124), torch.uint8, torch.contiguous_format    |   1534.1  |   4224.0
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   4177.4  |   3801.6
      Input (4, 3, 345, 456) -> (123, 124), torch.uint8, torch.channels_last        |    514.6  |   6131.6
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   4621.8  |   5427.6
      Input (1, 3, 345, 456) -> (567, 678), torch.uint8, torch.contiguous_format    |   2099.2  |   5614.6
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   2708.4  |   4988.2
      Input (1, 3, 345, 456) -> (567, 678), torch.uint8, torch.channels_last        |    536.6  |   6111.1
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |   5181.2  |   5923.8
      Input (4, 3, 345, 456) -> (567, 678), torch.uint8, torch.contiguous_format    |   8754.6  |  22512.5
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |  11052.0  |  19678.4
      Input (4, 3, 345, 456) -> (567, 678), torch.uint8, torch.channels_last        |   2025.3  |  24570.5
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |  20539.0  |  23203.3

Times are in microseconds (us).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   11.6  |    81.5
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   30.4  |    82.2
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   42.1  |    87.9
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   90.8  |    89.3
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   12.4  |   115.6
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   32.7  |   113.6
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   48.3  |   113.9
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   81.7  |   123.9
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   29.5  |    92.3
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |   53.1  |   100.3
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   94.3  |   210.8
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |  187.3  |   145.8


[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   11.1  |    84.1
      Input (1, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   27.9  |    84.0
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.contiguous_format  |   42.7  |    92.1
      Input (4, 3, 500, 400) -> (256, 256), torch.float32, torch.channels_last      |   91.9  |    92.4
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   12.8  |   115.4
      Input (1, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   27.5  |   115.6
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.contiguous_format  |   49.4  |   116.4
      Input (4, 3, 345, 456) -> (123, 124), torch.float32, torch.channels_last      |   83.1  |   128.4
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   29.7  |    94.9
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |   53.1  |    94.9
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   94.8  |   216.0
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |  189.2  |   150.3

Times are in microseconds (us).
```


- 16/10/2023 - v4, single cuda kernel
```
[----------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   1.6   |    26.9
      Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   1.1   |    28.9
      Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   6.5   |   104.1
      Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   4.4   |   107.4

Times are in milliseconds (ms).

[---------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                      |  Eager  |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   30.0  |    62.3
      Input (1, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   11.9  |    62.4
      Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.channels_last      |   84.4  |   227.1
      Input (4, 3, (345, 456)) -> (271, 272), torch.float32, torch.contiguous_format  |   43.2  |   211.3

Times are in microseconds (us).
```


- 13/10/2023 - v3 no split as separable code

```
[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   84.2  |    82.1
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   43.1  |    81.3

      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   29.4  |    82.2
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   12.0  |    81.9

Times are in microseconds (us).
```

- 12/10/2023 - v2

```
[-------------------------------- Interpolate bilinear, AA=true, cuda ----------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (4, 3, 345, 456) -> (34, 35), torch.float32, torch.channels_last        |  339.9  |   124.4
      Input (4, 3, 345, 456) -> (34, 35), torch.float32, torch.contiguous_format    |  308.1  |   116.4
      Input (1, 3, 345, 456) -> (34, 35), torch.float32, torch.channels_last        |   75.6  |   116.8
      Input (1, 3, 345, 456) -> (34, 35), torch.float32, torch.contiguous_format    |   69.0  |   125.7

      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   84.3  |    82.3
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   43.1  |    87.6
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   29.9  |    88.7
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   12.0  |    80.9

Times are in microseconds (us).
```

```
Torch version: 2.2.0a0+git38bb283
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_89,code=sm_89
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.2.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[2023-10-12 14:41:29,485] [0/10] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[2023-10-12 14:41:29,573] [0/10] torch._inductor.scheduler: [ERROR] Generating code for node buf7 with estimated runtime 0.0
[2023-10-12 14:41:53,799] [0/11] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[2023-10-12 14:43:51,322] [0/16] torch._inductor.scheduler: [ERROR] Generating code for node buf4 with estimated runtime 0.0
[2023-10-12 14:43:51,452] [0/16] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[2023-10-12 14:43:51,518] [0/16] torch._inductor.scheduler: [ERROR] Generating code for node buf7 with estimated runtime 0.0
[2023-10-12 14:44:15,213] [0/17] torch._inductor.scheduler: [ERROR] Generating code for node buf4 with estimated runtime 0.0
[2023-10-12 14:44:15,322] [0/17] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[2023-10-12 14:46:11,799] [0/22] torch._inductor.scheduler: [ERROR] Generating code for node buf4 with estimated runtime 0.0
[2023-10-12 14:46:11,911] [0/22] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[2023-10-12 14:46:12,025] [0/22] torch._inductor.scheduler: [ERROR] Generating code for node buf7 with estimated runtime 0.0
[2023-10-12 14:46:35,438] [0/23] torch._inductor.scheduler: [ERROR] Generating code for node buf4 with estimated runtime 0.0
[2023-10-12 14:46:35,543] [0/23] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0
[----------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |   Eager   |  Compiled
1 threads: ------------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (271, 272), torch.uint8, torch.contiguous_format    |    623.0  |   1882.2
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   1014.1  |   1466.0
      Input (1, 3, 345, 456) -> (271, 272), torch.uint8, torch.channels_last        |    251.3  |   2232.1
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   1537.4  |   1764.9
      Input (4, 3, 345, 456) -> (271, 272), torch.uint8, torch.contiguous_format    |   2333.5  |   7733.9
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   4366.2  |   5806.6
      Input (4, 3, 345, 456) -> (271, 272), torch.uint8, torch.channels_last        |    925.6  |   8910.6
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   6194.6  |   7525.3

      Input (1, 3, 345, 456) -> (567, 678), torch.uint8, torch.contiguous_format    |   2170.2  |   3020.9
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   2483.4  |   2512.8
      Input (1, 3, 345, 456) -> (567, 678), torch.uint8, torch.channels_last        |    534.7  |   4060.9
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |   5169.1  |   3782.0
      Input (4, 3, 345, 456) -> (567, 678), torch.uint8, torch.contiguous_format    |   7954.9  |  12137.4
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   9919.4  |   9874.2
      Input (4, 3, 345, 456) -> (567, 678), torch.uint8, torch.channels_last        |   2021.2  |  16475.9
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |  20689.3  |  14815.4

Times are in microseconds (us).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   11.7  |    92.8
      Input (1, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   31.8  |   101.2
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.contiguous_format  |   42.2  |   100.5
      Input (4, 3, 345, 456) -> (271, 272), torch.float32, torch.channels_last      |   84.8  |   109.9

      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   29.3  |   109.3
      Input (1, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |   52.8  |   109.6
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.contiguous_format  |   94.8  |   144.2
      Input (4, 3, 345, 456) -> (567, 678), torch.float32, torch.channels_last      |  187.4  |   112.3

Times are in microseconds (us).
```


- 09/10/2023 - v2
```

Torch version: 2.2.0a0+git38bb283
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_89,code=sm_89
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers
-Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.2.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


[2023-10-09 16:51:59,320] [0/10] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0, Error: cannot determine truth value of Relational
[2023-10-09 16:51:59,436] [0/10] torch._inductor.scheduler: [ERROR] Generating code for node buf7 with estimated runtime 0.0, Error: cannot determine truth value of Relational
[2023-10-09 16:52:23,743] [0/11] torch._inductor.scheduler: [ERROR] Generating code for node buf5 with estimated runtime 0.0, Error: cannot determine truth value of Relational
[---------------------------------- Interpolate bilinear, AA=true, cpu ----------------------------------]
                                                                                    |  Eager   |  Compiled
1 threads: -----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |   655.0  |   1849.6
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  1097.5  |   1466.2

      Input (1, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   255.4  |   2208.8
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  1607.5  |   1741.7

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.contiguous_format    |  2785.3  |   7473.9
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |  4393.0  |   5825.3

      Input (4, 3, 345, 456) -> (270, 270), torch.uint8, torch.channels_last        |   919.6  |   8868.9
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |  6479.0  |   7480.9

Times are in microseconds (us).

[--------------------------------- Interpolate bilinear, AA=true, cuda ---------------------------------]
                                                                                    |  Eager  |  Compiled
1 threads: ----------------------------------------------------------------------------------------------
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   11.9  |   100.3
      Input (1, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   29.5  |    93.5
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.contiguous_format  |   42.7  |    99.8
      Input (4, 3, 345, 456) -> (270, 270), torch.float32, torch.channels_last      |   85.0  |   100.5

Times are in microseconds (us).
```

- 09/10/2023 - v0
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


## Create fx graph, compile and run it and check

```
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.ir import InterpreterShim
from torch.fx.experimental.proxy_tensor import make_fx

            m = torch.nn.Conv2d(5, 6, [3, 3])

            def fn(inp, weight):
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            inp = torch.randn([2, 5, 16, 16])
            inps = [inp, m.weight.to(memory_format=fmt)]
            fn_fx = make_fx(fn)(*inps)
            fn_compiled = compile_fx_inner(fn_fx, inps)
            test_self = self
            conv_seen = False

            class RecordFunctions(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs if kwargs else {}
                    if func == torch.ops.aten.convolution.default:
                        # For CPU and mkldnn enable, we always using channles last
                        nonlocal fmt
                        if (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            fmt = torch.channels_last
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                out = fn_compiled(inps)

            self.assertTrue(conv_seen)
```