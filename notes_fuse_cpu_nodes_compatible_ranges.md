

## Removed code

```
# TODO: do we really need to compare free symbols in addition to already matching var ranges?
# extra_indexing_symbols = set.union(
#     *[f.free_symbols for f in extra_indexing_expr]
# )
# indexing_symbols = set.union(*[f.free_symbols for f in index_formulas])
# # remove symbols not present in var_ranges keys
# extra_indexing_symbols = {s for s in extra_indexing_symbols if s in expected_var_ranges}
# indexing_symbols = {s for s in indexing_symbols if s in expected_var_ranges}

# assert len(indexing_symbols - extra_indexing_symbols) == 0, (
#     indexing_symbols,
#     index_formulas,
#     extra_indexing_symbols,
#     extra_indexing_expr,
#     extra_indexing_constraints,
# )
```



## Tests

```
pytest -vvv test/inductor/test_perf.py -k test_fusion_choice4_cpu

python test/inductor/test_fused_attention.py -k test_pattern_fails_with_reuse_cpu && \
python test/inductor/test_aot_inductor.py -k test_seq_non_abi_compatible_cpu && \
python test/inductor/test_aot_inductor.py -k test_seq_abi_compatible_cpu && \
python test/inductor/test_fused_attention.py -k test_pattern_fails_with_reuse_cpu && \
python test/inductor/test_torchinductor.py -k test_softmax_one_kernel_loop_cpu && \
python test/inductor/test_torchinductor.py -k test_dtype_mismatch_issue_cpu && \
python test/inductor/test_cpu_repro.py -k test_ir_node_str && \
python test/inductor/test_cpu_repro.py -k test_masked_fill_softmax && \
python test/inductor/test_torchinductor.py -k test_cat_cpu && \
python test/inductor/test_torchinductor_dynamic_shapes.py -k test_add_complex3_dynamic_shapes_cuda && \
python test/inductor/test_torchinductor_dynamic_shapes.py -k test_avg_pool2d6_dynamic_shapes_cuda && \
python test/inductor/test_torchinductor_codegen_dynamic_shapes.py -k test_gather3_dynamic_shapes_cpu && \
python test/inductor/test_torchinductor_codegen_dynamic_shapes.py -k test_slice_scatter5_dynamic_shapes_cpu && \
pytest -vvvv test/inductor/test_compiled_autograd.py::TestAutogradWithCompiledAutograd::test_setitem
```


## Perfs on upsampling bicubic

```
python -u perf_interp_mode_custom.py --mode=bicubic
```

```
[------------------------------------------------------------------------------------------------ Interpolate, cpu ------------------------------------------------------------------------------------------------]
                                                                                                                                                   |  Eager (2.3.0a0)   |  Compiled (2.3.0a0)
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Downsampling to small output image 256x256
- No fusion (8ef4a43a31a)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)     |        1.745 (+-0.073)        |         3.598 (+-0.111)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |        1.749 (+-0.065)        |         3.852 (+-0.140)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)         |        2.804 (+-0.131)        |         3.784 (+-0.141)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |        2.817 (+-0.138)        |         4.134 (+-0.132)

- With fusion (PR 120077, merged as ffd0b4de1d3) -> fusion benefits are invisible
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)     |        1.756 (+-0.060)        |         3.574 (+-0.108)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |        1.753 (+-0.056)        |         3.686 (+-0.123)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)         |        2.834 (+-0.135)        |         3.782 (+-0.088)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |        2.862 (+-0.126)        |         4.130 (+-0.105)

- With extended fusion -> fusion benefits are invisible
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)     |        1.768 (+-0.061)        |         3.601 (+-0.108)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)    |        1.761 (+-0.075)        |         3.690 (+-0.112)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (256, 256)         |        2.876 (+-0.126)        |         3.719 (+-0.123)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (256, 256)        |        2.859 (+-0.123)        |         4.061 (+-0.131)

Upsampling to large output image 1024x1024
- No fusion (8ef4a43a31a)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)   |        27.097 (+-1.031)       |         56.546 (+-1.339)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)  |        26.972 (+-1.128)       |         66.710 (+-3.071)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)       |        43.651 (+-1.563)       |         61.883 (+-2.002)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)      |        43.777 (+-1.466)       |         80.301 (+-3.199)

- With fusion (PR 120077, merged as ffd0b4de1d3) -> visible only for ac=false and CF
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)   |        26.906 (+-1.212)       |         56.570 (+-1.846)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)  |        26.940 (+-1.122)       |         58.319 (+-1.924)    <---
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)       |        44.541 (+-1.838)       |         61.911 (+-2.280)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)      |        44.513 (+-1.792)       |         80.829 (+-2.849)

- With extended fusion -> visible only for ac=false and CF / CL
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)   |        27.171 (+-1.108)       |         56.638 (+-1.361)
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)  |        27.242 (+-1.067)       |         58.309 (+-1.500)    <---
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: True, antialias: False, osize: (1024, 1024)       |        44.792 (+-1.700)       |         58.871 (+-1.848)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: bicubic, align_corners: False, antialias: False, osize: (1024, 1024)      |        44.419 (+-1.432)       |         64.399 (+-1.467)    <---


Times are in milliseconds (ms).
```

Observations:
- For CL, ac=false and ac=true there is no last copy fusion into the main loop for "with fusion" option:
    - /tmp/pth/inductor/torch_compile_debug/run_2024_02_26_15_15_20_187785-pid_63107-ac=false-cl-w-cpp-nodes-fusion/torchinductor/model___9.0/output_code.py
    - /tmp/pth/inductor/torch_compile_debug/run_2024_02_26_15_12_45_864420-pid_62872-ac=false-cl-no-fusion/torchinductor/model___9.0/output_code.py
    - .
    - /tmp/pth/inductor/torch_compile_debug/run_2024_02_26_15_13_27_310222-pid_62950-ac=true-cl-no-fusion/torchinductor/model___9.0/output_code.py
    - /tmp/pth/inductor/torch_compile_debug/run_2024_02_26_15_16_21_626098-pid_63185-ac=true-cl-w-cpp-nodes-fusion/torchinductor/model___9.0/output_code.py
- For CF, ac=true, there is no last copy for "no fusion" and "with fusion" options -> there is no speed-up






### Questions:

We do the following:
```python
def get_indexing_ranges_exprs(node):
      if isinstance(node, FusedSchedulerNode):
      assert len(node.snodes) > 0, node.snodes
      var_ranges = None
      indexing_exprs = set()
      for snode in node.snodes:
            v, exprs = get_indexing_ranges_exprs(snode)
            if var_ranges is None:
                  var_ranges = v
            # TODO: this can be also a strong assumption that fused nodes have
            # the same variable and ranges
            assert var_ranges == v, (var_ranges, v, node.snodes)
            for expr in exprs:
                  indexing_exprs.add(expr)
      return var_ranges, list(indexing_exprs)
      else:
      assert isinstance(node, SchedulerNode)
      comp_buffer = node.node
      assert isinstance(comp_buffer, ir.ComputedBuffer)
      _, body, _ = comp_buffer.get_default_sizes_body()
      return body.var_ranges, list(body.indexing_exprs.values())
```

Question, does `assert var_ranges == v, (var_ranges, v, node.snodes)` hold for fused nodes?

1) How we get `var_ranges` ? -> For each snode from snodes from FusedSchedulerNode we get `body` using `get_default_sizes_body`.
2) `body.var_ranges` are defined as `LoopBody.var_ranges` and computed with
```python
args, var_ranges = dependencies.index_vars_squeeze(
    self.data.get_pointwise_size(), self.data.get_reduction_size(), prefix="q"
)
```
and basically is matching `ComputedBuffer.data.get_pointwise_size()`
