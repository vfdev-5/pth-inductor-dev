

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
python test/inductor/test_torchinductor_codegen_dynamic_shapes.py -k test_slice_scatter5_dynamic_shapes_cpu
```