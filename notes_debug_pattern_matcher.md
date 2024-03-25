


Output graphs:
```
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param4 = root._frozen_param4
    _frozen_param5 = root._frozen_param5
    _frozen_param8 = root._frozen_param8
    _frozen_param9 = root._frozen_param9
    mul = torch.ops.aten.mul.Tensor(arg8_1, 45.20535508292672);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 84);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None
    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None
    qconv = torch__inductor_fx_passes_quantization_qconv([], '', x = convert_element_type, x_scale = 0.02212127298116684, x_zp = 84, packed_weight = _frozen_param8, w_scale = _frozen_param4, w_zp = _frozen_param5, b = _frozen_param1, stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 1, inv_output_scale = 1.0, output_zero_point = 0, output_dtype = torch.float32, attr = 'none', o_inv_scale = 52.63383035733631, o_zp = 141, o_qmin = 0, o_qmax = 255, o_dtype = torch.uint8);  _frozen_param8 = _frozen_param4 = _frozen_param5 = _frozen_param1 = None
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(qconv, torch.float32);  qconv = None
    sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_4, 141);  convert_element_type_4 = None
    mul_4 = torch.ops.aten.mul.Tensor(sub_2, 0.018999187275767326);  sub_2 = None
    qconv_binary = torch__inductor_fx_passes_quantization_qconv_binary([], '', x = convert_element_type, x_scale = 0.02212127298116684, x_zp = 84, packed_weight = _frozen_param9, w_scale = _frozen_param2, w_zp = _frozen_param3, b = _frozen_param0, stride = [1, 1], padding = [0, 0], dilation = [1, 1], groups = 1, inv_output_scale = 1.0, output_zero_point = 0, output_dtype = torch.float32, attr = 'none', accum_after_dequant = mul_4);  convert_element_type = _frozen_param9 = _frozen_param2 = _frozen_param3 = _frozen_param0 = mul_4 = None
    mul_5 = torch.ops.aten.mul.Tensor(qconv_binary, 70.31565792152601);  qconv_binary = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(round_3, 0);  round_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.014221583493053913);  convert_element_type_6 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_6, (216, 36, 6, 1));  mul_6 = None
    return (inductor_force_stride_order_default,)
```


```
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):
    mul = torch.ops.aten.mul.Tensor(arg8_1, 45.20535508292672);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 84);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None
    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None
    convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32)
    sub = torch.ops.aten.sub.Tensor(convert_element_type_1, 84);  convert_element_type_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(sub, 0.02212127298116684);  sub = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
    sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_2, 84);  convert_element_type_2 = None
    mul_2 = torch.ops.aten.mul.Tensor(sub_1, 0.02212127298116684);  sub_1 = None
    dequantize_per_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default(arg6_1, arg4_1, arg5_1, 0, -128, 127, torch.int8);  arg6_1 = arg4_1 = arg5_1 = None
    convolution = torch.ops.aten.convolution.default(mul_1, dequantize_per_channel, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = dequantize_per_channel = arg1_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(convolution, 52.63383035733631);  convolution = None
    round_2 = torch.ops.aten.round.default(mul_3);  mul_3 = None
    add_1 = torch.ops.aten.add.Tensor(round_2, 141);  round_2 = None
    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
    sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_4, 141);  convert_element_type_4 = None
    mul_4 = torch.ops.aten.mul.Tensor(sub_2, 0.018999187275767326);  sub_2 = None
    dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(arg7_1, arg2_1, arg3_1, 0, -128, 127, torch.int8);  arg7_1 = arg2_1 = arg3_1 = None
    convolution_1 = torch.ops.aten.convolution.default(mul_2, dequantize_per_channel_1, arg0_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_2 = dequantize_per_channel_1 = arg0_1 = None
    add_2 = torch.ops.aten.add.Tensor(convolution_1, mul_4);  convolution_1 = mul_4 = None
    relu = torch.ops.aten.relu.default(add_2);  add_2 = None
    mul_5 = torch.ops.aten.mul.Tensor(relu, 70.31565792152601);  relu = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(round_3, 0);  round_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.014221583493053913);  convert_element_type_6 = None
    return (mul_6,)
```


```
accum
TensorBox(StorageBox(
  ComputedBuffer(name='buf5', layout=FlexibleLayout('cpu', torch.float32, size=[1, 6, 6, 6], stride=[216, 36, 6, 1]), data=Pointwise(
    'cpu',
    torch.float32,
    def inner_fn(index):
        _, i1, i2, i3 = index
        tmp0 = ops.load(buf4, i1 + 6 * i3 + 36 * i2)
        tmp1 = ops.to_dtype(tmp0, torch.float32, src_dtype=torch.uint8)
        tmp2 = ops.constant(141, torch.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = ops.constant(0.018999187275767326, torch.float32)
        tmp5 = tmp3 * tmp4
        return tmp5
    ,
    ranges=[1, 6, 6, 6],
    origin_node=mul_4,
    origins={mul_4, sub_2, convert_element_type_4}
  ))
))
computation_op
<OpOverload(op='onednn.qconv2d_pointwise', overload='binary')>
match.nodes
[qconv2d_pointwise_default, add_2, relu]
```



Output:
qconv2d_binary_matcher_nodes, match.nodes:  [qconv2d_pointwise_default, add_2, relu]

stats [('calls_captured', 511), ('unique_graphs', 62)]
inline_call []
frames [('total', 1), ('ok', 1)]
inductor [
    ('pattern_matcher_nodes', 29),
    ('qconv2d_weight_prepack_matcher_nodes', 12),
    ('pattern_matcher_count', 7),
    ('qconv2d_unary_matcher_nodes', 7),
    ('dequant_promotion_matcher_nodes', 3),
    ('qconv2d_binary_matcher_nodes', 3),
    ('qconv2d_weight_prepack_matcher_count', 2),
    ('dequant_promotion_matcher_count', 1),
    ('qconv2d_unary_matcher_count', 1),
    ('qconv2d_binary_matcher_count', 1)
]
aot_autograd [('total', 1), ('ok', 1)]


Expected:

qconv2d_binary_matcher_nodes, match.nodes:  [qconv2d_pointwise_default, convert_element_type_4, sub_2, mul_4, add_2, relu, mul_5, round_3, add_3, clamp_min_2, clamp_max_2, convert_element_type_5]

stats [('calls_captured', 511), ('unique_graphs', 62)]
inline_call []
frames [('total', 1), ('ok', 1)]
inductor [
    ('pattern_matcher_nodes', 38),
    ('qconv2d_weight_prepack_matcher_nodes', 12),
    ('qconv2d_binary_matcher_nodes', 12),
    ('pattern_matcher_count', 7),
    ('qconv2d_unary_matcher_nodes', 7),
    ('dequant_promotion_matcher_nodes', 3),
    ('qconv2d_weight_prepack_matcher_count', 2),
    ('dequant_promotion_matcher_count', 1),
    ('qconv2d_unary_matcher_count', 1),
    ('qconv2d_binary_matcher_count', 1)
]
aot_autograd [('total', 1), ('ok', 1)]



## Debuging test_qat_qconv2d_add_relu

Breakpoint on `m = entry.pattern.match(node)`, pattern_matcher.py:
```
"convert_element_type" in node.name and "GraphPatternEntry" not in type(entry).__name__
```


pattern in _match:
```
CallFunction(aten.clamp_max.default,
CallFunction(aten.clamp_min.default,
CallFunction(aten.add.Tensor,
CallFunction(aten.round.default,
CallFunction(aten.mul.Tensor,

CallFunction(aten.add.Tensor,
CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()),
CallFunction(aten.mul.Tensor,
CallFunction(aten.sub.Tensor,
CallFunction(prims.convert_element_type.default, KeywordArg('accum'), KeywordArg('accum_dq_dtype')),KeywordArg('accum_zp')), KeywordArg('accum_scale'))), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax'))
```

node.graph (`node.graph.python_code("root").src`):
```
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param4 = root._frozen_param4
    _frozen_param5 = root._frozen_param5
    _frozen_param8 = root._frozen_param8
    _frozen_param9 = root._frozen_param9

    // Replacable
    mul = torch.ops.aten.mul.Tensor(arg8_1, 45.20535508292672);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 84);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None
    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

    qconv2d_pointwise_default = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type, 0.02212127298116684, 84, _frozen_param9, _frozen_param2, _frozen_param3, _frozen_param0, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  _frozen_param9 = _frozen_param2 = _frozen_param3 = _frozen_param0 = None
    qconv2d_pointwise_default_1 = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type, 0.02212127298116684, 84, _frozen_param8, _frozen_param4, _frozen_param5, _frozen_param1, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type = _frozen_param8 = _frozen_param4 = _frozen_param5 = _frozen_param1 = None


    // Replacable
    mul_3 = torch.ops.aten.mul.Tensor(qconv2d_pointwise_default_1, 52.63383035733631);  qconv2d_pointwise_default_1 = None
    round_2 = torch.ops.aten.round.default(mul_3);  mul_3 = None
    add_1 = torch.ops.aten.add.Tensor(round_2, 141);  round_2 = None
    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None

    convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
    sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_4, 141);  convert_element_type_4 = None
    mul_4 = torch.ops.aten.mul.Tensor(sub_2, 0.018999187275767326);  sub_2 = None
    add_2 = torch.ops.aten.add.Tensor(qconv2d_pointwise_default, mul_4);  qconv2d_pointwise_default = mul_4 = None
    relu = torch.ops.aten.relu.default(add_2);  add_2 = None

    // Replacable
    mul_5 = torch.ops.aten.mul.Tensor(relu, 70.31565792152601);  relu = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 0);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None

    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 0);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.014221583493053913);  sub_3 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_6, (216, 36, 6, 1));  mul_6 = None
    return (inductor_force_stride_order_default,)
```


## Debugging TestPatternMatcher::test_qat_qconv2d_relu

- With is_optional flag
```
# _is_valid_quantized_conv_binary_optimization_pattern

match.nodes
[qconv2d_pointwise_default, relu_1, mul_4, round_3, add_2, clamp_min_2, clamp_max_2, convert_element_type_4]

compute_node.users
{relu_1: None}

binary_node_inputs
(qconv2d_pointwise_default,)
```


## Debugging test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qat_qconv2d_relu

- Expected:
```
entry.pattern:
CallFunction(aten.convolution.default,
CallFunction(aten.mul.Tensor,
CallFunction(aten.sub.Tensor,
CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')),
CallFunction(aten.clone.default,
CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), memory_format=KeywordArg('memory_format')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))



matched node: convolution_1
node.graph (`node.graph.python_code("root").src`):
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param4 = root._frozen_param4
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
    _frozen_param7 = root._frozen_param7
    mul = torch.ops.aten.mul.Tensor(arg8_1, 41.409472476905);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 93);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None
    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

>>> convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
    sub = torch.ops.aten.sub.Tensor(convert_element_type_1, 93);  convert_element_type_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(sub, 0.024149063974618912);  sub = None
    dequantize_per_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param4, _frozen_param2, _frozen_param3, 0, -128, 127, torch.int8);  _frozen_param4 = _frozen_param2 = _frozen_param3 = None
    clone_default = torch.ops.aten.clone.default(dequantize_per_channel, memory_format = torch.channels_last);  dequantize_per_channel = None
>>> convolution = torch.ops.aten.convolution.default(mul_1, clone_default, _frozen_param0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = clone_default = _frozen_param0 = None

    relu = torch.ops.aten.relu.default(convolution);  convolution = None
    mul_2 = torch.ops.aten.mul.Tensor(relu, 115.84175103354825);  relu = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None
    add_1 = torch.ops.aten.add.Tensor(round_2, 0);  round_2 = None
    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None

>>> convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
    sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_3, 0);  convert_element_type_3 = None
    mul_3 = torch.ops.aten.mul.Tensor(sub_1, 0.008632466197013855);  sub_1 = None
    dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param7, _frozen_param5, _frozen_param6, 0, -128, 127, torch.int8);  _frozen_param7 = _frozen_param5 = _frozen_param6 = None
    clone_default_1 = torch.ops.aten.clone.default(dequantize_per_channel_1, memory_format = torch.channels_last);  dequantize_per_channel_1 = None
>>> convolution_1 = torch.ops.aten.convolution.default(mul_3, clone_default_1, _frozen_param1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = clone_default_1 = _frozen_param1 = None

    relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    mul_4 = torch.ops.aten.mul.Tensor(relu_1, 123.52132192691278);  relu_1 = None
    round_3 = torch.ops.aten.round.default(mul_4);  mul_4 = None
    add_2 = torch.ops.aten.add.Tensor(round_3, 0);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
    sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_5, 0);  convert_element_type_5 = None
    mul_5 = torch.ops.aten.mul.Tensor(sub_2, 0.00809576828032732);  sub_2 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_5, (48, 16, 4, 1));  mul_5 = None
    return (inductor_force_stride_order_default,)


m.nodes:
[convert_element_type_3, sub_1, mul_3, dequantize_per_channel_1, clone_default_1, convolution_1]

-> qconv_weight_prepack
```


## Debugging test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qconv2d_add_3

#### Expected:

```
node.name: view -> matched GraphPatternEntry entry.pattern: CallFunction(aten.view.default, KeywordArg('arg'), KeywordArg('size'))
node.name: convert_element_type_10 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_8 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_6 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_1 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))

>>> node.name: mul_3 -> matched GraphPatternEntry entry.pattern: CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale'))

>>> node.name: convolution_1 -> matched GraphPatternEntry entry.pattern: CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))

node.name: convolution -> matched GraphPatternEntry entry.pattern: CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(aten.clone.default, CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), memory_format=KeywordArg('memory_format')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))
node.name: convert_element_type_9 -> matched LoweringPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(aten.clamp_max.default, CallFunction(aten.clamp_min.default, CallFunction(aten.add.Tensor, CallFunction(aten.round.default, CallFunction(aten.mul.Tensor, CallFunction(aten.cat.default, ListOf(CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, Arg(), Arg()), Arg()), Arg())), KeywordArg('dim')), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))

>>> node.name: convert_element_type_7 -> matched LoweringPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(aten.clamp_max.default, CallFunction(aten.clamp_min.default, CallFunction(aten.add.Tensor, CallFunction(aten.round.default, CallFunction(aten.mul.Tensor, CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))

>>> node.name: add_2 -> matched LoweringPatternEntry entry.pattern: CallFunction(aten.add.Tensor, CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()), KeywordArg('accum_after_dequant'))
```

```
pattern: "CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))"

CallFunction(
    aten.convolution.default,
    CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.sub.Tensor,
            CallFunction(
                prims.convert_element_type.default,
                KeywordArg('x'), KeywordArg('x_dq_dtype')
            ),
            KeywordArg('x_zp')
        ),
        KeywordArg('x_scale')
    ),
    CallFunction(
        quantized_decomposed.dequantize_per_channel.default,
        KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')
    ),
    KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups')
)


-- Graph:
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897)
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None


>   convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
>   sub = torch.ops.aten.sub.Tensor(convert_element_type_1, 64);  convert_element_type_1 = None
>   mul_1 = torch.ops.aten.mul.Tensor(sub, 0.021121200174093246);  sub = None
>   dequantize_per_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default(
>       arg4_1, arg2_1, arg3_1, 0, -128, 127, torch.int8);  arg4_1 = arg2_1 = arg3_1 = None
>   convolution = torch.ops.aten.convolution.default(
>       mul_1, dequantize_per_channel, arg0_1,
>       [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = dequantize_per_channel = arg0_1 = None

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1]);  arg8_1 = None
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

>>> add_1 = torch.ops.aten.add.Tensor(round_2, 0);  round_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None

>   convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)
>>> sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_3, 0);  convert_element_type_3 = None
>   mul_3 = torch.ops.aten.mul.Tensor(sub_1, 0.015795549377799034);  sub_1 = None


    convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
>>> sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_4, 0);  convert_element_type_4 = None
    mul_4 = torch.ops.aten.mul.Tensor(sub_2, 0.015795549377799034);  sub_2 = None

    add_2 = torch.ops.aten.add.Tensor(convolution, mul_4);  convolution = mul_4 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None

    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None

    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None

>   dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(
>       arg7_1, arg5_1, arg6_1, 0, -128, 127, torch.int8);  arg7_1 = arg5_1 = arg6_1 = None
>   convolution_1 = torch.ops.aten.convolution.default(
>       mul_3, dequantize_per_channel_1, arg1_1,
>       [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = dequantize_per_channel_1 = arg1_1 = None

    mul_7 = torch.ops.aten.mul.Tensor(convolution_1, 36.10318606192224);  convolution_1 = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None
    return (mul_10,)


-- Graph before lowering:
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
    _frozen_param8 = root._frozen_param8

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1])
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    _frozen_param9 = root._frozen_param9

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

    qconv2d_pointwise_default_1 = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type, 0.021121200174093246, 64, _frozen_param8, _frozen_param2, _frozen_param3, _frozen_param0, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type = _frozen_param8 = _frozen_param2 = _frozen_param3 = _frozen_param0 = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

>>> add_1 = torch.ops.aten.add.Tensor(round_2, 0);  round_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)

>>> sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_3, 0);  convert_element_type_3 = None

    mul_3 = torch.ops.aten.mul.Tensor(sub_1, 0.015795549377799034);  sub_1 = None
    add_2 = torch.ops.aten.add.Tensor(qconv2d_pointwise_default_1, mul_3);  qconv2d_pointwise_default_1 = mul_3 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None

    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None

    qconv2d_pointwise_default = torch.ops.onednn.qconv2d_pointwise.default(
        convert_element_type_2, 0.015795549377799034, 0, _frozen_param9, _frozen_param5, _frozen_param6, _frozen_param1,
        [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type_2 = _frozen_param9 = _frozen_param5 = _frozen_param6 = _frozen_param1 = None


    mul_7 = torch.ops.aten.mul.Tensor(qconv2d_pointwise_default, 36.10318606192224);  qconv2d_pointwise_default = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_10, (216, 36, 6, 1));  mul_10 = None
    return (inductor_force_stride_order_default,)


-- Lowering:
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
    _frozen_param8 = root._frozen_param8

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1])
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    _frozen_param9 = root._frozen_param9

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

    qconv2d_pointwise_default_1 = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type, 0.021121200174093246, 64, _frozen_param8, _frozen_param2, _frozen_param3, _frozen_param0, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type = _frozen_param8 = _frozen_param2 = _frozen_param3 = _frozen_param0 = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

>>> add_1 = torch.ops.aten.add.Tensor(round_2, 0);  round_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(add_1, 0);  add_1 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)

>>> sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_3, 0);  convert_element_type_3 = None

    mul_3 = torch.ops.aten.mul.Tensor(sub_1, 0.015795549377799034);  sub_1 = None
    add_2 = torch.ops.aten.add.Tensor(qconv2d_pointwise_default_1, mul_3);  qconv2d_pointwise_default_1 = mul_3 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None

    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None

>>> qconv2d_pointwise_default = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type_2, 0.015795549377799034, 0, _frozen_param9, _frozen_param5, _frozen_param6, _frozen_param1, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type_2 = _frozen_param9 = _frozen_param5 = _frozen_param6 = _frozen_param1 = None

    mul_7 = torch.ops.aten.mul.Tensor(qconv2d_pointwise_default, 36.10318606192224);  qconv2d_pointwise_default = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None

    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None

    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None

    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_10, (216, 36, 6, 1));  mul_10 = None
    return (inductor_force_stride_order_default,)
```


#### PR:
```
node.name: view -> matched GraphPatternEntry entry.pattern: CallFunction(aten.view.default, KeywordArg('arg'), KeywordArg('size'))
node.name: convert_element_type_10 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_8 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_6 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convert_element_type_1 -> matched GraphPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2'))
node.name: convolution -> matched GraphPatternEntry entry.pattern: CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(aten.clone.default, CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), memory_format=KeywordArg('memory_format')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))
node.name: convert_element_type_9 -> matched LoweringPatternEntry entry.pattern: CallFunction(prims.convert_element_type.default, CallFunction(aten.clamp_max.default, CallFunction(aten.clamp_min.default, CallFunction(aten.add.Tensor, CallFunction(aten.round.default, CallFunction(aten.mul.Tensor, CallFunction(aten.cat.default, ListOf(CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, Arg(), Arg()), Arg()), Arg())), KeywordArg('dim')), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))```
```


```
pattern 1: "CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(aten.clone.default, CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), memory_format=KeywordArg('memory_format')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))"

CallFunction(
    aten.convolution.default,
    CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.sub.Tensor,
            CallFunction(
                prims.convert_element_type.default,
                KeywordArg('x'), KeywordArg('x_dq_dtype')
            ),
            KeywordArg('x_zp')
        ),
        KeywordArg('x_scale')
    ),
    CallFunction(
        aten.clone.default,
        CallFunction(
            quantized_decomposed.dequantize_per_channel.default,
            KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')
        ),
        memory_format=KeywordArg('memory_format')
    ),
    KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups')
)

pattern 2: "CallFunction(aten.convolution.default, CallFunction(aten.mul.Tensor, CallFunction(aten.sub.Tensor, CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')), CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))"

CallFunction(
    aten.convolution.default,
    CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.sub.Tensor,
            CallFunction(
                prims.convert_element_type.default,
                KeywordArg('x'), KeywordArg('x_dq_dtype')
            ),
            KeywordArg('x_zp')
        ),
        KeywordArg('x_scale')
    ),
    CallFunction(
        quantized_decomposed.dequantize_per_channel.default,
        KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')
    ),
    KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups')
)


-- Graph 1:
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param4 = root._frozen_param4
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
    _frozen_param7 = root._frozen_param7

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897)
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None
    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None
    convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
    sub = torch.ops.aten.sub.Tensor(convert_element_type_1, 64);  convert_element_type_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(sub, 0.021121200174093246);  sub = None
    dequantize_per_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param4, _frozen_param2, _frozen_param3, 0, -128, 127, torch.int8);  _frozen_param4 = _frozen_param2 = _frozen_param3 = None
    clone_default = torch.ops.aten.clone.default(dequantize_per_channel, memory_format = torch.channels_last);  dequantize_per_channel = None
    convolution = torch.ops.aten.convolution.default(mul_1, clone_default, _frozen_param0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = clone_default = _frozen_param0 = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1]);  arg8_1 = None
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None
    clamp_min_1 = torch.ops.aten.clamp_min.default(round_2, 0);  round_2 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None

>   convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
>   mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.015795549377799034);  convert_element_type_3 = None

    add_2 = torch.ops.aten.add.Tensor(convolution, mul_3);  convolution = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None

    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None


>   dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param7, _frozen_param5, _frozen_param6, 0, -128, 127, torch.int8);  _frozen_param7 = _frozen_param5 = _frozen_param6 = None
>   convolution_1 = torch.ops.aten.convolution.default(mul_3, dequantize_per_channel_1, _frozen_param1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = dequantize_per_channel_1 = _frozen_param1 = None

    mul_7 = torch.ops.aten.mul.Tensor(convolution_1, 36.10318606192224);  convolution_1 = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_10, (216, 36, 6, 1));  mul_10 = None
    return (inductor_force_stride_order_default,)


-- Graph 2:
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1):

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897)
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

>   convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
|   sub = torch.ops.aten.sub.Tensor(convert_element_type_1, 64);  convert_element_type_1 = None
|   mul_1 = torch.ops.aten.mul.Tensor(sub, 0.021121200174093246);  sub = None
|   dequantize_per_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default(arg4_1, arg2_1, arg3_1, 0, -128, 127, torch.int8);  arg4_1 = arg2_1 = arg3_1 = None
>   convolution = torch.ops.aten.convolution.default(mul_1, dequantize_per_channel, arg0_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_1 = dequantize_per_channel = arg0_1 = None

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1]);  arg8_1 = None
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(round_2, 0);  round_2 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)

    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.015795549377799034);  convert_element_type_3 = None
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None

    mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.015795549377799034);  convert_element_type_4 = None
    add_2 = torch.ops.aten.add.Tensor(convolution, mul_4);  convolution = mul_4 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None

>   convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
|   sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
|   mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None
|   dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(arg7_1, arg5_1, arg6_1, 0, -128, 127, torch.int8);  arg7_1 = arg5_1 = arg6_1 = None
>   convolution_1 = torch.ops.aten.convolution.default(mul_3, dequantize_per_channel_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = dequantize_per_channel_1 = arg1_1 = None

    mul_7 = torch.ops.aten.mul.Tensor(convolution_1, 36.10318606192224);  convolution_1 = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None
    return (mul_10,)


-- Graph before lowering:
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
    _frozen_param8 = root._frozen_param8

    _frozen_param7 = root._frozen_param7

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1])
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None


    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

    qconv2d_pointwise_default = torch.ops.onednn.qconv2d_pointwise.default(
        convert_element_type, 0.021121200174093246, 64, _frozen_param8, _frozen_param2, _frozen_param3, _frozen_param0,
        [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type = _frozen_param8 = _frozen_param2 = _frozen_param3 = _frozen_param0 = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(round_2, 0);  round_2 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None

    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.015795549377799034);  convert_element_type_3 = None
    add_2 = torch.ops.aten.add.Tensor(qconv2d_pointwise_default, mul_3);  qconv2d_pointwise_default = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None
    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None

    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None

    dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(
        _frozen_param7, _frozen_param5, _frozen_param6, 0, -128, 127, torch.int8);  _frozen_param7 = _frozen_param5 = _frozen_param6 = None
    convolution_1 = torch.ops.aten.convolution.default(
        mul_3, dequantize_per_channel_1, _frozen_param1,
        [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = dequantize_per_channel_1 = _frozen_param1 = None

    mul_7 = torch.ops.aten.mul.Tensor(convolution_1, 36.10318606192224);  convolution_1 = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None
    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None
    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_10, (216, 36, 6, 1));  mul_10 = None
    return (inductor_force_stride_order_default,)


-- Lowering:
def forward(self, arg8_1):
    _frozen_param0 = root._frozen_param0
    _frozen_param1 = root._frozen_param1
    _frozen_param2 = root._frozen_param2
    _frozen_param3 = root._frozen_param3
    _frozen_param5 = root._frozen_param5
    _frozen_param6 = root._frozen_param6
>>> _frozen_param7 = root._frozen_param7
    _frozen_param8 = root._frozen_param8

    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(arg8_1, [3, 3], [1, 1])
    getitem = max_pool2d_with_indices[0];  max_pool2d_with_indices = None

    mul = torch.ops.aten.mul.Tensor(arg8_1, 47.34579435625897);  arg8_1 = None
    round_1 = torch.ops.aten.round.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(round_1, 64);  round_1 = None
    clamp_min = torch.ops.aten.clamp_min.default(add, 0);  add = None
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 255);  clamp_min = None

    convert_element_type = torch.ops.prims.convert_element_type.default(clamp_max, torch.uint8);  clamp_max = None

    qconv2d_pointwise_default = torch.ops.onednn.qconv2d_pointwise.default(convert_element_type, 0.021121200174093246, 64, _frozen_param8, _frozen_param2, _frozen_param3, _frozen_param0, [1, 1], [0, 0], [1, 1], 1, 1.0, 0, torch.float32, 'none', [], '');  convert_element_type = _frozen_param8 = _frozen_param2 = _frozen_param3 = _frozen_param0 = None

    mul_2 = torch.ops.aten.mul.Tensor(getitem, 63.30897242520228);  getitem = None
    round_2 = torch.ops.aten.round.default(mul_2);  mul_2 = None

    clamp_min_1 = torch.ops.aten.clamp_min.default(round_2, 0);  round_2 = None
    clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 255);  clamp_min_1 = None
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.uint8);  clamp_max_1 = None
    convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None

    mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.015795549377799034);  convert_element_type_3 = None
    add_2 = torch.ops.aten.add.Tensor(qconv2d_pointwise_default, mul_3);  qconv2d_pointwise_default = None
    mul_5 = torch.ops.aten.mul.Tensor(add_2, 36.10318606192224);  add_2 = None
    round_3 = torch.ops.aten.round.default(mul_5);  mul_5 = None
    add_3 = torch.ops.aten.add.Tensor(round_3, 81);  round_3 = None
    clamp_min_2 = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 255);  clamp_min_2 = None

    convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.uint8);  clamp_max_2 = None
    convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
    sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_6, 81);  convert_element_type_6 = None
    mul_6 = torch.ops.aten.mul.Tensor(sub_3, 0.02769838646054268);  sub_3 = None

>>> dequantize_per_channel_1 = torch.ops.quantized_decomposed.dequantize_per_channel.default(_frozen_param7, _frozen_param5, _frozen_param6, 0, -128, 127, torch.int8);  _frozen_param7 = _frozen_param5 = _frozen_param6 = None
>>> convolution_1 = torch.ops.aten.convolution.default(mul_3, dequantize_per_channel_1, _frozen_param1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_3 = dequantize_per_channel_1 = _frozen_param1 = None

    mul_7 = torch.ops.aten.mul.Tensor(convolution_1, 36.10318606192224);  convolution_1 = None
    round_4 = torch.ops.aten.round.default(mul_7);  mul_7 = None
    add_4 = torch.ops.aten.add.Tensor(round_4, 81);  round_4 = None
    clamp_min_3 = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 255);  clamp_min_3 = None
    convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.uint8);  clamp_max_3 = None
    convert_element_type_8 = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None

    sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_8, 81);  convert_element_type_8 = None
    mul_8 = torch.ops.aten.mul.Tensor(sub_4, 0.02769838646054268);  sub_4 = None
    cat = torch.ops.aten.cat.default([mul_6, mul_8], 1);  mul_6 = mul_8 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat, 36.10318606192224);  cat = None
    round_5 = torch.ops.aten.round.default(mul_9);  mul_9 = None
    add_5 = torch.ops.aten.add.Tensor(round_5, 81);  round_5 = None
    clamp_min_4 = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 255);  clamp_min_4 = None

    convert_element_type_9 = torch.ops.prims.convert_element_type.default(clamp_max_4, torch.uint8);  clamp_max_4 = None
    convert_element_type_10 = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
    sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_10, 81);  convert_element_type_10 = None
    mul_10 = torch.ops.aten.mul.Tensor(sub_5, 0.02769838646054268);  sub_5 = None

    inductor_force_stride_order_default = torch.ops.prims.inductor_force_stride_order.default(mul_10, (216, 36, 6, 1));  mul_10 = None
    return (inductor_force_stride_order_default,)
```




-- pattern:

CallFunction(prims.convert_element_type.default,
CallFunction(aten.clamp_max.default,
CallFunction(aten.clamp_min.default,
CallFunction(aten.add.Tensor,
CallFunction(aten.round.default,
CallFunction(aten.mul.Tensor,
CallFunction(aten.add.Tensor,
CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()),
CallFunction(aten.mul.Tensor,
CallFunction(aten.sub.Tensor,
CallFunction(prims.convert_element_type.default, KeywordArg('accum'), KeywordArg('accum_dq_dtype')), KeywordArg('accum_zp')), KeywordArg('accum_scale'))), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))


-- pattern:

CallFunction(aten.relu.default,
CallFunction(aten.add.Tensor,
CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()), KeywordArg('accum_after_dequant')))


-- pattern:

CallFunction(aten.add.Tensor,
CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()), KeywordArg('accum_after_dequant'))


-- pattern:

CallFunction(prims.convert_element_type.default,
CallFunction(aten.clamp_max.default,
CallFunction(aten.clamp_min.default,
CallFunction(aten.add.Tensor,
CallFunction(aten.round.default,
CallFunction(aten.mul.Tensor,
CallFunction(prims.convert_element_type.default,
CallFunction(prims.convert_element_type.default,
CallFunction(aten.add.Tensor,
CallFunction(onednn.qconv2d_pointwise.default, KeywordArg('x'), KeywordArg('x_scale'), KeywordArg('x_zp'), KeywordArg('packed_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('groups'), KeywordArg('inv_output_scale'), KeywordArg('output_zero_point'), KeywordArg('output_dtype'), KeywordArg('attr'), Arg(), Arg()),
CallFunction(aten.mul.Tensor,
CallFunction(aten.sub.Tensor,
CallFunction(prims.convert_element_type.default, KeywordArg('accum'), KeywordArg('accum_dq_dtype')), KeywordArg('accum_zp')), KeywordArg('accum_scale'))), KeywordArg('convert_dtype_after_inplace_add')), KeywordArg('autocast_output_quant_dtype')), KeywordArg('o_inv_scale'))), KeywordArg('o_zp')), KeywordArg('o_qmin')), KeywordArg('o_qmax')), KeywordArg('o_dtype'))
```



## Debugging test_qcat

```
m.nodes: [
    convert_element_type_5,
    mul_5,
    dequantize_per_channel_1,
    clone_default_1,
    convolution_1
]
pattern:

CallFunction(aten.convolution.default,
CallFunction(aten.mul.Tensor,
CallFunction(aten.sub.Tensor,
CallFunction(prims.convert_element_type.default, KeywordArg('x'), KeywordArg('x_dq_dtype')), KeywordArg('x_zp')), KeywordArg('x_scale')),
CallFunction(aten.clone.default,
CallFunction(quantized_decomposed.dequantize_per_channel.default, KeywordArg('q_weight'), KeywordArg('w_scale'), KeywordArg('w_zp'), KeywordArg('w_axis'), KeywordArg('w_quant_min'), KeywordArg('w_quant_max'), KeywordArg('w_dtype')), memory_format=KeywordArg('memory_format')), KeywordArg('b'), KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('is_transposed'), KeywordArg('out_padding'), KeywordArg('groups'))
```


## Tests

```
pytest -vvv test/inductor/test_mkldnn_pattern_matcher.py
pytest -vvv test/inductor/test_cpu_cpp_wrapper.py

pytest -vvv test/inductor/test_mkldnn_pattern_matcher.py -k test_qat

pytest -vvv test/inductor/test_mkldnn_pattern_matcher.py -k test_qat_qconv2d_add_relu

pytest -vvv test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qat_qconv2d_relu
pytest -vvv test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qat_qconv2d_relu6

pytest -vvv test/inductor/test_cpu_cpp_wrapper.py -k test_qconv2d_relu_cpu_cpp_wrapper
pytest -vvv test/inductor/test_cpu_cpp_wrapper.py::TestCppWrapper::test_qconv2d_maxpool2d_linear_dynamic_cpu_cpp_wrapper
pytest -vvv test/inductor/test_cpu_cpp_wrapper.py::DynamicShapesCppWrapperCpuTests::test_qlinear_relu_cpu_dynamic_shapes_cpp_wrapper
```

### TODO

```
FAILED [5.9788s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qat_qconv2d_add_relu
FAILED [2.4590s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qcat - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [3.0833s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qconv2d_add_3 - AssertionError: Scalars are not equal!
FAILED [2.9437s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qconv2d_add_relu_cpu - AssertionError: Scalars are not equal!
FAILED [2.0541s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qconv2d_relu6_cpu - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [2.0493s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qconv2d_relu_cpu - torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FAILED [2.2190s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qflatten - AssertionError: Scalars are not equal!
FAILED [2.6184s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qlinear_relu_cpu - AssertionError: Scalars are not equal!
FAILED [2.3638s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qlinear_relu_input_dim_exceeds_2 - AssertionError: Scalars are not equal!
FAILED [2.1750s] test/inductor/test_mkldnn_pattern_matcher.py::TestPatternMatcher::test_qmaxpool2d - AssertionError: Scalars are not equal!
FAILED [5.4504s] test/inductor/test_mkldnn_pattern_matcher.py::TestDynamicPatternMatcher::test_qconv2d_maxpool2d_linear_dynamic_cpu
```
