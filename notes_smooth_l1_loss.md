# Notes. Inductor, smooth_l1_loss task

Refs:
- https://pytorch.org/get-started/pytorch-2.0/
- https://github.com/pytorch/workshops/blob/master/NeurIPS_2022/TorchInductor%20Deep%20Dive%20-%20NeurIPS%202022.pdf
- https://www.youtube.com/watch?v=egZB5Uxki0I&ab_channel=EdwardZ.Yang%27sPyTorchandPL


Refs = Python API
Decompositions = ATEN API

## Test ref implementation
```
pytest -vvv test/test_ops.py -k smooth
```

## Check the implementation

```
TORCH_LOGS=+inductor python check_smooth_l1_loss.py
```


## Debugging

```
TORCH_COMPILE_DEBUG=1 python -m debugpy --wait-for-client --listen 5678 check_smooth_l1_loss.py
```

- Compile part:
```
compile ->

    torch._dynamo.optimize with _TorchCompileInductorWrapper("default", None, dynamic=False) ->

        backend = get_compiler_fn(backend) where backend is _TorchCompileInductorWrapper("default", None, dynamic=False) ->

        backend is debug_wrapper over _TorchCompileInductorWrapper


        _optimize_catch_errors(...) ->

            OptimizeContext(...) ->

    _TorchDynamoContext.__call__(func) ->

return wrapper
```

- Exec part:
```
_TorchDynamoContext.__call__::_fn ->

    OptimizeContext.on_enter() ->


```



## Compiled code


- case without `if beta==0.0` short-cut:
```
@persistent_reduction(
    size_hints=[1, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 20
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.abs(tmp2)
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 20.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
''')
```

- case with `if beta==0.0` short-cut:
```
@persistent_reduction(
    size_hints=[1, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 20
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.abs(tmp2)
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 20.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
''')
```



## useful cmds

```
find /tmp/torchinductor_root/triton/ -type f -newermt 2023-05-23


cat /tmp/torchinductor_root/triton/0/24887aa3330e5eabbfe05ad0f48db5d4/triton_.ttir
cat /tmp/torchinductor_root/triton/0/24887aa3330e5eabbfe05ad0f48db5d4/triton_.ttgir
cat /tmp/torchinductor_root/triton/0/24887aa3330e5eabbfe05ad0f48db5d4/triton_.llir
```