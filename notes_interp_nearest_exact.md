## Notes on upsample nearest exact decomposition


### Perf results

```bash
python -u perf_interp_nearest.py
```

### Tests

```
pytest -vvv test/test_decomp.py -k "nearest and exact"
pytest -vvv test/inductor/test_torchinductor_opinfo.py -k "nearest and exact"
pytest -vvv test/functorch/test_aotdispatch.py -k "nearest and exact"
pytest -vvv test/functorch/test_ops.py -k "nearest and exact"
pytest -vvv test/test_meta.py -k "nearest and exact"
pytest -vvv test/test_proxy_tensor.py -k "nearest and exact"
pytest -vvv test/test_decomp.py
pytest -vvv test/functorch/test_vmap.py -k "nearest and exact"
```
