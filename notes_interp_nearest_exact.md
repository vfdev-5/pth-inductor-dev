## Notes on upsample nearest exact decomposition


### Perf results


- Nightly
```
python -u perf_interp_nearest.py output --tag=Nightly --mode=nearest
```

- PR
```
python -u perf_interp_nearest.py output --tag=PR

python -u make_results_table_from_pickles.py output/$(date "+%Y%m%d-%H%M%S")-pr_vs_nightly.md output/XYZ-pr.pkl output/ABC-nightly.pkl

python -u perf_results_compute_speedup_v2.py output/20230706-135210-upsample-nearest-PR-vs-Nightly-speedup.md 'output/20230706-135210-upsample-nearest-PR.pkl' 'output/20230706-135210-upsample-nearest-Nightly.pkl' --compare "Compiled (2.1.0a0+gitd20adf4) PR;Compiled (2.1.0a0+gitd3ba890) Nightly;speed-up PR vs Nightly"
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
