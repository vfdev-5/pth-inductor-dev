## Notes on upsample nearest exact decomposition


### Perf results


- Nightly
```bash
python -u perf_interp_mode.py output --tag=Nightly --mode=nearest
```

- PR
```bash
python -u perf_interp_mode.py output --tag=PR --mode=nearest

pr_pkl=output/20240301-152339-upsample-nearest-PR.pkl
ni_pkl=output/20240301-151104-upsample-nearest-Nightly.pkl
out_name=$(date "+%Y%m%d-%H%M%S")-upsample-nearest-pr_vs_nightly

python -u make_results_table_from_pickles.py output/${out_name}.md $pr_pkl $ni_pkl
python -u perf_results_compute_speedup_v2.py output/${out_name}-speedup.md $pr_pkl $ni_pkl --compare "Compiled .+ PR;Compiled .+ Nightly;speed-up PR vs Nightly"
```


### Perf measures

- 01/03/2024
```
[-------------------------------------------------------------------------------------- Interpolate, cpu -------------------------------------------------------------------------------------]
                                                                                                                              |  Eager (2.3.0a0+gitb4324ed)   |  Compiled (2.3.0a0+gitb4324ed)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Split arange + dtype using to
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)  |       146.625 (+-6.871)       |        223.244 (+-7.778)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)  |       716.539 (+-37.323)      |       1176.777 (+-47.468)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)      |       598.751 (+-24.380)      |        205.540 (+-9.379)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)      |      3600.418 (+-115.252)     |       1094.375 (+-51.995)

- Nightly:
      Input (1, 3, 500, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (256, 256)  |       146.929 (+-7.091)       |        299.973 (+-8.340)
      Input (1, 3, 300, 400), torch.float32, torch.contiguous_format | mode: nearest, align_corners: None, osize: (600, 700)  |       721.970 (+-41.226)      |       1683.288 (+-68.194)
      Input (1, 3, 500, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (256, 256)      |       599.326 (+-27.796)      |        232.981 (+-9.247)
      Input (1, 3, 300, 400), torch.float32, torch.channels_last | mode: nearest, align_corners: None, osize: (600, 700)      |      3604.446 (+-127.043)     |       1268.975 (+-49.381)

Times are in microseconds (us).
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
