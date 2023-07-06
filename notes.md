# PyTorch Inductor dev notes


## Grid Sampler PR perfs


- Nightly
```
python -u perf_affine_grid_sampler.py output --tag=Nightly
```

- PR
```
python -u perf_affine_grid_sampler.py output --tag=PR


python -u perf_results_compute_speedup_v2.py output/20230706-135210-affine-grid-sampler-PR-vs-Nightly-speedup.md 'output/20230706-135210-affine-grid-sampler-PR.pkl' 'output/20230706-135210-affine-grid-sampler-Nightly.pkl' --compare "Compiled (2.1.0a0+gitd20adf4) PR;Compiled (2.1.0a0+gitd3ba890) Nightly;speed-up PR vs Nightly"
```

