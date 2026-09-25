# Neural operator comparison

Which operator-learning architecture works best on which kind of PDE data, and why. The benchmark covers five
families, each with its original version and a newer variant that targets a known weakness, trained under one
protocol on six standard tasks. The results sit next to the published numbers they build on.

Work in progress. Results and the write-up will be at https://anandk97.github.io/projects/.

## Models

| Family | Base | Newer variant | Implementation |
|---|---|---|---|
| DeepONet | DeepONet, Lu et al., *Nat. Mach. Intell.* 2021 | Shift-DeepONet, Lanthaler et al., ICLR 2023 | written from the papers (`noc/deeponet.py`) |
| Fourier NO | FNO, Li et al., ICLR 2021 | Local-kernel FNO, Liu-Schiaffini et al., ICML 2024 | [neuraloperator](https://github.com/neuraloperator/neuraloperator) library (MIT) |
| KAN | DeepOKAN, Abueidda et al., *CMAME* 2025 | KANO, Lee et al., arXiv:2509.16825 | written from the papers; KANO adapted to multi-channel 2D (see `noc/spectral.py`) |
| Transformer | Transolver, Wu et al., ICML 2024 | Transolver++, Luo et al., ICML 2025 | adapted from [thuml/Transolver](https://github.com/thuml/Transolver) and [thuml/Transolver_plus](https://github.com/thuml/Transolver_plus) (MIT) |
| PINN | PINN, Raissi et al., *J. Comput. Phys.* 2019 | PirateNet, Wang et al., *JMLR* 2024 | written from the papers (`pinn.py`); these solve one instance at a time |

## Tasks

| Task | Property it isolates | Data |
|---|---|---|
| Burgers, ν = 0.1 | smooth, periodic | FNO (Li et al. 2021) |
| Advection of square waves | transported discontinuities | generated from the exact solution, following Lanthaler et al. 2023 |
| Darcy flow, 85 × 85 | rough coefficient, non-periodic, elliptic | FNO |
| Navier–Stokes, ν = 1e-5 | time-dependent, multiscale | FNO |
| Airfoil | structured, deformed mesh | Geo-FNO (Li et al. 2023) |
| Elasticity | unstructured point cloud | Geo-FNO |

## Running

```bash
uv sync
uv run python data_fetch.py
uv run python train.py --task darcy --model fno --epochs 300
uv run python pinn.py --task burgers --model piratenet --instances 3
uv run python run_all.py main      # or: ablation, eval, scaling, pinn
uv run python export_web.py        # results -> web/operators-data.json
```

## Credits

The datasets belong to their authors (FNO, Geo-FNO), and third-party code keeps its license (see `third_party/`).
Every model here is cited above; see the docstrings for exactly what was adapted and how.
