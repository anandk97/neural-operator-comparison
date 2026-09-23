# Neural operator comparison: plan

Goal: find which operator architecture works best on which kind of data, and explain *why*, using published comparisons
plus our own controlled runs. This becomes Project 3 on anandk97.github.io. Cite every source properly.

## The 10 models (5 families × base + newer variant)

| Family | Base | Newer variant | Weakness the variant targets |
|---|---|---|---|
| DeepONet | DeepONet (Lu et al., *Nat. Mach. Intell.* 2021) | Shift-DeepONet (Lanthaler, Molinaro, Hadorn, Mishra, ICLR 2023) | The output is a *linear* combination of trunk basis functions, which cannot represent moving discontinuities (slow Kolmogorov n-width decay) |
| Fourier NO | FNO (Li et al., ICLR 2021) | FNO with local integral + differential kernels (Liu-Schiaffini et al., ICML 2024) | Spectral truncation is global; it loses sharp local features and has trouble with non-periodic boundaries |
| PINN | PINN (Raissi, Perdikaris, Karniadakis, *JCP* 2019) | PirateNet (Wang, Li, Chen, Perdikaris, *JMLR* 2024) | Optimization pathologies and spectral bias in deep MLPs. Uses no data, and each PDE instance is a separate solve |
| KAN | DeepOKAN (Abueidda, Pantidis, Mobasher, *CMAME* 2025) | KANO (Lee et al., arXiv 2509.16825, 2025) | Pure-spectral or pure-spatial bases; adds symbolic interpretability |
| Transformer | Transolver (Wu et al., ICML 2024) | Transolver++ (Luo et al., ICML 2025) | Irregular meshes; physics-attention over learned slices. ++ scales to large meshes |

Caveats:
- PINNs are not operators. They are included as a data-free reference for accuracy per unit of compute on a few test instances.
- KANO code availability is unverified. The fallback is a KAN-FNO hybrid.

## Datasets: each one isolates one property

| Dataset | Property being tested | Source for published numbers |
|---|---|---|
| 1D Burgers (ν = 1e-3) | Smooth, periodic, spectrally decaying | FNO paper; Lu et al. 2022 fair comparison |
| 1D advection with discontinuous initial data | Transported discontinuities (n-width) | Lanthaler et al. 2023; PDEBench |
| 2D Darcy, piecewise-constant coefficient | Rough input, non-periodic elliptic | FNO paper; PDEBench; Transolver |
| 2D Navier–Stokes vorticity | Time-dependent, multiscale | FNO paper; Transolver |
| Airfoil or elasticity (point cloud / mesh) | Irregular geometry | Geo-FNO; Transolver/++ |

Optional add-on: Kuramoto–Sivashinsky from the CTF project, which ties in Project 1.

## Experiments (the "why")

1. **Matched parameter budgets.** Relative L2 error on every dataset, reported next to the published numbers.
2. **Data scaling.** Train on 100, 300 and 1,000+ samples to measure how much each architecture's inductive bias helps.
3. **Resolution transfer.** Train at one resolution and test at another (the discretization-invariance claim).
4. **Error spectrum.** Error by Fourier mode, to show where each model's error lives: high modes, near shocks, or at boundaries.
5. **Noise robustness.** Add input noise; Lu et al. 2022 report that FNO degrades.
6. **Cost.** Training time, inference time and GPU memory on the RTX 4080 Laptop (12 GB).

## Literature to anchor on

- Lu et al., "A comprehensive and fair comparison of two neural operators", CMAME 2022 (arXiv 2111.05512)
- Lanthaler et al., "Nonlinear reconstruction for operator learning of PDEs with discontinuities", ICLR 2023 (arXiv 2210.01074)
- Takamoto et al., PDEBench, NeurIPS 2022
- Kovachki et al., "Neural operator: learning maps between function spaces", JMLR 2023
- Shikhman, "Diagnosing failure modes of neural operators across diverse PDE families", arXiv 2601.11428 (2026)
- Numerical PDE solvers vs neural solvers, arXiv 2507.21269 (for honest framing)
