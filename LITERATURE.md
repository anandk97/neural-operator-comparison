# Published results we compare against

Numbers are copied from the cited papers, and each comes with its paper's own setup. Where our protocol differs
(epochs, resolution, metric), the difference is noted so the comparison isn't over-read.

## Standard benchmarks (relative L2, lower is better)

From Wu et al., *Transolver*, ICML 2024, Table 2, and Luo et al., *Transolver++*, ICML 2025. Both trained 500 epochs.
Transolver++ used batch size 4 and 4 to 8 layers depending on the task.

| Model | Elasticity | Airfoil | Navier–Stokes | Darcy |
|---|---:|---:|---:|---:|
| FNO (Li et al. 2021) | — | — | 0.1556 | 0.0108 |
| Geo-FNO (Li et al. 2023) | 0.0229 | 0.0138 | 0.1556 | 0.0108 |
| F-FNO (Tran et al. 2023) | 0.0263 | 0.0078 | 0.2322 | 0.0077 |
| U-NO (Rahman et al. 2023) | 0.0258 | 0.0078 | 0.1713 | 0.0113 |
| LSM (Wu et al. 2023) | 0.0218 | 0.0059 | 0.1535 | 0.0065 |
| Galerkin Transformer (Cao 2021) | 0.0240 | 0.0118 | 0.1401 | 0.0084 |
| GNOT (Hao et al. 2023) | 0.0086 | 0.0076 | 0.1380 | 0.0105 |
| Transolver (Wu et al. 2024) | 0.0064 | 0.0053 | 0.0900 | 0.0057 |
| Transolver++ (Luo et al. 2025) | 0.0052 | 0.0048 | 0.0719 | 0.0049 |

A dash means the paper reports no result, because plain FNO needs a regular grid.

## DeepONet vs FNO: Lu et al., CMAME 2022, "A comprehensive and fair comparison of two neural operators"

Relative L2 (%), mean ± std:

| Problem | DeepONet | POD-DeepONet | FNO |
|---|---:|---:|---:|
| Burgers ν = 0.1 (128 points) | 2.15 ± 0.09 | 1.94 ± 0.07 | 1.93 ± 0.04 |
| Darcy, piecewise constant (29 × 29) | 2.98 ± 0.03 | 2.32 ± 0.03 | 2.41 ± 0.03 |
| Advection I, square wave (40 points) | 0.22 ± 0.03 | 0.04 ± 0.00 | 0.66 ± 0.10 |

- **Noise:** 0.1% Gaussian noise on the advection input raised FNO's error to 270%, while DeepONet's error was 0.36%.
- **Geometry:** plain FNO cannot be applied to complex geometries without extensions (dgFNO+). On those domains
  DeepONet and POD-DeepONet were more accurate.

## Discontinuities: Lanthaler, Molinaro, Hadorn & Mishra, ICLR 2023

Median relative L1 (%):

| Problem | DeepONet | Shift-DeepONet | FNO |
|---|---:|---:|---:|
| Linear advection, square waves (2048 points) | 7.95 | 2.76 | 0.71 |
| Inviscid Burgers | 28.5 | 7.83 | 1.57 |
| Lax–Sod shock tube | 4.22 | 2.76 | 1.56 |

**Theory:** operators with *linear reconstruction*, such as DeepONet and PCA-Net, need a number of basis functions that
grows quickly as accuracy improves when solutions carry moving discontinuities (slow Kolmogorov n-width decay).
FNO and Shift-DeepONet reconstruct nonlinearly and avoid this lower bound.

**How our advection task differs:** we use heights 0.2–0.8 and widths 0.05–0.3 as they do, but centres uniform on [0, 1),
speed 1, T = 0.5, and 1024 points. We report relative L2.

**Why this contrasts with Lu et al.:** at 40 grid points and widths of 0.3–0.6, a jump spans few basis functions, so
DeepONet does well there. At 1024–2048 points with narrow waves, it does not.

## Other variants (their own benchmarks)

- **Local-kernel FNO,** Liu-Schiaffini et al., ICML 2024: adding local integral and differential kernels to FNO lowers
  error on Darcy, Navier–Stokes and shallow-water problems, where FNO's global, band-limited kernels blur local
  features.
- **DeepOKAN,** Abueidda et al., CMAME 2025: lower error than DeepONet on 1D sinusoidal waves, 2D orthotropic
  elasticity and a transient Poisson problem.
- **KANO,** Lee et al., 2025: FNO fails on position-dependent operators such as x²f − f″, and KANO recovers them,
  sometimes in closed form. Its tests are small 1D problems and quantum Hamiltonians.
- **PirateNet,** Wang et al., JMLR 2024: deeper physics-informed networks become trainable with identity-initialised
  adaptive residual connections, reaching state-of-the-art PINN results on Allen–Cahn, Korteweg–de Vries,
  Gray–Scott, Ginzburg–Landau and lid-driven cavity flow.
- **Robustness,** Shikhman, arXiv:2601.11428 (2026): across 750 trained FNO, DeepONet and CNO models, in-distribution
  accuracy did not predict robustness to shifts in coefficients, boundary conditions, resolution or time horizon.
  Failure patterns depended on both the architecture and the PDE family.
