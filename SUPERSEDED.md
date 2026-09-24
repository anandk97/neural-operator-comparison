# Superseded runs

Kept in runs/_superseded for the record; not used in results.

- **Burgers: DeepONet, Shift-DeepONet and DeepOKAN.** These were trained before the trunk got the periodic Fourier
  features that Lu et al. (2022) use on periodic domains. The fix was applied before any other DeepONet-family run.
- **Darcy: Transolver and Transolver++; Navier–Stokes: Transolver.** These were trained without the unified
  positional encoding (distances to an 8 x 8 reference grid) that the authors' Darcy and Navier–Stokes configurations
  enable. Without it, Darcy training error stalled at about 10–14%.
