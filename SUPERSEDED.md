# Superseded runs

Kept in runs/_superseded for the record; not used in results.

- **Burgers: DeepONet, Shift-DeepONet and DeepOKAN.** These were trained before the trunk got the periodic Fourier
  features that Lu et al. (2022) use on periodic domains. The fix was applied before any other DeepONet-family run.
- **Darcy: Transolver and Transolver++; Navier–Stokes: Transolver.** These were trained without the unified
  positional encoding (distances to an 8 x 8 reference grid) that the authors' Darcy and Navier–Stokes configurations
  enable. Without it, Darcy training error stalled at about 10–14%.
- **Navier–Stokes: Transolver at the authors' settings, without gradient clipping** (as in their NS script). With our
  teacher-forced one-step training at batch size 2, training became unstable from about epoch 20 and diverged to NaN
  at epoch 118. It was re-run with clipping at 0.1, the value the authors use for their other benchmarks.
