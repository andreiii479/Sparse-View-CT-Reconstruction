# Sparse-View CT Reconstruction

Reconstructing CT images from a small number of noisy X-ray projections, written in plain MATLAB/Octave with no toolboxes and no built-in solvers.

The solver combines **FISTA** (accelerated proximal gradient) with **iteratively reweighted total variation**. It sets its own regularization strength based on how undersampled each problem is.

![Ground truth, sinogram and reconstruction for tiers 1–4](docs/results.png)

<sub>Each row is one tier (1 → 4, top to bottom). Columns: ground truth · measured sinogram · reconstruction.</sub>

## The problem

A CT scanner measures line integrals of an object's attenuation from several angles. Together these measurements form a *sinogram* `b`. Reconstruction means recovering the image `x` from

```
b = A·x + noise
```

where `A` is the forward projection operator. With few angles there are fewer measurements than unknowns, so the system is underdetermined and a plain least-squares solve produces streaky, noisy images. The goal is to recover a clean image anyway.

The project has four difficulty tiers:

| Tier | Image size | Angles | Detector bins | Noise σ | Measurements / pixels |
|:----:|:----------:|:------:|:-------------:|:-------:|:---------------------:|
| 1 | 64 × 64   | 90 | 91  | 0.01 | 2.00 |
| 2 | 128 × 128 | 60 | 182 | 0.03 | 0.67 |
| 3 | 128 × 128 | 20 | 182 | 0.06 | 0.22 |
| 4 | 256 × 256 | 30 | 363 | 0.10 | 0.17 |

### Constraints

These rules ruled out the usual shortcuts, so every part of the solver is written by hand:

- No `pcg`, `bicgstab`, `gmres`, `lsqr`, `svd`, `eig`, `radon`, `iradon`, `\`, `inv` or `pinv`
- No toolboxes (Image Processing, Optimization, …)
- The only access to `A` is through `forward_op` (A·x) and `adjoint_op` (Aᵀ·b)
- Each reconstruction must finish within **120 s**, and the output must be non-negative

## Method

All of the solver code is in [`reconstruct.m`](reconstruct.m). It solves

```
minimize   ½‖A·x − b‖²  +  λ · Σ wᵢ |∇x|ᵢ      subject to  x ≥ 0
```

1. **Step size from power iteration.** It runs 20 iterations of power iteration on `AᵀA` to estimate the Lipschitz constant `L`, then uses a step size of `1 / (1.01·L)`.
2. **FISTA outer loop.** Each iteration takes a gradient step on the data term, applies the TV proximal operator, projects onto `x ≥ 0`, and applies Nesterov momentum. It stops early when the relative change falls below `1.6e-5`.
3. **Weighted TV proximal step.** A few iterations of Chambolle's dual projection algorithm with per-pixel weights `W`. The dual variables are warm-started between FISTA iterations, so only 5 inner iterations are needed each time.
4. **Reweighting.** After each FISTA round, the weights are set to `W = 1 / (|∇x| + ε)` and normalized. Edges that the previous round found get a smaller penalty, and flat regions get a larger one. This moves the regularizer closer to an ℓ₀-style gradient-sparsity prior, which keeps edges sharp instead of letting TV blur them.
5. **Adaptive parameters.** λ, ε and the number of reweighting rounds are chosen from the sampling ratio `(angles × detectors) / pixels`. Well-sampled problems get light regularization, and very undersampled ones get heavy regularization.

| Sampling ratio | λ | ε | Reweighting rounds |
|:--------------:|:----:|:----:|:--:|
| ≥ 1.0  | 0.02 | 0.02 | 4 |
| ≥ 0.35 | 0.04 | 0.03 | 5 |
| ≥ 0.19 | 0.25 | 0.10 | 6 |
| < 0.19 | 0.70 | 0.06 | 4 |

A time guard stops the solver at 115 s so it always stays within the limit.

## Results

Results on the practice problems, run with GNU Octave 9.4 on a laptop CPU. Score = `0.6·PSNR + 0.4·(100·SSIM)`.

| Tier | PSNR (dB) | SSIM | Score | Iterations | Time |
|:----:|:---------:|:----:|:-----:|:----------:|:----:|
| 1 | 65.84 | 0.9999 | 79.50 | 308  | 7.3 s  |
| 2 | 55.84 | 0.9990 | 73.46 | 503  | 19.9 s |
| 3 | 51.45 | 0.9991 | 70.84 | 1008 | 16.2 s |
| 4 | 56.19 | 0.9996 | 73.70 | 800  | 62.0 s |

## Running it

### With Docker (matches the grading environment)

```bash
make build   # build the ct-student image (Octave 11.1)
make p1      # run tier 1 practice; also p2, p3, p4
make dev     # interactive shell inside the container
```

Each run writes a reconstruction PNG and a text report to `results/`. The Docker image copies the code in when it's built, so rebuild after you change anything. The `make p*` targets already rebuild for you. See [`DOCKER_CHEATSHEET.md`](DOCKER_CHEATSHEET.md) for the plain `docker` commands.

### With a local Octave or MATLAB

```bash
octave --no-gui --eval "tier=1; use_practice=true; run_tier"
```

## Project structure

```
reconstruct.m        Reconstruction algorithm (FISTA + reweighted TV)
run_tier.m           Driver: load a tier, reconstruct, score, save results
operators/           forward_op (A·x), adjoint_op (Aᵀ·b), apply_ata
evaluation/          PSNR, SSIM and composite score
utils/               Problem loading and visualization helpers
data/practice/       Practice problems with ground truth (.mat) and previews
docs/                Images for this README
Dockerfile, Makefile, docker-compose.yml
```

## Acknowledgements

This was a university course project. The course provided the project skeleton: the operators, evaluation metrics, driver script, practice data and Docker setup. The reconstruction algorithm in `reconstruct.m` is my own work.
