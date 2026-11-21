# CERBERUS-PINN

**Physics-informed neural estimation of heterogeneous anomalous diffusion from short 3D trajectories, with per-window uncertainties.**

Cerberus is a physics-informed neural network (PINN) that reads a short window of a 3D single-particle trajectory and returns three quantities, each with a calibrated uncertainty:

| Head | Quantity | Meaning |
|---|---|---|
| 1 | **H**, Hurst exponent | Directional memory. H > 1/2 persistent, H = 1/2 Brownian, H < 1/2 antipersistent |
| 2 | **log₁₀ D**, generalized diffusion coefficient | Amplitude of the motion, in length² / time^(2H) |
| 3 | **σ_loc**, localization error | Static positional noise of the tracker |

The name comes from the architecture: one shared trunk with three output heads, like the three-headed dog of Greek myth.

The model was built to analyse **Drosophila embryonic hemocytes** (macrophages) imaged in 3D by selective plane illumination microscopy (SPIM). Their motion is heterogeneous anomalous diffusion: H and D change along each track. Cerberus estimates them in sliding windows of 20 frames so the changes can be followed through development.

---

## Contents

- [Why Cerberus](#why-cerberus)
- [Model](#model)
- [Effective diffusivity D_eff](#effective-diffusivity-d_eff)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Application: hemocyte motility in Drosophila embryos](#application-hemocyte-motility-in-drosophila-embryos)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Why Cerberus

Short windows are needed to resolve heterogeneity in time, but in short windows H and D are strongly confounded:

- D has units length²/time^(2H), so its value depends on H and on the time unit.
- Most of the estimation error in log₁₀ D is inherited from the error in H. At 20 frames, the H–log₁₀ D error correlation is about −0.8.
- Mean-squared-displacement (MSD) fits on 20 points are biased and give no reliable error bar.

Cerberus addresses this in three ways:

1. **Joint estimation.** H, D and σ_loc share a trunk and are learned together, so the network can represent their coupling.
2. **Physics-informed loss.** The training loss includes the exact Gaussian likelihood of fractional Gaussian noise (fGn) increments, alongside heteroscedastic supervised terms.
3. **Calibrated uncertainty.** Each head predicts a mean and a standard deviation. Seed-sweep checkpoints can be ensembled to add epistemic uncertainty.

---

## Model

### Generative model

Each axis is an independent FBM with covariance

```
C(s, t) = D (s^2H + t^2H − |t − s|^2H)
```

so that, per axis,

```
MSD(t)             = 2 D t^2H
Var[Δx over lag Δt] = 2 D Δt^2H
```

Observed positions add Gaussian localization error σ_loc. The training simulator can also include **motion blur** from a finite exposure time T_E (duty cycle R = T_E/Δt), using the blurred fGn increment covariance of Backlund et al. (2015).

### Architecture

```
3D window (N positions)
        │
        ▼
 increments, scale-normalised  ──►  shared MLP trunk
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
          H head                  log10 D head              σ_loc head
     (mean, std; bounded)   (mean, std; analytic scale    (mean, std)
                             reconstruction + learned
                             residual)
```

Design points:

- **Scale equivariance.** The network never sees the absolute scale of the trajectory. log₁₀ D is reconstructed analytically from the increment variance, Δt and the blur factor κ(H, R); the network learns only a bounded residual. Because of this, a wide log₁₀ D prior does not pull estimates toward its centre.
- **Rotation invariance.** All three axes share one covariance Σ(H, D, σ_loc), and the input statistics are invariant to rotations of the frame.
- **Training.** Fresh simulations every epoch (no fixed training pool), AdamW, learning-rate scheduling and mixed precision. The fGn likelihood term is ramped in during training.
- **Single configuration point.** Window length (`NUM_POINTS`), Δt range, H range, log₁₀ D range, SNR range and blur ratio R are set in one `Config` dataclass. They are stored in every checkpoint, so each checkpoint documents its own training settings.

### Inference modes

| Mode | Use |
|---|---|
| Fixed window | One estimate per window, e.g. benchmarking on simulated windows |
| Sliding window | One estimate per time point along long tracks. Overlapping windows are combined (common-mode at stride 1, inverse-variance at stride ≥ N). Tracks are split at gaps |
| Experimental | Sliding window plus data ingestion, unit conversion, cleaning and goodness-of-fit diagnostics for real tracking files |

---

## Effective diffusivity D_eff

D cannot be compared across cells with different H, because its units change with H. Cerberus therefore also reports an **effective diffusivity**

```
D_eff = D · Δt^(2H − 1) · [det R(H)]^(1/N)
```

where R(H) is the N × N fGn correlation matrix of the increments. D_eff has ordinary units of length²/time.

- **Fisher-orthogonal to H.** At leading order, the estimation error of D_eff is uncorrelated with that of H. The structural H–D degeneracy is removed rather than averaged over.
- **Model-free form.** For any process with finite second moments and increment covariance Σ,
  ```
  D_eff = (det Σ)^(1/N) / (2 Δt)
  ```
  This is the geometric mean of the one-step prediction-error variances (Durbin–Levinson factorisation): the **innovation diffusivity**, i.e. the unpredictable part of each step. The FBM expression above is a special case.
- **Interpretation.** D_eff behaves as an **effective temperature** of the motion. H measures its **directional memory**.

---

## Repository layout

```
.
├── PINN_FBM_3D.py                    # network definition, simulator, training
├── INFER_PINN_FBM_3D.py              # fixed-window inference
├── INFER_PINN_FBM_3D_SLIDING.py      # sliding-window inference (simulated data)
├── INFER_PINN_FBM_3D_EXPERIMENTAL.py # sliding-window inference (real tracks)
├── GRID_POSTERIOR_FBM_3D.py          # exact grid-posterior reference
├── checkpoints/                      # trained weights, fbm_pinn_N{n}_R{r}*.pt
└── examples/
    └── hemocytes/                    # hemocyte analysis notebooks / scripts
```

> Adjust the names above to match the files in this release.

---

## Installation

```bash
git clone https://github.com/<user>/cerberus-pinn.git
cd cerberus-pinn
python -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy pandas
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. A GPU speeds up training. Inference runs on a CPU.

---

## Quick start

All scripts list their options with `--help`.

### 1. Train

```bash
python PINN_FBM_3D.py --seed 0
```

For an ensemble, train several seeds. Each checkpoint stores its full training configuration.

### 2. Infer on simulated tracks

Input CSV columns: `pid, x, y, z, t` (plus true `H, D` if available for scoring).

```bash
python INFER_PINN_FBM_3D_SLIDING.py \
    --input  tracks.csv \
    --weights checkpoints/fbm_pinn_s0.pt checkpoints/fbm_pinn_s1.pt checkpoints/fbm_pinn_s2.pt \
    --output predictions.csv
```

Passing several checkpoints to `--weights` combines them as an equal-weight Gaussian mixture and splits the reported error into aleatoric, epistemic and total parts.

### 3. Infer on experimental tracks

```bash
python INFER_PINN_FBM_3D_EXPERIMENTAL.py \
    --input  hemocyte_tracks.csv \
    --weights checkpoints/fbm_pinn_s*.pt \
    --pixel_size 0.xx --z_step 0.xx --frame_interval 17.6 \
    --output hemocyte_predictions.csv
```

The experimental front end:

- auto-detects TrackMate, trackpy, Imaris and generic column layouts (override with `--col-*`)
- converts pixels, z-steps and frames to physical units. Output D is in µm²/s^(2H) when the input is in µm and s
- removes NaNs, duplicate `(pid, t)` rows, stuck particles and gap-filled links, and corrects stage drift
- reports per-track **isotropy**, because the model assumes one covariance for all three axes
- **refuses 2D data** (z ≡ 0). Treating 2D data as 3D biases log₁₀ D by −0.477 dex and distorts H
- checks the exposure ratio R of the data against the checkpoint and stops on a mismatch unless `--allow-r-mismatch` is given
- computes a **goodness-of-fit** score from whitened residuals, calibrated against simulation (`gof_pct`)

### Output

One row per `(pid, t)` with point estimates and one-sigma errors:

```
pid, t, H_predicted, H_error, log10D_predicted, log10D_error,
D_eff, D_eff_error, sigma_loc_predicted, gof_pct, ...
```

`log10D_error` already includes the contribution propagated from H. Do not add the H error again.

---

## Application: hemocyte motility in Drosophila embryos

Hemocytes are the macrophages of *Drosophila melanogaster*. During embryogenesis they disperse from the head along stereotyped routes and then settle to patrol the tissues, where they deposit and maintain extracellular matrix and carry out immune surveillance. Earlier work showed that their motion is heterogeneous anomalous diffusion (Korabel et al., 2022). Cerberus was developed to follow how this motion is reorganised over development.

### Data

- 3D hemocyte tracks from SPIM recordings, frame interval Δt = 17.6 s.
- Continuous six-hour recordings (six consecutive 200-frame movies) of control and *LanB1*-deficient (laminin) embryos, plus shorter recordings of further embryos, including collagen IV mutants.
- Sliding 20-frame windows (about 5.9 min) give H, log₁₀ D and D_eff at each time point of each track.

### Analysis pipeline

1. Convert and clean the tracks with `INFER_PINN_FBM_3D_EXPERIMENTAL.py`.
2. Run sliding-window inference with an ensemble of checkpoints trained at the hemocyte frame interval and exposure ratio.
3. Compute D_eff per window.
4. Bin estimates by developmental time and compare genotypes. Confidence intervals come from a whole-curve **cluster bootstrap** over tracks (or embryos), because overlapping windows from one track are correlated.

### Main observations (manuscript in preparation)

- **Warm, then cool.** In the first hour of dispersal, hemocytes gain speed and persistence. Over the patrol phase, D_eff falls steadily.
- **Control cells cool but remember.** D_eff falls about threefold while H stays persistent (H ≈ 0.63 late in development). Faster cells are also more persistent throughout.
- ***LanB1*-deficient cells cool further and forget.** D_eff falls about sixfold, H approaches Brownian values, and the speed–persistence coupling decays and disappears.
- **Collagen IV mutants** cool but keep their persistence. This points to laminin, rather than collagen IV, as the matrix component that sustains persistent patrolling.

The fall of D_eff with retained memory is consistent with an **annealing-like search**: a high effective temperature first disperses the cells, and a falling one then lets them examine their local environment more finely, as in simulated annealing.

---

## Limitations

- **σ_loc is weakly identifiable from 20-frame windows.** In synthetic recovery tests, the σ_loc head largely returns its training prior. Treat it as a nuisance parameter. If localization error matters, estimate it from whole trajectories, or supply a known value and use a model trained for it. For the hemocyte data, whole-track estimates show localization noise is negligible (σ_xy < 0.02 µm, σ_z ≈ 0.03 µm).
- **The model must match the data.** The training ranges of H, SNR, Δt and the exposure ratio R must cover the experiment. Retrain if they do not; the stored configuration makes checking straightforward.
- **The model assumes isotropy.** All three axes share one covariance. Anisotropic motion, or z-precision much worse than xy, violates this. The experimental script reports both but does not correct them.
- **FBM within a window.** Within each window the motion is modelled as FBM. Use `gof_pct` to flag windows where this fails, e.g. smoother-than-FBM steps at lag 1, caused by motion blur or short-time persistence.
- **Overlapping windows are correlated.** Do not treat sliding-window estimates from one track as independent samples. Resample whole tracks or embryos.

---

## Citation

If you use Cerberus, please cite:

```bibtex
@article{korabel_cerberus,
  author  = {Nickolay Korabel, Tom H. Millard, and Thomas Andrew Waigh},
  title   = {From dispersal to patrol: the extracellular matrix shapes an annealing-like search by Drosophila embryonic macrophages},
  note    = {Under review}
}
```

Related work:

- N. Korabel and T. A. Waigh, *Cell Rep. Phys. Sci.* **6**, 102891 (2025). Predecessor tandem-LSTM estimators.
- T. A. Waigh and N. Korabel, *Rep. Prog. Phys.* (2023). Review of heterogeneous anomalous diffusion in biology.
- N. Korabel et al., *Commun. Phys.* (2022). Hemocytes in *Drosophila* embryos move via heterogeneous anomalous diffusion.
- M. P. Backlund, R. Joyner and W. E. Moerner, *Phys. Rev. E* **91**, 062716 (2015). Motion-blur corrections for fBm.
- T. Savin and P. S. Doyle, *Biophys. J.* **88**, 623 (2005). Static and dynamic errors in particle tracking.

## License

MIT (or as specified in `LICENSE`).

## Contact

Nick Korabel, Department of Physics, Astronomy and Mathematics, University of Hertfordshire.
