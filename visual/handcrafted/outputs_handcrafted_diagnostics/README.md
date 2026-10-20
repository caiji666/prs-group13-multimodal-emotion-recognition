# Handcrafted feature diagnostics (train-only, read-only)

This diagnostic does not modify the feature set, train a classifier, select a model, or use validation/test feature values.

- Input: `outputs_augmentation\m3_visual_handcrafted_augmented.npz`
- SHA-256: `ce102216922cb6b9790ee2e02d31c2913e2e8f857c1d73041ba84e85b7b54a1b`
- Rows analysed: train only, 960 clips from 16 actors; pooled [960, 320], sequence [960, 16, 320] (full file: [1440, 320] / [1440, 16, 320]; val/test read only for shape checks).
- Schema: 4 x 16 x 5 = 320 (`{subband}_r{row}c{col}_{stat}`).
- NaN: 0, Inf: 0; max |pooled - sequence mean| = 1.34e-06.
- Constant features: 0; near-zero std: std<=1e-12: 0, std<=1e-08: 0, std<=1e-06: 0, scale_aware: 0.

## Correlation (Pearson, train, |r|, upper triangle)

51040 pairs considered; constant features excluded: none.

| Threshold | Pairs | of which same-cell var-energy |
|---|---|---|
| \|r\| >=0.95 | 88 | 48 |
| \|r\| >=0.99 | 55 | 48 |
| \|r\| >=0.999 | 39 | 39 |
| \|r\| >=0.9999 | 20 | 20 |

Highest |r| = 0.99999998. Top 20 pairs (full list in `high_correlation_pairs.csv`):

| Feature A | Feature B | r | same cell | var-energy same cell |
|---|---|---|---|---|
| HH_r3c0_var | HH_r3c0_energy | 0.99999998 | True | True |
| HH_r3c3_var | HH_r3c3_energy | 0.99999998 | True | True |
| HH_r2c3_var | HH_r2c3_energy | 0.99999998 | True | True |
| HH_r0c2_var | HH_r0c2_energy | 0.99999997 | True | True |
| HH_r1c0_var | HH_r1c0_energy | 0.99999996 | True | True |
| HH_r0c1_var | HH_r0c1_energy | 0.99999995 | True | True |
| HH_r2c2_var | HH_r2c2_energy | 0.99999995 | True | True |
| HH_r1c3_var | HH_r1c3_energy | 0.99999994 | True | True |
| HH_r0c3_var | HH_r0c3_energy | 0.99999994 | True | True |
| HH_r2c1_var | HH_r2c1_energy | 0.99999994 | True | True |
| HH_r1c2_var | HH_r1c2_energy | 0.99999993 | True | True |
| HH_r2c0_var | HH_r2c0_energy | 0.99999993 | True | True |
| HH_r3c1_var | HH_r3c1_energy | 0.99999992 | True | True |
| HH_r1c1_var | HH_r1c1_energy | 0.99999992 | True | True |
| HH_r3c2_var | HH_r3c2_energy | 0.99999988 | True | True |
| HH_r0c0_var | HH_r0c0_energy | 0.99999986 | True | True |
| LH_r0c0_var | LH_r0c0_energy | 0.99992911 | True | True |
| LH_r2c2_var | LH_r2c2_energy | 0.99992616 | True | True |
| LH_r2c1_var | LH_r2c1_energy | 0.99990633 | True | True |
| HL_r0c0_var | HL_r0c0_energy | 0.99990425 | True | True |

## Variance vs energy (64 subband x cell groups)

- **Per frame (mathematical):** energy = var + mean^2 holds in 64/64 groups (rtol 1e-05, atol 1e-06); max |residual| = 1.54e-06.
- **After temporal pooling:** pooled_energy = pooled_var + mean_t(frame_mean^2) holds in 64/64 groups (max |residual| 9.75e-07). Using pooled_mean^2 instead would be wrong (max |residual| 0.291).
- **Pooled statistical correlation:** median |r(var, energy)| = 0.999533, minimum 0.056825; groups with |r| >= threshold: {'0.95': 48, '0.99': 48, '0.999': 39, '0.9999': 20}; groups whose pooled var and energy columns are identical: 0.
- Per subband median |r|: LL 0.352555, LH 0.999489, HL 0.999572, HH 1.000000.
- A high correlation is a statistical relation over clips; it does not mean the two pooled columns are identical.

## Scale by statistic type and subband

| Statistic | Count | Median mean | Median std | Min std | Max std | Constant | Near-zero (scale-aware) | High-corr involvement |
|---|---|---|---|---|---|---|---|---|
| mean | 64 | 0.0005778 | 0.001912 | 4.13e-05 | 0.2896 | 0 | 0 | 20 |
| var | 64 | 0.001011 | 0.0004265 | 1.329e-05 | 0.1162 | 0 | 0 | 57 |
| energy | 64 | 0.001027 | 0.0004272 | 1.33e-05 | 0.7117 | 0 | 0 | 73 |
| entropy | 64 | 3.929 | 0.308 | 0.1493 | 0.9748 | 0 | 0 | 5 |
| edge | 64 | 0.08619 | 0.02021 | 0.00374 | 0.2125 | 0 | 0 | 21 |

| Subband | Count | Median mean | Median std | Min std | Max std | Constant | Near-zero (scale-aware) | High-corr involvement |
|---|---|---|---|---|---|---|---|---|
| LL | 80 | 0.8992 | 0.1616 | 0.01529 | 0.9748 | 0 | 0 | 36 |
| LH | 80 | 0.001647 | 0.001912 | 0.0001282 | 0.6376 | 0 | 0 | 45 |
| HL | 80 | 0.003679 | 0.002796 | 8.169e-05 | 0.5801 | 0 | 0 | 47 |
| HH | 80 | 0.000131 | 8.869e-05 | 1.329e-05 | 0.6035 | 0 | 0 | 48 |

## Conclusion (rule fixed in advance)

Rule: constant features > 0 or >= 16 features in pairs with |r| >= 0.9999. Constant features: 0; features in |r| >= 0.9999 pairs: 40.

Substantial near-duplication was found by the pre-defined rule; consider discussing a pre-defined reduced handcrafted control before the formal Level-2 run. No feature is removed here, and test results must not decide any removal.

No feature was removed. Validation and test feature values were not used, and test results must never decide which features to remove.
