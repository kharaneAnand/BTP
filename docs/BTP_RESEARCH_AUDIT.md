# BTP Research Audit and Improvement Plan

**Working branch:** `research/best-results`  
**Reference repository:** `kharaneAnand/OTFS-NeuroRx` (read-only)  
**Scope:** Static source review of the research pipeline and saved result artifacts. No training or GPU execution is claimed by this audit.

## 1. What the saved evidence currently supports

The saved V1 test evaluation has 135 frames (15 per each of nine SNR/velocity cells). The reported OAMP-DL checkpoint obtains BER 0.03296, SER 0.06345, and symbol NMSE 0.10175. It beats the MMSE baseline on all three aggregate metrics and in all nine saved condition summaries.

The receiver bank does not currently demonstrate a useful adaptive-selection gain. On the original test run, the controller pipeline reports BER 0.03534 versus 0.03296 for always using OAMP-DL. On the independently generated seed-43 test split, it reports BER 0.03939 versus 0.03606 for always OAMP-DL. The controller therefore should not be presented as improving receiver accuracy on the existing evidence.

Few-shot adaptation is also inconclusive: paired confidence intervals for the reported conditions cross zero. It should remain experimental and should not be described as a demonstrated gain.

## 2. High-priority technical findings

### A. The learned detector named OAMP-DL is not a direct unfolding of the analytical OAMP implementation

The analytical detector in `src/receivers/oamp.py` explicitly forms a regularized linear estimator, applies a trace-based orthogonalization scale, and propagates an effective variance estimate into the QPSK posterior-mean denoiser.

The learned detector in `models/oamp/oamp_dl.py` instead uses a column-normalized matched-filter residual update, followed by learned step, damping, and variance scalars. Its source does not implement the analytical detector's trace-based orthogonalization operation or its linear-estimator construction. It is therefore safer to describe the current model as a **learned unfolded matched-filter / iterative QPSK detector** until a mathematical equivalence is established or a true OAMP unfolding is implemented and tested.

This distinction matters: OAMP's defining construction relies on a decorrelated linear estimator and a divergence-free nonlinear estimator (or an equivalent orthogonalization construction). See Ma and Ping, *Orthogonal AMP*, and the OTFS UAMP detector literature:
- https://arxiv.org/abs/1602.06509
- https://ieeexplore.ieee.org/document/9492800

**Action:** preserve the current checkpoint and result as a strong learned-iterative baseline. Implement a separate mathematically faithful OAMP-DL/UAMP-DL candidate, with explicit equations, unit tests for orthogonalization/variance updates, and a separate output directory. Do not overwrite or relabel historical artifacts.

### B. The controller's adaptation recommendation could select an unsupported receiver

The previous controller chose its ADAPT recommendation by maximum *raw* confidence margin, even though margins are on receiver-specific scales. It could recommend PI-EGNN or the original GNN although the implemented few-shot adapter only updates OAMP-DL's scalar controls.

**Change on this branch:** candidate adaptation is ranked by each receiver's own standardized confidence score and recommendations are restricted to receivers with an implemented adaptation routine. If no supported adapter exists, the controller returns KEEP rather than emitting an unsupported ADAPT recommendation. Tests were updated for this contract.

### C. Controller selection is not currently an environment-shift-aware policy

The environment distance is computed and logged, but it does not affect KEEP/SWITCH/ADAPT. The current controller is driven by a receiver confidence margin and frozen overall BER gates. Therefore the current evidence supports a reliability-triggered candidate-check prototype, not a validated environment-shift-aware adaptation system.

**Action:** keep the shift detector and reliability estimator as separate modules. Before connecting them, calibrate the shift score on held-out nominal streams and physically meaningful shifted channels (fractional Doppler spread, path count/delay profile, Doppler distribution, CSI error, pilot SNR, and noise mismatch). Report false-alarm rate, detection delay, and power at a predeclared threshold. Do not use synthetic additive corruption alone as proof of channel-shift detection.

### D. Evaluation comparisons should use paired uncertainty estimates

Per-frame receiver predictions are available, so receiver comparisons should be made on paired per-frame differences, not inferred from overlap/non-overlap of separate mean confidence intervals. The per-condition groups contain only 15 frames; uncertainty is consequently large.

**Action:** add paired bootstrap confidence intervals (resampling frames within each condition and preserving receiver pairing), report BER/SER/NMSE deltas, and pre-register the primary metric and acceptance criterion. Keep the test split sealed until the model and hyperparameters are frozen.

### E. Generalization evidence is narrow

Seed 43 is a useful independently generated dataset check, but the channel/SNR/velocity ranges are the same as V1. It does not establish robustness to a new channel family or an out-of-range operating regime.

**Action:** test separately on (i) unseen random seeds within the same distribution, (ii) held-out velocity/SNR values between and beyond training points, and (iii) channel-model shifts (path count, delay profile, Doppler spectrum, fractional-Doppler spread, pilot mismatch, and CSI estimation error). Keep all generated data in new experiment directories.

### F. The original controller BER gate used test-set metrics

The original `configs/reliability_controller_v1.json` populated its fixed BER gate with the same aggregate values reported by the held-out test evaluation files. Although test outcomes were not used to tune the confidence thresholds, using test-derived BER to decide which candidates are eligible still lets test labels influence routing.

**Correction on this branch:** recomputed the gate values from the 135 validation rows per receiver in `experiments/reliability_detector/train_validation_features.csv`. The controller configuration now identifies the validation split as the source and explicitly states that test labels are not used to configure the gate. The held-out test split can now be used for the final policy evaluation.

Validation-derived BER gate:
- MMSE: 0.06475443
- OAMP-DL: 0.02833132
- Original GNN: 0.03270934
- PI-EGNN: 0.03556763

These values are calculated from saved per-frame validation metrics; they are not newly trained or newly inferred results.

### G. The legacy controller evaluator scored the post-decision receiver on the current frame

The sequential policy observes a frame using `active_model_before`, then makes a decision that changes `active_model_after` for the next frame. The legacy evaluator joined offline metrics against `active_model_after`, which incorrectly credited the newly selected receiver with the frame that had already been processed.

**Correction on this branch:** `evaluations/supervisor/evaluate_reliability_controller.py` now records `receiver_used_for_frame` and joins the current frame's metrics against that receiver. Switch destinations remain reported from `active_model_after`, which is the correct next-frame state. The unified `run_pipeline.py` batch summary already scores `receiver.used_for_this_frame` and was not affected by this specific alignment bug.

## 3. Experimental roadmap (ordered by expected research value)

1. **Reproducibility and audit tests**
   - Validate dataset file hashes, split disjointness, condition balance, shapes, finite values, and absence of target/ground-truth inputs in live inference.
   - Run the full unit-test suite and record Python/PyTorch/CUDA versions and hardware.
   - Add paired receiver comparison reports generated only from saved per-frame predictions.

2. **Correct model naming and a faithful model-driven baseline**
   - Keep the current learned iterative detector as a separate baseline.
   - Implement a mathematically faithful OAMP or UAMP detector using the OTFS system's structure.
   - Compare MMSE, current learned iterative detector, analytical OAMP/UAMP, and the learned orthogonalized version under identical observations, noise variance, and test frames.

3. **Train a genuine receiver bank**
   - Train specialists with intentionally different operating distributions or objectives, e.g. low-SNR, high-Doppler/fractional-Doppler, CSI-mismatch, and unseen channel profiles.
   - Select specialists only using validation data.
   - Require a candidate to beat the incumbent on its predeclared regime and satisfy a global non-inferiority bound before allowing routing.

4. **Make adaptation meaningful**
   - Train/validate an adapter for each receiver that can be adapted, or limit the controller to the one supported adapter.
   - Compare frozen model, adapted model, and retrained-from-scratch reference on held-out frames.
   - Use multiple calibration sizes and random seeds; report paired confidence intervals and adaptation cost.

5. **Validate shift detection**
   - Evaluate physical distribution shifts and nominal drift separately.
   - Calibrate threshold on a separate calibration stream, report false alarms per frame/hour and detection delay.
   - Verify that the shift detector adds value over receiver confidence alone through an ablation.

6. **Final paper-quality reporting**
   - Regenerate every table and figure directly from machine-readable result files.
   - Include negative results and uncertainty; never edit reported values manually.
   - Archive configs, code revision, seeds, split manifests, checkpoints, and logs for every reported result.

## 4. Changes made on this branch so far

- Updated `models/supervisor/reliability_controller.py` to avoid unsupported adaptation recommendations and compare supported adaptation choices on standardized confidence scales.
- Updated `tests/test_reliability_controller.py` to assert the supported-adapter behavior and the no-supported-adapter fallback.
- Replaced the controller's test-derived BER gate with BER computed from the saved validation split (135 frames per receiver).
- Corrected the legacy controller evaluator to score the receiver that actually processed each frame, not the post-decision receiver intended for the next frame.

## 5. Corrected V1 test-stream replay

The saved 135-frame V1 test stream was replayed with the corrected validation-only BER gate. The replay verified that the saved per-frame confidence inputs remain aligned with the controller's active receiver state.

| Decision | Count |
|---|---:|
| KEEP | 132 |
| SWITCH | 1 |
| ADAPT | 2 |

The single switch selected the original GNN. Both adaptation recommendations were restricted to OAMP-DL. The receiver used for a frame is the pre-decision active receiver; a switch applies to the next frame.

| Metric | Controller pipeline | Always learned detector | Paired difference |
|---|---:|---:|---:|
| BER | 0.035376 | 0.032961 | +0.002415 |
| SER | 0.068076 | 0.063446 | +0.004630 |
| NMSE | 0.109195 | 0.101745 | +0.007449 |

Condition-stratified paired bootstrap 95% intervals for controller-minus-static differences:
- BER: [0.001409, 0.003462]
- SER: [0.002697, 0.006683]
- NMSE: [0.004696, 0.010251]

All three intervals are strictly positive, so this corrected V1 replay shows the controller pipeline is worse than always using the current best frozen detector on this test stream. The switch/adapt counts are useful behavior diagnostics, not evidence of a performance improvement. Because the raw V1 NPZ frames are not checked into the repository, this is a deterministic replay of saved confidence and per-frame receiver metrics, not a fresh re-inference of V1.

## 6. Validation status

- The CPU GitHub Actions runner executed the unit suite successfully: **23 tests passed** (including controller transition, unsupported-adapter, current-frame metric-alignment, and stratified-bootstrap regression tests).
- `run_pipeline.py` now writes `per_frame_receiver_metrics.csv` and adds paired, condition-stratified percentile-bootstrap 95% confidence intervals for adaptive-minus-always-OAMP BER/SER/NMSE to `summary.json`.
- The first attempt to rerun the original V1 test split stopped before inference because the repository checkout does not contain the raw `datasets/otfs/raw/` frame files. The historical V1 metrics and validation feature CSVs remain available, but the original raw frames are not versioned in this repository.
- To avoid fabricating or reconstructing V1, the end-to-end workflow now generates new isolated OTFS datasets with seeds 43 and 44 and evaluates the frozen receiver bank/controller on each generated test split. The generated data stays inside each run directory and is not committed to Git.
- The GitHub-hosted runner is CPU-only. The saved receiver checkpoints are reused for evaluation; no new full-scale model training is claimed by the current workflow. New training will be attempted only if a compute budget is realistic and the resulting run can be validated against held-out data.

The reproducibility command for the unit suite is:

```bash
python -m unittest discover -s tests -v
```

The end-to-end seed-43/44 workflow is the next validation stage. Keep `main` and the original `OTFS-NeuroRx` repository unchanged until all generated-data evaluations and result audits are complete.
