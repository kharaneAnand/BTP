# Noise-aware OTFS receiver bank: V2 preregistered experiment

## Research question

Does the velocity-plus-SNR specialist bank still improve over a static OAMP-DL receiver when both the noise variance and the pilot-derived channel matrix are reconstructed from received observations, rather than supplied by simulator metadata?

This experiment addresses two oracle dependencies found in the V1 pipeline:
1. OAMP-DL receives the exact simulator noise variance derived from SNR.
2. The stored V1 `h_hat` was generated using a channel-estimation threshold proportional to the exact simulator noise variance.

The V1 SNR-specialist result remains a positive two-seed pilot, but it does not resolve these deployment dependencies.

## V2 waveform

- Preserve the 16 x 32 DD grid, QPSK data, four-path channel family, delay support and fractional-Doppler generation.
- Keep the same nine nominal SNR/velocity cells for the primary matched comparison.
- Use a fixed pilot power of 10 linear units (10 dB above unit-average-power data symbols), independent of the data noise variance.
- Preserve the embedded pilot and configured guard geometry.
- Do not overwrite V1 data or historical result artifacts. V2 samples are generated in a separate directory.

The fixed pilot power makes the transmitted pilot amplitude known from public configuration and removes the V1 pilot-power dependence on simulator noise.

## Target-free noise and channel front end

The receiver-side noise estimator:
1. Reads only `rx_dd` and the fixed pilot/guard configuration.
2. Computes received energy in each of the five pilot channel-estimation delay columns (pilot delay through pilot delay + max_delay).
3. Selects the minimum-energy CE delay column. With four paths and five possible delay bins, at least one CE delay column has no path response under this simulator's delay-support construction.
4. Uses the sample mean of the 16 complex Doppler-bin magnitudes squared as the raw complex noise-power estimate.
5. Applies a single multiplicative calibration fitted on the training split only. Validation and test noise labels are not used to fit this calibration.

The estimated noise power is then used to reconstruct the pilot-derived channel matrix from `rx_dd`, using the fixed known pilot amplitude and a threshold of three times the estimated noise standard deviation. The V2 deployable receiver input is therefore `rx_dd`, estimated `h_hat`, and estimated noise power. It does not read `tx_dd`, `h_dd`, simulator SNR, simulator noise power or true channel paths.

The simulator noise and true channel are retained only for offline audit after inference. The estimator's minimum-column selection can have downward selection bias when multiple CE columns are noise-only; the training-only calibration corrects average bias but does not eliminate per-frame variance. Report this uncertainty.

## Receiver bank and fair baselines

- Oracle-noise generalist: initialized from the existing OAMP-DL checkpoint and fine-tuned on V2 training data with the oracle V2 channel estimate and simulator noise. This is explicitly a non-deployable upper-bound reference.
- Noise-estimated generalist: initialized from the same checkpoint and fine-tuned with the deployable noise estimate and reconstructed channel matrix.
- Three velocity specialists: fine-tuned from the noise-estimated generalist on training data at 30, 120 and 500 km/h.
- Three SNR specialists: fine-tuned from the same noise-estimated generalist on training data at 10, 15 and 20 dB.

All generalist fine-tuning uses the same optimizer, learning-rate schedule, epoch cap and validation early stopping. All specialists use the same fixed fine-tuning schedule. No validation or test samples enter gradient updates.

## Router

The primary router is single-pass:
- Its features are computed from the received observation, estimated channel matrix, and estimated noise power.
- Features include channel row/column energy statistics, energy concentration, received-observation energy, matched-filter residual consistency and estimated noise power.
- It does not require running all seven candidate receivers before making a decision.
- A separate ridge BER predictor is fitted for each candidate using validation-only receiver BER labels and shared, candidate-independent features.
- The router selects the receiver with the lowest predicted BER.
- It receives no SNR/velocity labels, transmitted symbols, true channel, or test BER.

For offline scientific analysis, all candidates may be evaluated on the held-out test frames after the routing policy is frozen. Those counterfactual outputs are not available to the deployed router.

## Fresh-seed protocol

- Initial end-to-end smoke/pilot: seed 57. Use only to catch implementation problems and assess whether the method is worth freezing.
- Confirmatory replications: seeds 58--61, with identical code, hyperparameters, and router settings frozen before any confirmatory test results are inspected.
- Each seed has 900 frames and a stratified 630/135/135 train/validation/test split.
- If seed 57 causes a code or method change, it is exploratory and is not pooled into confirmatory evidence.

## Required reporting

For each seed and pooled confirmatory set, report:
- BER, SER and symbol NMSE for oracle-noise generalist, noise-estimated generalist, velocity-only router, and expanded velocity-plus-SNR router.
- Paired, condition-stratified 95% bootstrap intervals for all policy differences.
- Channel-estimation NMSE for the estimated and oracle pilot front ends.
- Noise-estimator ratio, bias, dispersion and per-condition error.
- Receiver-selection counts, router calibration features and train/validation/test sample counts.
- Measured selected-only inference latency, feature-extraction overhead, and checkpoint/memory cost.
- Per-seed heterogeneity and failures; do not hide a weak seed behind a pooled average.

## Acceptance criterion

The V2 expanded bank is considered a paper-worthy accuracy improvement only if:
1. Pooled paired BER 95% CI versus the noise-estimated static generalist is entirely below zero across seeds 58--61.
2. At least three of the four confirmatory seeds have lower BER; any regression seed is reported and explained.
3. The upper endpoint of the pooled paired SER-difference 95% interval is below a predeclared 1% relative non-inferiority margin (0.01 times the static baseline SER).
4. The upper endpoint of the pooled paired NMSE-difference 95% interval is below a predeclared 1% relative non-inferiority margin (0.01 times the static baseline NMSE).
5. The expanded bank's result is not achieved by simulator SNR/noise labels, test-label routing, true channel inputs, or oracle channel-estimation thresholds.
6. The single-pass router's measured latency and model footprint are reported.
7. The same method is evaluated on a separate unseen-SNR/velocity test after the primary confirmatory results are frozen.

The oracle-noise baseline is a reference, not the deployment comparator. If the expanded bank fails these criteria, report the negative/inconclusive result and continue with a justified next experiment rather than tuning on the same test sets.


## Exploratory noise-estimator diagnostics (seeds 55/56)

These runs are development diagnostics only; they are not confirmatory receiver results.

- The first fixed pilot-delay-column estimator failed: it overestimated simulator noise by approximately 6.2x on seed 56.
- Source inspection found the cause: the channel generator explicitly moves the minimum-delay path to zero, so the nominal pilot-delay column contains a real path response and is not noise-only.
- The corrected raw estimator selects the minimum-energy column within the five pilot CE delay columns. The channel has four paths and five possible integer delay bins, so at least one CE column is unoccupied under the current channel model.
- On the two exploratory test sets, the raw estimated/true noise-power ratio was 0.9487 (seed 55) and 0.9342 (seed 56). The condition-stratified 95% intervals were [0.9096, 0.9889] and [0.8974, 0.9714]. The estimator therefore has a consistent downward bias of roughly 5--7%.
- The raw estimate was within 30% of true noise on 71.9% and 79.3% of frames, respectively. Per-frame dispersion is expected because only 16 complex Doppler samples are averaged.

A multiplicative calibration is now fitted from the training split only and evaluated on held-out validation/test frames. Seeds 55/56 remain exploratory because they informed the estimator correction; the V2 receiver smoke seed 57 is also exploratory. Confirmatory receiver evidence must come from frozen fresh seeds 58--61.
