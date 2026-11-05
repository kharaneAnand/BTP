# Specialist-bank experiment: preregistered protocol

## Why this experiment is necessary

The previous receiver bank is dominated by the pooled OAMP-DL model. On the two fresh 135-frame test sets, the existing controller selected a non-OAMP action only rarely and did not improve on static OAMP-DL. This is not a threshold-tuning problem: a controller cannot reliably route to a specialist that does not exist.

The independent per-frame audit shows that the bank has limited but non-zero complementarity. The best receiver chosen retrospectively with test labels (an unattainable oracle, not a deployable method) has lower BER than OAMP-DL on 32/135 seed-43 frames and 38/135 seed-44 frames. The corresponding mean BER reductions are 0.001359 and 0.001570. This motivates training specialists and learning a router, but does not prove that either will improve deployment performance.

## Experiment

- Generate new, independent datasets with seeds 45 and 46. Each has 900 frames, stratified 70/15/15 by the nine SNR/velocity conditions.
- Keep the existing generalist OAMP-DL checkpoint frozen as the baseline.
- Initialize three separate OAMP-DL specialists from that checkpoint. Fine-tune each on the training split only, using the 30, 120, or 500 km/h training regimes respectively. No validation or test labels enter gradient updates.
- Evaluate the generalist and all three specialists on validation and test frames.
- Fit a separate ridge regression calibration model for each candidate using validation-only BER targets and inference-available features: pilot-based channel-matrix statistics, received-observation energy, candidate-output confidence, and candidate residual consistency. The router feature vector excludes simulator SNR and velocity labels.
- At test time, select the candidate with the lowest predicted BER. The router receives neither velocity labels nor SNR labels nor transmitted symbols nor test BER. The receiver network itself still consumes a noise-variance input; the current simulator derives that input from configured SNR, so deployment requires a pilot/noise estimator and this assumption is stated as a limitation.
- Compare the routed receiver with always-generalist OAMP-DL. Report the paired difference and a condition-stratified bootstrap 95% interval.
- Report the per-frame oracle only as an unattainable upper bound. It uses test labels and is never used for routing.

## Decision semantics

KEEP means the validation-calibrated router selects the original generalist. SWITCH means it selects one of the trained specialists. ADAPT is deliberately not emitted by this experiment: the previous few-shot procedure is a separate, post-stream calibration experiment and is not an online weight update. Counting a post-stream adaptation recommendation as a live adaptation decision would overstate the implemented behavior.

## Success criterion

The specialist system is considered to improve only if the routed test BER is lower than the static generalist on both independent seeds and the paired, condition-stratified 95% confidence interval for routed-minus-static BER is entirely below zero. SER and NMSE must be reported alongside BER. If this criterion is not met, the negative result is retained and the report must not claim a successful adaptive receiver.

## Reproducibility and limitations

The workflow runs on CPU, saves specialist checkpoints, validation/test per-receiver metrics, decision logs, and summaries as GitHub Actions artifacts. This is a simulated OTFS study with the existing V1 channel generator and fixed pilot/channel-estimation model. The OAMP-DL detector still receives the configured noise variance from the simulator, rather than an independently estimated noise variance; results should not be generalized to measured radio hardware without a separate experimental campaign.

## Follow-up confirmatory replications

The first clean two-seed evaluation produced a pooled, condition-stratified paired BER difference of -0.00013084 (95% CI [-0.00023148, -0.00004026]) and SER difference of -0.00025161 (95% CI [-0.00044283, -0.00008052]). The pooled NMSE difference was -0.00014338 (95% CI [-0.00033376, 0.00003163]). Seed 45's individual BER interval crossed zero, while seed 46's did not. Therefore, the original strict per-seed success criterion is not fully satisfied, even though the pooled BER/SER comparison is favorable.

To assess reproducibility without tuning on the test sets, the exact same code, checkpoint initialization, learning rate, specialist training epochs, router features, and ridge regularization are frozen and run on two additional independent seeds, 47 and 48. No hyperparameter changes are permitted after inspecting seeds 45/46. These are confirmatory replications, not a new tuned model.


## Four-seed frozen-method evaluation (final report result)

The same code and hyperparameters were run on independent seeds 47 and 48 without modification. Across seeds 45--48 (540 test frames), the pooled paired, condition-stratified bootstrap results are:

- BER: static 0.03697665, routed 0.03689110; routed-minus-static -0.00008555, 95% CI [-0.00014845, -0.00002516].
- SER: static 0.07033514, routed 0.07017411; difference -0.00016103, 95% CI [-0.00028180, -0.00004529].
- NMSE: static 0.11257751, routed 0.11250080; difference -0.00007672, 95% CI [-0.00018562, 0.00002388].
- Decisions: KEEP 126, SWITCH 414, ADAPT 0. All three specialists were selected across the four seeds.
- The pooled router closes approximately 23.8% of the per-frame oracle BER gap. The oracle remains an unattainable test-label upper bound.

Seed-level results are heterogeneous. Seeds 46 and 47 have individually significant BER reductions; seed 45 is inconclusive, and seed 48 has a small positive routed-minus-static BER mean with a confidence interval crossing zero. Thus the pooled BER/SER gains are statistically supported, but the strict original per-seed success criterion is not met and improvement is not guaranteed on every seed. NMSE remains inconclusive in the pooled analysis. This mixed result is reported explicitly rather than claiming a universal win.
