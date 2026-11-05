# Specialist-bank experiment: preregistered protocol

## Why this experiment is necessary

The previous receiver bank is dominated by the pooled OAMP-DL model. On the two fresh 135-frame test sets, the existing controller selected a non-OAMP action only rarely and did not improve on static OAMP-DL. This is not a threshold-tuning problem: a controller cannot reliably route to a specialist that does not exist.

The independent per-frame audit shows that the bank has limited but non-zero complementarity. The best receiver chosen retrospectively with test labels (an unattainable oracle, not a deployable method) has lower BER than OAMP-DL on 32/135 seed-43 frames and 38/135 seed-44 frames. The corresponding mean BER reductions are 0.001359 and 0.001570. This motivates training specialists and learning a router, but does not prove that either will improve deployment performance.

## Experiment

- Generate new, independent datasets with seeds 45 and 46. Each has 900 frames, stratified 70/15/15 by the nine SNR/velocity conditions.
- Keep the existing generalist OAMP-DL checkpoint frozen as the baseline.
- Initialize three separate OAMP-DL specialists from that checkpoint. Fine-tune each on the training split only, using the 30, 120, or 500 km/h training regimes respectively. No validation or test labels enter gradient updates.
- Evaluate the generalist and all three specialists on validation and test frames.
- Fit a separate ridge regression calibration model for each candidate using validation-only BER targets and inference-available features: pilot-based channel-matrix statistics, received-observation energy, candidate-output confidence, and candidate residual consistency.
- At test time, select the candidate with the lowest predicted BER. The router receives neither velocity labels nor transmitted symbols nor test BER.
- Compare the routed receiver with always-generalist OAMP-DL. Report the paired difference and a condition-stratified bootstrap 95% interval.
- Report the per-frame oracle only as an unattainable upper bound. It uses test labels and is never used for routing.

## Decision semantics

KEEP means the validation-calibrated router selects the original generalist. SWITCH means it selects one of the trained specialists. ADAPT is deliberately not emitted by this experiment: the previous few-shot procedure is a separate, post-stream calibration experiment and is not an online weight update. Counting a post-stream adaptation recommendation as a live adaptation decision would overstate the implemented behavior.

## Success criterion

The specialist system is considered to improve only if the routed test BER is lower than the static generalist on both independent seeds and the paired, condition-stratified 95% confidence interval for routed-minus-static BER is entirely below zero. SER and NMSE must be reported alongside BER. If this criterion is not met, the negative result is retained and the report must not claim a successful adaptive receiver.

## Reproducibility and limitations

The workflow runs on CPU, saves specialist checkpoints, validation/test per-receiver metrics, decision logs, and summaries as GitHub Actions artifacts. This is a simulated OTFS study with the existing V1 channel generator and fixed pilot/channel-estimation model; results should not be generalized to measured radio hardware without a separate experimental campaign.