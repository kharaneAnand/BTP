# Causal self-supervised online adaptation experiment

## Motivation

The previous few-shot script optimized OAMP-DL against `tx_dd` targets from a labelled validation subset. That is an offline supervised calibration experiment; a deployed receiver does not know transmitted data symbols. This experiment explicitly removes that information path.

## Protocol

- Generate fresh independent OTFS datasets for seeds 49--52 (900 frames per seed).
- Keep the generalist OAMP-DL checkpoint frozen as the static baseline.
- Randomize the held-out test stream order deterministically per seed to mix the nine SNR/velocity conditions.
- Process frames in batches of eight. The first batch is warm-up and is detected with the unchanged model.
- Before processing each subsequent batch, compute normalized innovation on the *previous* batch. Calibrate the trigger from the 90th percentile of *eight-frame validation-window mean* normalized innovation scores; compare each previous test-window mean with this frozen threshold. Matching the window aggregation avoids comparing a test-batch mean with a per-frame quantile.
- Split each previous eight-frame batch into six adaptation frames and two guard frames. Update only the existing OAMP-DL scalar controls (step, damping, variance logits) for five AdamW steps using the six adaptation frames; accept only if the self-supervised objective also improves on the two guard frames.
- The self-supervised objective combines normalized observation consistency, a QPSK constellation prior, and an anchor penalty to the pre-update parameters. The two guard frames are never used for gradient updates and provide an out-of-update-batch acceptance check.
- Accept an update only if the objective decreases and relative parameter drift stays below 20%; otherwise roll back.
- The observation loader opens only `rx_dd` and `h_hat` while adapting and inferring; it does not load `tx_dd`. Transmitted symbols are opened only after every test-stream prediction and update has completed, for offline BER/SER/NMSE scoring.
- The current batch is never used to update the model that detects that same batch. Updates use only past windows, making the experiment causal at the batch level.

## Required comparisons and reporting

Report static versus online-adapted OAMP-DL BER, SER and NMSE, paired condition-stratified 95% bootstrap intervals, number of trigger events, accepted updates, rejected updates, and the fraction of frames processed after an accepted update. Compare against the previously reported specialist-router result, but do not combine their test sets or tune on their outcomes.

## Interpretation rules

- A nonzero ADAPT count is not itself success. The adaptation must improve held-out BER/SER with confidence intervals, without materially degrading NMSE.
- If no updates trigger, report that the innovation threshold was not crossed; do not manufacture ADAPT events.
- If updates trigger but are rejected, report the rollback rate.
- If accepted updates fail to improve held-out metrics, retain the negative result and investigate a better unsupervised objective or pilot-supervised adaptation.
- This is decision-directed self-supervision, not guaranteed error-free adaptation: wrong soft decisions can cause error propagation. The anchor and rollback are safeguards, not proof of stability.
- The detector still receives simulator-derived noise variance in this codebase. Real deployment requires an independently estimated noise variance.


## Implementation audit notes

- Trigger threshold is calibrated once from validation-only eight-frame window-mean normalized-innovation scores (90th percentile); it is frozen before test-stream evaluation.
- The held-out stream order is randomized deterministically using the experiment seed.
- The receiver-input loader reads only received DD observations and pilot-derived channel estimates. It does not access transmitted symbols during adaptation or inference.
- The current implementation still uses simulator-configured SNR to construct the noise-power input. This is a known simulator-information dependency, not a deployable noise estimator.
- The workflow is an experiment runner, not evidence of a successful result. No result should be added to the paper until the four fresh seeds complete and the paired test comparison is reviewed.
