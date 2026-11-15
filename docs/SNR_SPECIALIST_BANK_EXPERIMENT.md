# SNR-specialist receiver-bank experiment

## Motivation

The existing main-branch specialist experiment adds three velocity-specialist OAMP-DL models to the pooled generalist. Its four-seed result shows a small pooled BER/SER reduction, but the strict per-seed BER success criterion is not met and pooled NMSE remains inconclusive. The earlier online self-supervised adaptation experiment also did not establish a reliable gain: the corrected four-seed run triggered no update on seeds 49--51 and one accepted update on seed 52, where BER/SER were slightly worse.

This experiment tests a distinct, falsifiable hypothesis: an SNR-specialist bank may add useful complementarity that is not captured by velocity-only specialists.

## Models and training protocol

- Preserve the generalist OAMP-DL checkpoint and all existing velocity specialists.
- Fine-tune three additional OAMP-DL specialists from the same frozen generalist initialization, using only training frames at 10, 15, or 20 dB respectively. Each specialist sees all training velocities at its SNR.
- Use the same eight-epoch fine-tuning schedule, optimizer, and training loss as the existing velocity-specialist experiment.
- Do not use validation or test targets for gradient updates.
- Evaluate all models on the same validation and test frames.

The experiment reports two policies on each fresh seed:
1. The original velocity-only bank (generalist plus the three existing velocity specialists).
2. The expanded bank (generalist, velocity specialists, and SNR specialists).

## Router information contract

Each candidate has its own ridge BER predictor, fitted only on validation receiver outputs and validation BER labels. The expanded feature vector includes pilot-derived channel matrix statistics, received-observation energy, candidate confidence and residual-consistency features, plus the logarithm of the noise variance already supplied to the receiver.

The router does not read SNR or velocity metadata columns, transmitted symbols, or test BER. The configured noise variance is an existing simulator input to OAMP-DL, not a newly inferred measurement; in this simulator it is derived from the configured SNR. This makes the current experiment an upper bound on what a future deployable noise-estimation front end could support. A real receiver must estimate noise variance from pilots/received observations and must not be given simulator metadata.

SNR and velocity labels are used only to form training subsets and stratified bootstrap groups. The test split is used once for final evaluation.

## Evaluation and acceptance

For both policies, report BER, SER and NMSE against always-generalist OAMP-DL, with paired condition-stratified bootstrap 95% confidence intervals. Report per-seed decisions and candidate metrics. The oracle (per-frame best receiver) is a test-label-only upper bound and is never available to the router.

A positive claim requires:
- lower routed BER on both independent seeds 53 and 54;
- a pooled paired, condition-stratified BER interval entirely below zero;
- no material SER regression;
- NMSE reported even if inconclusive;
- all test frames scored using the exact receiver selected before that frame's labels are opened.

If these conditions fail, retain the result as negative or inconclusive. Do not tune thresholds or model settings against these test results. Two seeds are a pilot replication, not evidence of universal generalization.

## Known limitations

- OAMP-DL is a learned iterative matched-filter residual-update detector; this experiment does not make it a mathematically faithful classical OAMP/UAMP implementation.
- The simulator supplies noise variance from SNR metadata. Deployment needs a pilot/noise estimator.
- The channel family and pilot/channel-estimation assumptions remain synthetic and fixed.
- Additional specialist checkpoints and inference choices increase memory/compute cost; report that cost before claiming a practical benefit.
