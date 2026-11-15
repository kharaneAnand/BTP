# Diverse receiver-bank addendum: linear MMSE candidate

Status: exploratory implementation; not confirmatory evidence.

## Motivation

The V2 smoke result showed that the current candidate bank is dominated by OAMP-DL-derived models. A router cannot demonstrate meaningful receiver complementarity if all candidates share essentially the same detection family. This addendum adds a parameter-free classical linear MMSE detector as a structurally distinct candidate.

## Detector definition

Given the receiver-side observation vector y, estimated channel matrix H_hat, and estimated complex noise variance sigma_hat^2, the detector computes

    x_hat = (H_hat^H H_hat + sigma_hat^2 I)^(-1) H_hat^H y.

It uses the same packed inference contract as OAMP-DL: data-domain y, pilot-derived H_hat, and training-calibrated noise estimate. It does not consume transmitted symbols, true channel, true noise, SNR labels, or velocity labels. It has no learned parameters and is not fine-tuned.

## Evaluation plan

- Include linear MMSE as a candidate in the expanded router, while retaining the static noise-estimated OAMP-DL generalist as the primary comparator.
- Report linear MMSE independently by all nine SNR/velocity cells, alongside OAMP-DL and the selected policy.
- Report paired per-frame and condition-stratified comparisons; do not assume in advance that MMSE wins any condition.
- Record how often the validation-fitted router selects MMSE and whether those choices generalize on held-out test data.
- Compare accuracy against inference latency and memory. Linear MMSE requires a regularized complex linear solve, so runtime must be measured rather than presumed lower.
- Add unit tests against an independent NumPy linear-system reference.

## Protocol status

This addition changes the candidate bank after exploratory seed 57. Therefore seed 57 remains exploratory, and the previous planned seeds 58--61 must not be described as confirmatory for this revised bank unless the revised protocol and criteria are frozen before their test results are inspected. Use fresh independent seeds for any confirmatory run after freezing this addendum. If the linear MMSE candidate never wins validation or test conditions, report that honestly and do not manipulate condition definitions or thresholds to manufacture complementarity.

## Research limitation

Linear MMSE is a classical baseline, not a novel receiver contribution by itself. A positive routed result would support the value of heterogeneous receiver selection in this simulated setting; a paper-level contribution would still require a stronger methodological novelty, independent confirmatory replications, unseen-SNR/velocity generalization, and transparent complexity/latency analysis.
