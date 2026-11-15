# IEEE paper readiness and claim boundary

Updated: 6 November 2026

## Main-branch contribution currently supported by results

The merged main branch contains the three-specialist OAMP-DL bank and validation-calibrated ridge router. Specialists are fine-tuned from the generalist checkpoint using training data from 30, 120, and 500 km/h regimes. The router is fitted on validation data using observable channel/receiver features; it does not receive SNR/velocity labels or test BER.

Four independent seeds (45--48; 540 held-out frames) support a modest pooled BER/SER reduction:
- BER: 0.03697665 static vs 0.03689110 routed; difference -0.00008555, 95% CI [-0.00014845, -0.00002516].
- SER: 0.07033514 static vs 0.07017411 routed; difference -0.00016103, 95% CI [-0.00028180, -0.00004529].
- NMSE: 0.11257751 static vs 0.11250080 routed; difference -0.00007672, 95% CI [-0.00018562, +0.00002388], inconclusive.
- Decision counts: KEEP 126, SWITCH 414, ADAPT 0.

Seed 46 and 47 show individually significant BER reductions; seed 45 is inconclusive; seed 48 has a small non-significant regression. The strict per-seed success criterion was not met. This is a modest average gain under the tested simulator, not universal improvement.

## Separate online-adaptation experiment (not merged)

PR #3 contains an experimental causal decision-directed self-supervised adaptation method. It uses past received-observation windows, a validation-calibrated innovation trigger, a QPSK prior and an anchor penalty, with rollback if the self-supervised objective or parameter-drift guard fails. Transmitted test symbols are not loaded until after all online inference/adaptation has finished.

This method has not yet produced verified four-seed results. It must not be included in the paper's empirical claims until the workflow succeeds, the code is independently checked, and BER/SER/NMSE confidence intervals are reviewed. A nonzero ADAPT count alone is not success.

## Architecture and limitations that must be disclosed

1. Three fine-tuned specialist OAMP-DL detectors were added; no new detector architecture was introduced in the specialist-router experiment.
2. The router is a per-candidate validation-calibrated ridge regression BER predictor.
3. The OAMP-DL detector is an unfolded learned iterative matched-filter residual-update network; do not call it a mathematically faithful classical OAMP/UAMP implementation without additional derivation and validation.
4. Noise variance is still supplied from simulator SNR metadata. A real receiver must estimate it independently, preferably from pilots/received observations.
5. Experiments use a synthetic OTFS channel generator and fixed pilot/channel-estimation assumptions; there is no measured-hardware validation.
6. The per-frame oracle uses test labels and is only an unattainable upper bound.
7. Report seed-level heterogeneity and the inconclusive NMSE result.

## Recommended IEEE submission gate

Before submission: (a) complete online-adaptation runs on fresh seeds; (b) include static, specialist-router, and online-adapted baselines on identical paired test frames; (c) include ablations for no adaptation, no rollback, no constellation prior, and threshold choice; (d) report runtime/complexity and update overhead; (e) use a pilot-based noise estimator; (f) add a broader literature review and avoid an unverified novelty claim; (g) have a co-author independently reproduce the final table and figures.


## Related-work and novelty boundary (literature check, 6 November 2026)

The broad idea of adapting learned OTFS receivers to updated channel profiles is not new. Park et al., *Auto-Encoder Based OTFS Modulation and Detection With Meta-Learning*, IEEE Access, 2023, DOI: 10.1109/ACCESS.2023.3271993, explicitly develop meta-learning for an OTFS autoencoder transceiver under newly updated channel profiles.

Model-driven unfolded OTFS detectors are also established. Examples include *Gaussian AMP Aided Model-Driven Learning for OTFS System*, IEEE Communications Letters, 2022, DOI: 10.1109/LCOMM.2022.3206414, and *Model-Driven Deep Learning Assisted Detector for OTFS With Channel Estimation Error*, IEEE Communications Letters, 2024, DOI: 10.1109/LCOMM.2024.3362970.

Self-supervised online adaptation of neural channel estimators has also been studied in OFDM (e.g., ChannelMAE: Self-Supervised Learning Assisted Online Adaptation of Neural Channel Estimators). This is adjacent rather than identical to OTFS detection, but it rules out claiming self-supervised online adaptation as a new concept by itself.

Therefore, do not claim that OTFS adaptation, model-driven OAMP learning, meta-learning, or self-supervised online adaptation is individually novel. A defensible potential contribution—subject to a complete literature review and validated experiments—would be the specific combination of (i) a reliability-calibrated bank of mobility-specialized unfolded receivers, (ii) causal decision-directed self-supervised updates with an independent observation guard and rollback, and (iii) paired evaluation under controlled OTFS distribution shifts. The paper must demonstrate that this combination improves over the strongest static specialist/router baselines at acceptable runtime and update cost.


## Updated evidence and claim boundary (15 November 2026)

### SNR-specialist bank pilot (merged PR #5)

The expanded seven-receiver bank (generalist + three velocity specialists + three SNR specialists) was evaluated on two independent 135-frame test sets (seeds 53 and 54). Recomputed from the uploaded per-frame test CSVs:

- Static noise-oracle OAMP-DL: BER 0.03008253, SER 0.05771940, NMSE 0.09312466.
- Expanded velocity+SNR router: BER 0.02997182, SER 0.05752818, NMSE 0.09316725.
- Paired condition-stratified BER difference: -0.00011071, 95% CI [-0.00021135, -0.00001510].
- SER difference: -0.00019122, 95% CI [-0.00038245, approximately 0].
- NMSE difference: +0.00004258, 95% CI [-0.00010564, +0.00021303].

Seed 53 has a clear BER reduction; seed 54 is nearly unchanged and its BER interval crosses zero. The pooled BER result is statistically supported, but this is only a two-seed pilot and the router includes log noise variance derived from simulator SNR. It is not yet a deployable receiver result.

### Pilot noise-estimator diagnostics

The original fixed pilot-delay-column estimator is rejected: the channel generator moves the minimum-delay path to zero, so that column can contain a real path response. The corrected candidate selects the minimum-energy column among the five pilot CE delay columns. A multiplicative correction is fitted using the training split only.

Exploratory V1 seeds 55/56 show calibrated test estimated/true noise ratios of 1.0345 (95% CI [0.9919, 1.0782]) and 1.0138 (95% CI [0.9739, 1.0542]). Per-frame standard deviation remains about 25--27%, so the estimate is a noisy input rather than a precise per-frame measurement. These seeds informed the estimator correction and are not confirmatory receiver evidence.

### V2 noise-aware receiver experiment (not yet paper evidence)

V2 uses fixed pilot power, estimates noise from received pilot-CE observations, and reconstructs `h_hat` using that estimated noise. It trains oracle-noise and noise-estimated generalists from the same OAMP-DL initialization, then adds velocity and SNR specialists and a shared-feature single-pass router. The router does not use SNR/velocity labels or test BER.

Seed 57 is an exploratory smoke test. Fresh seeds 58--61 are reserved for frozen confirmatory replications. No V2 performance improvement may be claimed until those jobs complete and the paired results meet the preregistered BER/SER/NMSE criteria. A separate unseen-SNR/velocity evaluation is also required before a broad generalization claim.

### Current paper claim

The strongest completed evidence is a small pooled BER reduction from the V1 SNR-specialist pilot, with seed-level heterogeneity and a simulator-noise dependency. It is promising but not sufficient for a strong paper claim. The V2 noise-aware receiver experiment is the current path to a more defensible result. Do not claim universal superiority, real-radio deployment, or novelty of OTFS adaptation/unfolding itself. The current OAMP-DL code is a learned iterative matched-filter residual-update network; it is not a mathematically faithful classical OAMP implementation.
