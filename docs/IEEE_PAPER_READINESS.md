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
