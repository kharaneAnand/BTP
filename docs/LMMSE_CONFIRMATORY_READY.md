# Frozen heterogeneous receiver-bank confirmatory run

This marker triggers fresh independent seeds 62--65 for the revised receiver bank containing the noise-estimated OAMP-DL generalist, OAMP-DL specialists, and the parameter-free linear MMSE candidate.

Protocol freeze:
- Candidate set and router implementation are frozen at this branch revision.
- Seeds 62--65 are the planned fresh confirmatory replications for this revised bank.
- Seed 57 and earlier V2 runs are exploratory only.
- No test labels, true channel, true noise, SNR labels, or velocity labels are permitted as router inputs.
- Report per-condition candidate performance, selected receiver counts, paired condition-stratified uncertainty, noise/channel estimation audits, and latency.
- Passing this primary workflow does not establish unseen-SNR/velocity generalization or measured-radio validity. Those remain separate requirements.

The acceptance criteria in docs/NOISE_AWARE_SPECIALIST_EXPERIMENT.md remain the primary BER/SER/NMSE and integrity criteria. The new linear MMSE candidate is evaluated as an additional heterogeneous option; it is not assumed to win any condition.
