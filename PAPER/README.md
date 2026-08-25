# Entropy-Gated Plasticity Regulation for Continual Learning Without Episodic Memory

**Author:** Lemos J.

**Part of a series on memory-free continual learning:**
* [Part 1: EGPR Foundation](../01_EGPR_FOUNDATION)
* [Part 2: SLM & Soft-OGP](../antena-cl-slm)
* [Part 3: 1-Bit QUBO Routing](../03_1BIT_QUBO)


## Abstract
Catastrophic forgetting remains a central challenge in continual learning. Existing approaches based on regularization or episodic memory offer effective solutions but introduce specific computational and memory tradeoffs. This work presents Entropy-Gated Plasticity Regulation (EGPR), a lightweight mechanism that modulates learning rates based on the temporal derivative of predictive entropy. By calibrating the entropy distribution on an initial task and gating subsequent weight updates through a sigmoid function of the entropy z-score, EGPR selectively inhibits plasticity when the network encounters out-of-distribution data.

## Files Included
*   `EGPR_PAPER.md`: The main preprint manuscript containing the theoretical framework, mathematical formulation, and ablation studies.
*   `src/`: Directory containing the core Python implementation (`antena_plasticity_benchmark.py` and `egpr_publication_benchmark.py`).
*   `egpr_publication_results_bwt.csv`: Raw data (60 rows) across 6 benchmarks and 10 strategies (EWC, Replay, EGPR variants) used for statistical validation.
*   `RESULTS_SUMMARY.md`: Executive summary of the empirical findings.
*   `egpr_ablation.png`: Visualization of the ablation study.
*   `egpr_pareto_fronts.png`: Pareto front analysis comparing Retention vs. Acquisition tradeoffs.
*   `LICENSE`: Dual-license file (AGPLv3 / Commercial).

## License & Usage
© 2026 Jero Lemos. All rights reserved.

This repository is dual-licensed:
1. **Open Source:** Available under the GNU Affero General Public License v3.0 (AGPLv3) strictly for open-source and non-commercial use.
2. **Commercial:** A proprietary commercial license is available for enterprise integration without source-code disclosure requirements. Contact Jero Lemos (jerolemos@proton.me) for details.

## Citation
If you use this work in your research, please cite it as:
```bibtex
@misc{lemos2026egpr,
  title={Entropy-Gated Plasticity Regulation for Continual Learning Without Episodic Memory},
  author={Lemos, Jero},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22098875},
  url={https://doi.org/10.5281/zenodo.22098875}
}
```
