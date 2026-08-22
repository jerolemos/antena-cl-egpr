# EGPR V1 — Final Results (5 Seeds, August 2026)

## Key Claims (Statistically Validated)

### Claim 1: EGPR retains significantly better than EWC (memory-free)
Wilcoxon signed-rank test: p = 0.031 in 5 of 6 benchmarks.

| Benchmark | EGPR Ret | EWC Ret | Delta |
|---|---|---|---|
| Split-MNIST | 85.7% | 10.0% | **+75.7 pp** |
| Rotated-MNIST | 58.5% | 34.2% | **+24.3 pp** |
| Split-FashionMNIST | 39.1% | 11.5% | **+27.6 pp** |
| Split-CIFAR10 | 19.1% | 0.0% | **+19.1 pp** |
| Split-CIFAR100 | 12.1% | 0.0% | **+12.1 pp** |
| Permuted-MNIST | 88.0% | 87.7% | +0.3 pp (n.s., p=0.094) |

### Claim 2: EGPR+Replay synergy improves retention over Replay alone
p = 0.031 in ALL 6 benchmarks — the strongest result.

| Benchmark | Replay | EGPR+Replay | Delta |
|---|---|---|---|
| Split-MNIST | 84.4% | 95.0% | **+10.6 pp** |
| Permuted-MNIST | 85.0% | 88.2% | **+3.2 pp** |
| Rotated-MNIST | 49.0% | 63.5% | **+14.5 pp** |
| Split-FashionMNIST | 72.1% | 81.6% | **+9.5 pp** |
| Split-CIFAR10 | 14.2% | 59.5% | **+45.3 pp** |
| Split-CIFAR100 | 1.9% | 14.9% | **+13.0 pp** |

### Claim 3: dH/dt enables acquisition under entropy gating
Without dH/dt, acquisition drops by 24.1 pp in Rotated-MNIST and 15.9 pp in Split-FashionMNIST.

### Claim 4: DMP contributes meaningfully to retention
Without DMP, retention drops by 17.0 pp in Split-MNIST and 13.1 pp in Split-FashionMNIST.

## Boundary Conditions (Documented)
- **Permuted-MNIST**: EGPR advantage vs EWC is negligible (+0.3 pp, n.s.)
- **Split-FashionMNIST**: EGPR standalone retention is low (39.1%) but EGPR+Replay rescues (81.6%)
- **CIFAR-10**: Strong EGPR+Replay (59.5%) but standalone EGPR limited (19.1%)
- **CIFAR-100**: Data-starved scenario; all methods near chance

## Honest Assessment
- EGPR prioritizes retention over acquisition
- In balanced H-mean, Replay outperforms standalone EGPR in most benchmarks
- EGPR is most valuable as a **complementary module** (EGPR+Replay)
- Best standalone use case: memory-free devices where replay is not possible

## Files
- `EGPR_PAPER.md` — Full paper
- `../SCRIPTS/benchmark_egpr_publication.py` — Benchmark script
- `../DATA/RUN_5SEEDS_DEFINITIVE/egpr_publication_results_bwt.csv` — Raw data
- `../DATA/RUN_5SEEDS_DEFINITIVE/egpr_tuned_params.json` — Selected hyperparameters
- `../DATA/RUN_5SEEDS_DEFINITIVE/egpr_pareto_fronts.png` — Pareto front visualization
- `../DATA/RUN_5SEEDS_DEFINITIVE/egpr_ablation.png` — Ablation study visualization

## Seeds
42, 100, 2026, 7, 314

## Hardware
AMD Ryzen 7, CPU only, Python 3.10, PyTorch
