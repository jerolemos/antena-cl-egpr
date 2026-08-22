# Entropy-Gated Plasticity Regulation (EGPR)

Official open-source implementation and benchmark suite for **Entropy-Gated Plasticity Regulation (EGPR)**.

* **Part 1 of the Memory-Free Continual Learning Series:**
  * **[Part 1: EGPR Foundation](https://github.com/jerolemos/antena-cl-egpr)** (Vision benchmarks: Split-MNIST, Permuted-MNIST, Rotated-MNIST, Split-FashionMNIST, Split-CIFAR10, Split-CIFAR100)
  * **[Part 2: SLM](https://github.com/jerolemos/antena-cl-slm)** (Small Language Models & Soft-OGP)
  * **[Part 3: 1-Bit](https://github.com/jerolemos/antena-cl-1bit)** (1-Bit Discrete Subspace Routing)

---

## 📌 Abstract

Catastrophic forgetting remains a central challenge in continual learning. This work presents **Entropy-Gated Plasticity Regulation (EGPR)**, a lightweight mechanism that modulates learning rates dynamically based on the temporal derivative of predictive entropy ($dH/dt$). By calibrating the entropy distribution on an initial task and gating subsequent weight updates through a sigmoid function of the entropy z-score, EGPR selectively inhibits plasticity when the network encounters out-of-distribution data.

```
[ Predictive Entropy H ] ---> [ Z-Score Calibration ] ---> [ Sigmoid Gate G_base ]
                                                                   |
[ Temporal Derivative dH/dt ] -------------------------------------> [ Effective LR: eta_eff ]
```

---

## 🏆 Key Benchmark Results

Results averaged over $N = 5$ random seeds (mean ± std). **Bold** indicates best memory-free method per benchmark.

### 1. Retention Performance ($R$)

| Benchmark | SGD | EWC (tuned $\lambda$) | Replay (N=200) | **EGPR (Default)** | **EGPR (Tuned)** |
|---|---|---|---|---|---|
| Split-MNIST | 0.4 ± 0.1 | 10.0 ± 1.0 | 84.4 ± 1.5 | 85.7 ± 3.8 | **86.8 ± 1.5** |
| Permuted-MNIST | 86.4 ± 0.6 | 87.7 ± 0.4 | 85.0 ± 1.8 | **88.0 ± 0.1** | 87.7 ± 0.4 |
| Rotated-MNIST | 31.6 ± 1.3 | 34.2 ± 1.4 | 49.0 ± 1.2 | 58.5 ± 1.5 | **58.9 ± 1.2** |
| Split-FashionMNIST | 9.3 ± 1.4 | 11.5 ± 2.4 | 72.1 ± 3.4 | 39.1 ± 1.8 | **63.8 ± 7.9** |
| Split-CIFAR10 | 0.0 ± 0.0 | 0.0 ± 0.0 | 14.2 ± 3.1 | 19.1 ± 3.8 | **29.9 ± 2.6** |
| Split-CIFAR100 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.9 ± 0.7 | **12.1 ± 1.6** | 9.9 ± 0.8 |

*Wilcoxon signed-rank test comparing EGPR vs. EWC retention yields $p = 0.031$ in 5 of 6 benchmarks (Permuted-MNIST: $p = 0.094$).*

### 2. Synergy with Experience Replay

Combining EGPR with a small replay buffer ($N=200$) yields statistically significant retention gains over standard replay alone ($p = 0.031$ in all 6 benchmarks):

| Benchmark | Replay Alone | EGPR + Replay | Gain |
|---|---|---|---|
| Split-MNIST | 84.4% | 95.0% | +10.6 pp |
| Permuted-MNIST | 85.0% | 88.2% | +3.2 pp |
| Rotated-MNIST | 49.0% | 63.5% | **+14.5 pp** |
| Split-FashionMNIST | 72.1% | 81.6% | +9.5 pp |
| Split-CIFAR10 | 14.2% | 59.5% | **+45.3 pp** |
| Split-CIFAR100 | 1.9% | 14.9% | **+13.0 pp** |

---

## 🔬 Documented Boundary Conditions

1. **Permuted-MNIST:** Minimal advantage (+0.5 pp over EWC) because pixel permutation alters spatial structure without shifting output entropy distributions.
2. **CIFAR-10 / CIFAR-100 (Semantic Distance Deadlock):** When the semantic distance between consecutive tasks is vast, entropy spikes aggressively ($z > +8$), driving the gate to the exploration floor ($\varepsilon = 0.02$). At 2% plasticity, learning updates are too slow to alter predictions, causing $dH/dt \approx 0$ and locking the network. This delineates that entropy gating is viable when distribution shifts allow steep, continuous entropy gradients.

---

## 🚀 Quick Start & Installation

```bash
pip install -r requirements.txt
```

### Reproducing All 6 Benchmarks Across 5 Seeds
```bash
python SCRIPTS/benchmark_egpr_publication.py
```
This script executes all 6 benchmarks across 5 random seeds (42, 100, 2026, 7, 314), generating `egpr_publication_results_bwt.csv`.

---

## 📂 Repository Structure

```
.
├── CORE/
│   ├── egpr_publication_benchmark.py # Core publication benchmark suite
│   └── egpr_definitive_benchmark.py  # Definitive benchmark module
├── SCRIPTS/
│   └── benchmark_egpr_publication.py # Standalone CLI benchmark runner
├── PAPER/
│   ├── EGPR_PAPER.md                 # Full scientific preprint manuscript
│   ├── EGPR_PAPER.tex                # LaTeX source for submission
│   ├── egpr_publication_results_bwt.csv # Raw experimental data (60 runs)
│   ├── egpr_ablation.png             # Ablation study visualization
│   └── egpr_pareto_fronts.png        # Pareto front retention vs. acquisition
├── DOCS/                             # Extended technical documentation
├── LICENSE                           # Dual license notice (AGPLv3 / Commercial)
├── requirements.txt                  # Dependencies
└── README.md
```

---

## ⚖️ License & Dual-Licensing

This repository is dual-licensed:

1. **Open-Source & Academic License:** Licensed under the **GNU Affero General Public License v3.0 (GNU AGPLv3)**. Free for research, academic use, and open-source projects. Any derivative network service or software must make its source code publicly available under AGPLv3.
2. **Commercial & Enterprise License:** For commercial deployment, proprietary hardware integration, closed-source enterprise software, or silicon IP integration, a commercial license is required.

For commercial licensing inquiries, please contact:
- **Author:** Lemos J. (jerolemos@proton.me)
- **Repository:** https://github.com/jerolemos/antena-cl-egpr
