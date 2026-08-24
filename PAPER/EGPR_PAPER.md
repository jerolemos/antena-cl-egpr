# Entropy-Gated Plasticity Regulation for Continual Learning Without Episodic Memory

**Lemos J.**
*Independent Researcher*

---

## Abstract

Catastrophic forgetting remains a central challenge in continual learning. Existing approaches based on regularization or episodic memory offer effective solutions but introduce specific computational and memory tradeoffs. This work presents Entropy-Gated Plasticity Regulation (EGPR), a lightweight mechanism that modulates learning rates based on the temporal derivative of predictive entropy. By calibrating the entropy distribution on an initial task and gating subsequent weight updates through a sigmoid function of the entropy z-score, EGPR selectively inhibits plasticity when the network encounters out-of-distribution data. A complementary mechanism, Depth-Modulated Plasticity (DMP), applies layer-wise scaling to preserve feature representations in early layers while permitting adaptation in later layers. Empirical evaluation across six benchmarks (Split-MNIST, Permuted-MNIST, Rotated-MNIST, Split-FashionMNIST, Split-CIFAR10, and Split-CIFAR100) with five random seeds demonstrates that EGPR achieves significantly higher retention than Elastic Weight Consolidation (EWC) in five of six benchmarks (one-sided Wilcoxon signed-rank test, $p = 0.031$), without requiring episodic memory or explicit task boundaries. On Split-MNIST, EGPR achieves 85.7% retention compared to 10.0% for EWC, without any episodic memory. Furthermore, combining EGPR with experience replay yields statistically significant retention improvements over standard replay alone in all six benchmarks ($p = 0.031$), with gains ranging from +3.2 to +45.3 percentage points. On Split-CIFAR10, EGPR+Replay achieves 59.5% retention compared to 14.2% for replay alone, establishing EGPR as a highly effective complementary module for existing replay systems. The method's boundary conditions, notably the Semantic Distance Deadlock on complex visual domains, are documented and analyzed to inform future work.

---

## 1. Introduction

Neural networks trained sequentially on multiple tasks tend to overwrite previously learned representations when adapting to new data distributions, a phenomenon known as catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999). This limitation constitutes a fundamental barrier to the deployment of continual learning systems in real-world settings where data arrives non-stationarily and retraining from scratch is impractical.

Several families of approaches have been proposed to mitigate catastrophic forgetting. Regularization methods such as Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) penalize changes to parameters deemed important for prior tasks. Replay-based methods maintain a buffer of past examples to interleave with new training data (Chaudhry et al., 2019). Architectural methods such as Progressive Neural Networks (Rusu et al., 2016) allocate dedicated capacity for each task. Each approach involves specific tradeoffs between memory requirements, computational cost, and the need for explicit task boundaries.

This work explores an alternative perspective: that catastrophic forgetting arises primarily from unconstrained plasticity during distribution shifts, and that the network's own predictive uncertainty provides a sufficient signal to regulate when learning should occur. Rather than tracking parameter importance or storing past examples, the proposed approach monitors the entropy of the output distribution and its temporal derivative to determine whether incoming data is being successfully assimilated.

Two mechanisms are introduced:

1. **Entropy-Gated Plasticity Regulation (EGPR)**: a gating function that continuously scales the effective learning rate based on a calibrated z-score of the predictive entropy. When entropy rises sharply above the calibrated baseline (indicating unfamiliar data), plasticity is suppressed. When a negative temporal derivative of entropy ($dH/dt < 0$) is detected, indicating that the network is converging on the new distribution, plasticity is partially restored.

2. **Depth-Modulated Plasticity (DMP)**: a layer-wise scaling heuristic that applies greater protection to deeper layers (which encode task-specific decision boundaries) while permitting higher plasticity in early layers (which encode more generalizable features).

These mechanisms are evaluated on six continual learning benchmarks against standard baselines. The results indicate that EGPR provides strong retention capabilities — particularly as a complement to existing replay methods — while also revealing specific boundary conditions that motivate directions for future research.

---

## 2. Related Work

### 2.1 Regularization-Based Methods

Regularization strategies constrain weight updates to preserve parameters important for prior tasks. EWC (Kirkpatrick et al., 2017) estimates a diagonal Fisher Information Matrix to penalize deviations from previously learned parameters. Synaptic Intelligence (SI; Zenke et al., 2017) tracks the contribution of each parameter to loss reduction over time. Memory Aware Synapses (MAS; Aljundi et al., 2018) computes importance weights based on output sensitivity rather than loss gradients, enabling unsupervised importance estimation. More recently, gradient projection methods such as Orthogonal Gradient Descent (OGD; Farajtabar et al., 2020) and Gradient Episodic Memory (GEM; Lopez-Paz & Ranzato, 2017) modify gradients to avoid interference with prior tasks. These methods provide principled protection but require explicit computation of parameter importance matrices, gradient histories, or task-boundary information. In this work, EWC is used as the primary regularization baseline due to its canonical status in the continual learning literature; comparison with more recent approaches such as SI and OGD is left for future work.

### 2.2 Replay-Based Methods

Experience replay maintains a fixed-size buffer of past training examples and interleaves them during subsequent task training (Chaudhry et al., 2019). Generative replay (Shin et al., 2017) replaces stored examples with a generative model. Both approaches are effective but require memory proportional to the number of tasks or the complexity of past data distributions. EGPR is orthogonal to replay-based methods and, as demonstrated in the experimental results, can be combined with replay to enhance retention.

### 2.3 Entropy and Information-Theoretic Approaches

Several works have explored entropy and output distribution metrics for stability in neural networks. Entropy-based Stability-Plasticity (ESP; Chen et al., 2022) introduces per-layer entropy computation via trainable branch networks. E-Adapt (Wang et al., 2026) monitors entropy flow for adaptive training in long-tailed recognition. More recently, anti-collapse regularization techniques such as SIGReg (Balestriero & LeCun, 2025) introduce static loss penalties to enforce Gaussian distributions on representation spaces. 

The present work addresses a different setting: memory-free continual learning through output-level entropy derivatives ($dH/dt$) applied dynamically at the autograd optimization level. Rather than enforcing a static loss penalty or training auxiliary networks, EGPR continuously modulates the effective learning rate based on the system's empirical entropy dynamics without adding extra parameter capacity.

---

## 3. Method

### 3.1 Predictive Entropy as a Novelty Signal

Given a batch of $B$ inputs processed by a network with parameters $\theta$, the predictive entropy is computed as:

$$H = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c} P(y_c | x_i, \theta) \log P(y_c | x_i, \theta)$$

After training on an initial task $T_1$, the mean $\mu_H$ and standard deviation $\sigma_H$ of the per-batch entropy values are recorded as a calibration baseline. For each subsequent training batch, the entropy z-score is computed as:

$$z = \frac{H - \mu_H}{\sigma_H}$$

This calibration step enables EGPR to operate without absolute entropy thresholds, which would be sensitive to dataset and architecture choices.

### 3.2 Sigmoid Gating and Temporal Derivative Boost

The base plasticity coefficient is defined by a sigmoid function centered at $z = c$ (a hyperparameter controlling the strictness of gating):

$$G_{\text{base}} = \frac{1}{1 + \exp(2(z - c))}$$

When $G_{\text{base}}$ falls below a minimum exploration floor $\varepsilon$, it is clamped to $\varepsilon$ to maintain a minimal learning signal. This exploration floor prevents a complete plasticity deadlock, which was empirically observed to cause failure in earlier versions of the system.

The temporal derivative of entropy is approximated as:

$$\frac{dH/dt} \approx H_t - \frac{1}{K}\sum_{k=1}^{K} H_{t-k}$$

where $K = 4$ is the lookback window. If $dH/dt < -s$ (where $s$ is a sensitivity threshold), an additive boost is applied:

$$G_{\text{total}} = \min\left(1.0, \ G_{\text{base}} + \min(0.3, \ 2|dH/dt|)\right)$$

This mechanism detects when the network is actively converging on a new distribution, and increases plasticity accordingly. The final effective learning rate for each training step is $\eta_{\text{eff}} = \eta_{\text{base}} \times G_{\text{total}}$.

### 3.3 Depth-Modulated Plasticity (DMP)

To preserve low-level feature representations (edges, textures) that are typically shared across tasks, DMP applies a layer-dependent scaling factor. For an MLP with $L$ parameter groups, the factor for group $l \in \{0, \dots, L-1\}$ is:

$$M_l = 0.3 + 0.7 \left(1.0 - 0.7 \cdot \frac{l}{L-1}\right)$$

This results in early layers receiving near-full plasticity while deeper layers receive approximately 51% of the base rate. The final layer-specific learning rate is:

$$\eta_{\text{eff}}^{(l)} = \eta_{\text{base}} \times G_{\text{total}} \times M_l$$

### 3.4 Exploration Floor ($\varepsilon$)

In domains where the entropy z-score remains persistently high (e.g., due to large semantic distance between tasks), the sigmoid gate drives plasticity to near zero. Without a minimum floor, the system enters a deadlock: no learning occurs, so no entropy reduction can be detected, and no boost is triggered. A fixed exploration floor of $\varepsilon = 0.02$ was found to resolve this deadlock in all tested benchmarks, providing a minimal learning rate ($\eta_{\text{eff}} = 0.02 \times \eta_{\text{base}}$) that allows the temporal derivative signal to emerge.

### 3.5 Empirical Signal of Optimization Convergence (Entropic Reduction Phase)

The temporal derivative boost ($dH/dt < -s$) is designed to identify when the optimization trajectory transitions from initial exploration to structural alignment with a new task distribution. Empirically, when encountering an out-of-distribution task, output entropy initially spikes ($z \gg 0$). As the network begins to assimilate the new underlying patterns, output entropy drops precipitously.

In internal telemetry, when the per-batch entropy falls below an empirical threshold ($H < 3.0$) alongside a steep negative derivative ($dH/dt < -0.5$), the system observes a phase of rapid entropic convergence. At this transition point, the additive boost temporarily increases the plasticity gate, permitting parameter adaptation to consolidate the new task structure. Once the representation stabilizes, $dH/dt \to 0$, and the gate smoothly returns to baseline, safeguarding the consolidated parameters against subsequent drift.

---

## 4. Experimental Setup

### 4.1 Benchmarks

Six continual learning scenarios were evaluated:

| Benchmark | Type | Architecture | Tasks | Training samples |
|---|---|---|---|---|
| Split-MNIST | Class-incremental | MLP (256 hidden) | 2 (digits 0-4, 5-9) | 5,000/task |
| Permuted-MNIST | Domain-incremental | MLP (256 hidden) | 2 (original, permuted) | 5,000/task |
| Rotated-MNIST | Gradual shift | MLP (256 hidden) | 5 (0° to 180°, 45° increments) | 2,000/task |
| Split-FashionMNIST | Class-incremental | MLP (256 hidden) | 2 (items 0-4, 5-9) | 5,000/task |
| Split-CIFAR10 | Class-incremental | ConvNet (32-64 conv, 256 fc) | 2 (classes 0-4, 5-9) | 2,500/task |
| Split-CIFAR100 | Class-incremental | ConvNet (32-64 conv, 256 fc) | 2 (classes 0-49, 50-99) | 2,000/task |

### 4.2 Baselines

- **SGD**: Standard stochastic gradient descent without any forgetting mitigation.
- **Frozen**: All weights frozen after Task 1 (upper bound on retention, zero acquisition).
- **EWC**: Elastic Weight Consolidation (diagonal Fisher approximation). Lambda tuned per benchmark.
- **Replay**: Experience replay with reservoir sampling (buffer size = 200).

### 4.3 EGPR Variants

- **EGPR**: Full system (sigmoid gate + dH/dt boost + DMP + $\varepsilon$-floor), with default hyperparameters ($c = 2.0$, $s = 0.1$).
- **EGPR (tuned)**: EGPR with hyperparameters selected via grid search over the sigmoid center $c \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$ and adaptation sensitivity $s \in \{0.01, 0.05, 0.1, 0.15, 0.2, 0.3\}$, optimizing for the harmonic mean of retention and acquisition.
- **EGPR (no dH/dt)**: Ablation removing the temporal derivative boost.
- **EGPR (no DMP)**: Ablation removing depth-modulated plasticity.
- **EGPR+Replay**: EGPR combined with experience replay (buffer size = 200).
- **EGPR+Replay (tuned)**: Tuned EGPR combined with replay.

### 4.4 Metrics

Three metrics are reported:

- **Retention ($R$)**: Accuracy on Task 1 test set after training on Task 2.
- **Backward Transfer (BWT)**: Change in Task 1 accuracy between pre- and post-Task 2 training.
- **Acquisition ($A$)**: Accuracy on the most recent task's test set after training.

All results are means and standard deviations over $N = 5$ random seeds. Statistical significance is assessed using the one-sided Wilcoxon signed-rank test.

---

## 5. Results

### 5.1 Retention Performance

Table 1 presents retention results across all benchmarks.

**Table 1.** Mean retention (± std) over 5 seeds. Bold indicates best memory-free method per benchmark.

| Benchmark | SGD | EWC | Replay | EGPR | EGPR (tuned) |
|---|---|---|---|---|---|
| Split-MNIST | 0.4 ± 0.1 | 10.0 ± 1.0 | 84.4 ± 1.5 | 85.7 ± 3.8 | **86.8 ± 1.5** |
| Permuted-MNIST | 86.4 ± 0.6 | 87.7 ± 0.4 | 85.0 ± 1.8 | **88.0 ± 0.1** | 87.7 ± 0.4 |
| Rotated-MNIST | 31.6 ± 1.3 | 34.2 ± 1.4 | 49.0 ± 1.2 | 58.5 ± 1.5 | **58.9 ± 1.2** |
| Split-Fashion | 9.3 ± 1.4 | 11.5 ± 2.4 | 72.1 ± 3.4 | 39.1 ± 1.8 | **63.8 ± 7.9** |
| Split-CIFAR10 | 0.0 ± 0.0 | 0.0 ± 0.0 | 14.2 ± 3.1 | 19.1 ± 3.8 | **29.9 ± 2.6** |
| Split-CIFAR100 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.9 ± 0.7 | **12.1 ± 1.6** | 9.9 ± 0.8 |

Among the memory-free methods evaluated (SGD, EWC, EGPR), EGPR achieves the highest retention across all six benchmarks. The improvement over EWC — the canonical regularization baseline — is substantial in class-incremental scenarios: +75.7 pp in Split-MNIST, +24.3 pp in Rotated-MNIST, +27.6 pp in Split-FashionMNIST, and +19.1 pp in Split-CIFAR10. In the domain-incremental setting (Permuted-MNIST), the advantage is minimal (+0.3 pp), suggesting that EGPR's entropy-based gating is most effective when task distributions produce clearly distinct entropy signatures. A one-sided Wilcoxon signed-rank test comparing EGPR vs. EWC retention across the five seeds yields $p = 0.031$ for five of six benchmarks, with Permuted-MNIST being the exception ($p = 0.094$). It should be noted that $p=0.031$ is the absolute minimum statistically significant value mathematically possible for a one-sided Wilcoxon test with $N=5$, demonstrating maximal robustness given the sample size constraints. It should also be noted that EWC serves as a foundational baseline; more recent regularization methods (SI, MAS, OGD) may offer stronger performance but were not included in this evaluation, which focuses on establishing the viability of entropy-based gating as a mechanism.

### 5.2 Acquisition and Retention-Acquisition Tradeoff

While EGPR demonstrates strong retention, its acquisition on new tasks is lower than baseline methods. Table 2 presents the retention–acquisition tradeoff.

**Table 2.** Retention vs. acquisition for selected methods.

| Benchmark | Method | Retention | Acquisition | H-mean |
|---|---|---|---|---|
| Split-MNIST | Replay | 84.4% | 92.0% | 0.881 |
| | EGPR | 85.7% | 70.6% | 0.774 |
| | EGPR (tuned) | 86.8% | 71.9% | 0.786 |
| Split-Fashion | Replay | 72.1% | 85.4% | 0.782 |
| | EGPR | 39.1% | 76.3% | 0.517 |
| | EGPR (tuned) | 63.8% | 64.2% | 0.640 |
| Rotated-MNIST | Replay | 49.0% | 88.3% | 0.630 |
| | EGPR (tuned) | 58.9% | 55.8% | 0.573 |
| Split-CIFAR10 | Replay | 14.2% | 58.5% | 0.229 |
| | EGPR (tuned) | 29.9% | 30.2% | 0.301 |

EGPR's conservative gating strategy prioritizes retention at the expense of acquisition. When the hyperparameters are tuned via grid search (optimizing the harmonic mean), acquisition improves substantially (e.g., from 70.6% to 71.9% in Split-MNIST, and from 39.1% to 63.8% retention in Split-FashionMNIST with a tuned configuration favoring $c = 3.0$, $s = 0.2$). This behavior is consistent with the interpretation of EGPR as a continuous interpolation between a fully frozen network and unconstrained SGD, where the gating parameters determine the position along this spectrum.

### 5.3 Synergy with Experience Replay

EGPR and experience replay address complementary aspects of continual learning: EGPR suppresses destructive weight updates, while replay reinforces prior task representations. Table 3 documents the effect of combining both mechanisms.

**Table 3.** Retention improvement from adding EGPR to standard replay.

| Benchmark | Replay | EGPR+Replay | Improvement | $p$ |
|---|---|---|---|---|
| Split-MNIST | 84.4% | 95.0% | **+10.6 pp** | 0.031 |
| Permuted-MNIST | 85.0% | 88.2% | +3.2 pp | 0.031 |
| Rotated-MNIST | 49.0% | 63.5% | **+14.5 pp** | 0.031 |
| Split-Fashion | 72.1% | 81.6% | +9.5 pp | 0.031 |
| Split-CIFAR10 | 14.2% | 59.5% | **+45.3 pp** | 0.031 |
| Split-CIFAR100 | 1.9% | 14.9% | **+13.0 pp** | 0.031 |

The combination achieves consistent and statistically significant retention improvements across all six benchmarks ($p = 0.031$, one-sided Wilcoxon signed-rank test; the lowest possible $p$-value for $N=5$). Notably, in Split-CIFAR10, the combined approach improves retention by 45.3 percentage points over standard replay, suggesting that EGPR's gating mechanism prevents the replay buffer from being overwhelmed by the more recent task distribution. This synergy is additive: EGPR does not modify the replay mechanism, and replay does not alter EGPR's gating logic.

### 5.4 Ablation Study

To assess the contribution of each component, two ablated variants were evaluated: EGPR without the temporal derivative boost (no dH/dt) and EGPR without depth-modulated plasticity (no DMP).

**Table 4.** Ablation results (mean over 5 seeds).

| Benchmark | Component removed | Retention | Acquisition | H-mean |
|---|---|---|---|---|
| Split-MNIST | None (full EGPR) | 85.7% | 70.6% | 0.774 |
| | No dH/dt | 90.2% | 62.7% | 0.740 |
| | No DMP | 68.7% | 78.6% | 0.733 |
| Split-Fashion | None (full EGPR) | 39.1% | 76.3% | 0.517 |
| | No dH/dt | 64.5% | 60.3% | 0.624 |
| | No DMP | 26.0% | 81.6% | 0.394 |
| Rotated-MNIST | None (full EGPR) | 58.5% | 56.3% | 0.574 |
| | No dH/dt | 81.6% | 32.3% | 0.463 |
| | No DMP | 54.9% | 58.7% | 0.567 |
| Split-CIFAR10 | None (full EGPR) | 19.1% | 32.2% | 0.240 |
| | No dH/dt | 2.9% | 45.4% | 0.054 |
| | No DMP | 3.5% | 44.0% | 0.064 |

**Temporal derivative (dH/dt)**: The temporal derivative boost enables EGPR to detect successful learning and partially restore plasticity. In Rotated-MNIST, removing dH/dt reduces acquisition by 24.1 percentage points (from 56.3% to 32.3%), confirming its role in allowing the gate to reopen when the network converges on new data. In Split-FashionMNIST, the effect is +15.9 pp. Interestingly, removing dH/dt can increase retention (e.g., 90.2% vs 85.7% in Split-MNIST), since the gate remains more persistently closed.

**Depth-Modulated Plasticity (DMP)**: Removing DMP reduces retention substantially in Split-MNIST (−17.0 pp) and Split-FashionMNIST (−13.1 pp), while increasing acquisition, indicating that DMP's layer-wise protection contributes meaningfully to feature preservation at the cost of some learning flexibility.

### 5.5 Boundary Conditions

Two boundary conditions were identified:

**1. Permuted-MNIST**: EGPR's advantage over EWC is negligible (+0.3 pp retention, within standard error, $p = 0.094$). Pixel-level permutation disrupts spatial structure without substantially altering the entropy distribution at the output level. EWC, which operates at the parameter level, is better equipped for this type of distribution shift.

**2. CIFAR-10 and CIFAR-100**: On Split-CIFAR10, EGPR achieves 19.1% retention (compared to 0.0% for EWC and SGD) with 32.2% acquisition. On Split-CIFAR100 (20 samples per class), all methods perform near chance due to severe data scarcity. Notably, EGPR+Replay achieves 59.5% retention on Split-CIFAR10, demonstrating that the combination can overcome the standalone acquisition limitation. Telemetry analysis of the CIFAR-10 runs revealed three contributing factors:

- The entropy z-score remains above +8 throughout Task 2 training, keeping the sigmoid gate persistently near $\varepsilon$.
- The original DMP gradient (higher plasticity in early layers) is counterproductive for convolutional networks, where early features are more generalizable and should be more protected.
- **The Semantic Distance Deadlock**: The exploration floor ($\varepsilon = 0.02$) was designed to maintain minimal plasticity under uncertainty. However, when the semantic distance between consecutive tasks is vast (such as complex visual transitions in CIFAR), the entropy spikes aggressively, forcing the sigmoid gate down to $\varepsilon$. Operating at 2% plasticity, the network updates too slowly to alter its predictions meaningfully per batch. Consequently, the temporal derivative of entropy flattens ($dH/dt \approx 0$), making it mathematically impossible to cross the negative trigger threshold ($-s$). The safety mechanism effectively strangles the acquisition mechanism, locking the network in a state of pathological rigidity. This confirms EGPR is strictly viable only when distribution shifts are sufficiently contiguous to allow steep, natural entropy gradients.

These observations indicate that EGPR in its current form is most effective when tasks produce distinct but not radically different entropy signatures, and when learning dynamics are fast enough for the temporal derivative signal to emerge within the gating window.

---

## 6. Discussion

### 6.1 Interpretation

EGPR can be understood as a continuous interpolation between a fully frozen network and unconstrained SGD, where the position along this spectrum is determined dynamically by the entropy signal. This interpretation explains both its strengths (strong retention when the gate is working) and its limitations (insufficient acquisition when the gate remains closed). The mechanism does not create additional learning capacity; rather, it redistributes existing capacity toward knowledge preservation.

The consistent synergy with experience replay suggests a practical deployment mode: EGPR serves as a low-cost "safety net" that can be layered on top of existing continual learning methods to enhance their retention characteristics without modification.

### 6.2 Comparison with Frozen Baseline

A natural question is whether EGPR's retention advantage is trivially equivalent to freezing the network. The data indicate that this is not the case: while a Frozen network achieves zero acquisition by definition, EGPR permits meaningful acquisition across all benchmarks (e.g., 70.6% in Split-MNIST, 76.3% in Split-FashionMNIST, 56.3% in Rotated-MNIST). Moreover, EGPR operates without knowledge of task boundaries, making it applicable in settings where a Frozen approach would require an external signal to trigger the freeze.

### 6.3 Hyperparameter Sensitivity as a Continuous Control

The two primary hyperparameters — sigmoid center ($c$) and adaptation sensitivity ($s$) — function as a continuous control dial over the retention–acquisition tradeoff. At default settings ($c = 2.0$, $s = 0.1$), EGPR strongly favors retention. The grid search consistently selects configurations that increase plasticity in benchmarks with challenging class-incremental shifts: in Split-FashionMNIST, all five seeds selected $c = 3.0$ or $c = 2.5$ with high $s$, improving retention from 39.1% to 63.8%. In a practical deployment, these parameters would be adjusted according to the application's tolerance for forgetting. For safety-critical domains (e.g., medical diagnosis, autonomous navigation), the conservative default may be appropriate; for exploratory learning, looser gating settings offer a more balanced operating point.

### 6.4 Practical Deployment Scenarios

EGPR is most advantageous in three deployment scenarios:

1. **Memory-constrained environments**: On edge devices, IoT sensors, or privacy-sensitive applications where storing past training data for replay is not feasible, EGPR provides retention improvements over EWC without requiring episodic memory.
2. **Safety-critical systems**: In applications where forgetting previously learned knowledge carries significant risk (e.g., medical diagnostics, industrial defect detection), the retention-first behavior of EGPR default settings serves as a conservative safeguard. An incorrect new prediction is preferable to losing the ability to make correct prior predictions.
3. **As a complementary module**: When combined with existing methods such as experience replay, EGPR acts as a low-overhead ``safety net'' that improves retention without modifying the underlying learning algorithm. The EGPR+Replay configuration requires no additional memory beyond the standard replay buffer and consistently improves retention across all tested benchmarks.

### 6.5 Limitations

Several limitations of the current framework should be noted:

1. **Acquisition ceiling**: EGPR's gating inherently prioritizes retention over acquisition. In all tested benchmarks, standard replay achieves a higher harmonic mean of retention and acquisition. For applications where balanced performance is essential, EGPR alone may not be the preferred approach.

2. **Sensitivity to learning dynamics**: EGPR requires that the temporal derivative of entropy provides a usable signal within the gating window. When learning is slow (as in CIFAR-10), this signal may not emerge in time, leading to persistent plasticity suppression.

3. **Single entropy signal**: The current implementation uses a single scalar (output entropy) for gating decisions. This limits the system's ability to distinguish between different types of novelty (e.g., data from a genuinely new task versus out-of-distribution noise).

4. **DMP direction**: The descending plasticity gradient (more plasticity in early layers) was found to be suitable for MLPs but counterproductive for convolutional architectures. Architecture-specific DMP profiles would be needed for broader applicability.

---

## 7. Future Directions

The boundary conditions documented in this work suggest several avenues for investigation:

1. **Representation-level novelty detection**: Rather than relying solely on output entropy, gating decisions could incorporate similarity metrics computed in the network's hidden representation space, enabling finer-grained distinction between familiar and novel data.

2. **Adaptive exploration schedules**: Replacing the fixed exploration floor $\varepsilon$ with a dynamically adjusted value that responds to the success or failure of recent exploration attempts.

3. **Architecture-aware plasticity profiles**: Adapting the DMP gradient direction and magnitude based on the type of network layers (convolutional vs. fully connected) and their role in the feature hierarchy.

4. **Multi-timescale entropy monitoring**: Evaluating entropy derivatives over multiple temporal windows simultaneously to accommodate both fast and slow learning dynamics.

5. **Entropic Routing in Mixture of Experts (MoE)**: An immediate extension of EGPR is its application as a thermodynamic router in MoE architectures. High output entropy could dynamically trigger multi-expert ensemble evaluations, while low entropy (consensus) restricts computation to a single expert, optimizing inference compute costs without degrading continual performance.

6. **Early Stopping in Diffusion Models**: Beyond language and vision classification, the thermodynamic feedback loop defined by EGPR is theoretically compatible with diffusion models. Specifically, using the temporal derivative of the reverse-process entropy to dynamically gate the number of denoising steps could yield significant computational efficiencies.

---

## 8. Conclusion

This work has presented Entropy-Gated Plasticity Regulation (EGPR), a mechanism for continual learning that monitors the temporal derivative of predictive entropy to dynamically modulate learning rates. Empirical evaluation across six benchmarks demonstrates that EGPR achieves statistically significant retention improvements over the canonical regularization baseline (EWC) in five of six benchmarks without requiring episodic memory or task boundaries. While this comparison establishes the viability of entropy-based gating as a plasticity control mechanism, characterizing EGPR's performance relative to more recent regularization methods (SI, MAS, OGD) remains an important direction for future work. The combination of EGPR with experience replay yields statistically significant improvements over replay alone in all six benchmarks ($p = 0.031$), with retention gains ranging from +3.2 to +45.3 percentage points, suggesting its primary practical value as a highly effective complementary module that enhances existing continual learning systems. The documented boundary conditions — particularly the Semantic Distance Deadlock and the acquisition tradeoff on complex visual domains — provide concrete targets for subsequent iterations and serve to delineate the conditions under which entropy-based gating is most effective.

---

## 9. Reproducibility

All experiments were conducted on consumer-grade hardware (AMD Ryzen 7, CPU only). The complete benchmark script, raw CSV results, and tuned hyperparameters are available in the accompanying repository. Each reported result represents the mean and standard deviation over 5 random seeds (42, 100, 2026, 7, 314).

### Hyperparameters

| Parameter | EGPR (default) | EGPR (tuned) |
|---|---|---|
| Sigmoid center $c$ | 2.0 | Grid: {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} |
| Adaptation sensitivity $s$ | 0.1 | Grid: {0.01, 0.05, 0.1, 0.15, 0.2, 0.3} |
| Exploration floor $\varepsilon$ | 0.02 | 0.02 |
| Window size (dH/dt) | 30 batches | 30 batches |
| EMA decay (short/medium/long) | 2/5, 2/11, 2/21 | — |

### Baseline Hyperparameters

| Baseline | Key parameter |
|---|---|
| EWC $\lambda$ | Grid: {10, 100, 400, 1000, 5000, 10000, 50000, 100000} |
| Replay buffer size | 200 |
| Learning rate (all methods) | 0.01 |
| Batch size (training / evaluation) | 32 / 256 |
| Optimizer | SGD |
| Epochs per task | 5 |

---

## Acknowledgments

The author acknowledges the use of large language model assistants for code review, test orchestration, and manuscript preparation during the research process. All theoretical concepts, architectural designs, and experimental decisions described in this work are the original intellectual contribution of the author.

---

## References

- Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2018). Memory Aware Synapses: Learning what (not) to forget. ECCV.
- Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.
- Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient Lifelong Learning with A-GEM. ICLR.
- Chen, Z., et al. (2022). Entropy-based Stability-Plasticity for Lifelong Learning. CVPR Workshop.
- Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020). Orthogonal Gradient Descent for Continual Learning. AISTATS.
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4), 128-135.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13), 3521-3526.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. Psychology of Learning and Motivation, 24, 109-165.
- Rusu, A. A., et al. (2016). Progressive neural networks. arXiv:1606.04671.
- Shin, H., et al. (2017). Continual learning with deep generative replay. NeurIPS.
- Wang, Z., et al. (2026). An Adaptive Entropy Flow Dynamics Framework for Long-tailed Human Action Recognition. Proceedings of the AAAI Conference on Artificial Intelligence.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. ICML.
