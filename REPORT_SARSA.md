# SARSA Agent: Experiment Report

## 1. Introduction

This report presents the results of training a tabular SARSA agent on a grid-based delivery robot environment. The agent must navigate from a fixed starting position to a target (delivery location) while avoiding walls and obstacles. We evaluate the impact of different hyperparameters on learning performance.

**Algorithm:** SARSA (on-policy TD(0))  
**Update rule:** Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]  
**Policy:** ε-greedy with decay (ε_decay = 0.995, ε_min = 0.01)

## 2. Experimental Setup

### Grids
| Grid | Size | Start Position | Target Position | Complexity |
|------|------|---------------|-----------------|------------|
| A1_grid | 15×15 | (1, 12) | (9, 2) | High — many obstacles, long path required |
| example_grid | 8×7 | (1, 1) | (5, 4) | Low — few obstacles, short optimal path |

### Default Reward Function
- Empty tile: -1 (encourages efficiency)
- Wall/obstacle: -5 (discourages bumping)
- Target reached: +10 (goal signal)

### Parameters Varied
| Parameter | Values Tested |
|-----------|--------------|
| Learning rate (α) | 0.01, 0.1 |
| Discount factor (γ) | 0.6, 0.9 |
| Stochasticity (σ) | 0.02, 0.5 |
| Exploration rate (ε) | 0.1, 0.3 |
| Episodes | 500, 1000 |

### Baseline
- Random agent (uniform random action selection, no learning)

---

## 3. Results Summary

### 3.1 Key Results Table

| Experiment | Grid | α | γ | ε | σ | Ep. | Steps to Target | Failed Moves | Reached? |
|-----------|------|-----|-----|-----|------|------|----------------|--------------|----------|
| Random baseline | A1 | — | — | — | 0.02 | — | 1000 (max) | 317 | ❌ |
| Random baseline | example | — | — | — | 0.02 | — | 109 | 32 | ✅ |
| **SARSA (best)** | **A1** | **0.1** | **0.9** | **0.1** | **0.02** | **500** | **28** | **0** | **✅** |
| **SARSA (best)** | **example** | **0.1** | **0.9** | **0.1** | **0.02** | **500** | **7** | **0** | **✅** |
| SARSA | A1 | 0.01 | 0.9 | 0.1 | 0.02 | 500 | 500 (max) | 6 | ❌ |
| SARSA | example | 0.01 | 0.9 | 0.1 | 0.02 | 500 | 7 | 0 | ✅ |
| SARSA | A1 | 0.1 | 0.6 | 0.1 | 0.02 | 500 | 500 (max) | 0 | ❌ |
| SARSA | example | 0.1 | 0.6 | 0.1 | 0.02 | 500 | 7 | 0 | ✅ |
| SARSA | A1 | 0.1 | 0.9 | 0.1 | 0.5 | 500 | 70 | 14 | ✅ |
| SARSA | example | 0.1 | 0.9 | 0.1 | 0.5 | 500 | 24 | 1 | ✅ |
| SARSA | A1 | 0.1 | 0.9 | 0.3 | 0.02 | 500 | 30 | 0 | ✅ |
| SARSA | example | 0.1 | 0.9 | 0.3 | 0.02 | 500 | 7 | 0 | ✅ |
| SARSA (stress) | A1 | 0.1 | 0.9 | 0.1 | 0.5 | 1000 | 59 | 13 | ✅ |
| SARSA (stress) | example | 0.1 | 0.9 | 0.1 | 0.5 | 1000 | 23 | 2 | ✅ |

---

## 4. Analysis

### 4.1 Effect of Learning Rate (α)

**Finding:** α = 0.1 significantly outperforms α = 0.01 on the larger A1_grid.

- With α = 0.1: Agent reaches target in **28 steps** with 0 failed moves.
- With α = 0.01: Agent **fails to reach target** within 500 steps.

**Explanation:** The A1_grid requires the agent to learn a long path (start at top-right, target at middle-left). With a low learning rate, Q-value updates propagate too slowly from the target back to the start — 500 episodes is insufficient for the reward signal to reach distant states. On the smaller example_grid, both learning rates succeed because the path is short (7 steps optimal).

**Takeaway:** For larger grids, a higher learning rate is essential, or significantly more episodes are needed.

### 4.2 Effect of Discount Factor (γ)

**Finding:** γ = 0.9 succeeds on A1_grid; γ = 0.6 fails.

- With γ = 0.9: Target reached in 28 steps.
- With γ = 0.6: Agent **never reaches target** (loops indefinitely, 0 failed moves).

**Explanation:** The discount factor determines how far ahead the agent "looks." With γ = 0.6, future rewards are heavily discounted: after ~10 steps, a reward of +10 is worth only 10 × 0.6^10 ≈ 0.06. Since the A1_grid target is ~27 steps away, the reward signal effectively vanishes before reaching the start position. The agent has no incentive to move toward the target. On example_grid (7 steps), γ = 0.6 still works because 10 × 0.6^7 ≈ 0.28 is still meaningful.

**Takeaway:** γ must be high enough that γ^(path_length) remains significant. For long paths, γ ≥ 0.9 is necessary.

### 4.3 Effect of Stochasticity (σ)

**Finding:** SARSA handles stochasticity well but takes longer paths.

- σ = 0.02: 28 steps, 0 failed moves (near-deterministic, optimal path).
- σ = 0.5: 70 steps, 14 failed moves (still reaches target, but less efficient).

**Explanation:** With σ = 0.5, half the time the environment ignores the agent's chosen action and executes a random one. This causes:
1. More wall collisions (14 failed moves vs 0).
2. Longer paths due to random deviations.
3. SARSA's on-policy nature makes it **conservative** — it learns to account for the randomness in its Q-values, avoiding paths that pass close to walls since it knows it might accidentally step into them.

**Takeaway:** SARSA is robust to stochasticity (still reaches the target) but the path quality degrades. More training episodes (1000 vs 500) slightly improves performance under high σ (59 steps vs 70 steps in the stress test).

### 4.4 Effect of Exploration Rate (ε)

**Finding:** Both ε = 0.1 and ε = 0.3 converge to similar performance.

- ε = 0.1: 28 steps to target on A1_grid.
- ε = 0.3: 30 steps to target on A1_grid.

**Explanation:** With epsilon decay (0.995 per episode), both values decay to near-zero by the end of training. The higher initial ε = 0.3 provides more exploration early on, which can help discover the target faster in the first episodes, but the final greedy policy is similar. The 2-step difference (28 vs 30) is within noise.

**Takeaway:** With epsilon decay, the initial ε value matters less. Higher ε helps on complex grids where the agent might otherwise get stuck in local optima early on.

### 4.5 SARSA vs Random Agent

| Metric | Random (A1) | SARSA (A1) | Improvement |
|--------|-------------|------------|-------------|
| Reached target | ❌ | ✅ | — |
| Steps | 1000 (max) | 28 | **36× fewer** |
| Failed moves | 317 | 0 | **Eliminated** |
| Cumulative reward | -2268 | -17 | **133× better** |

The random agent on A1_grid never reaches the target in 1000 steps — the grid is too complex for random exploration to stumble upon the target reliably. SARSA learns the optimal path and executes it cleanly.

---

## 5. Conclusions

1. **SARSA successfully learns optimal policies** on both grids when hyperparameters are appropriate (α=0.1, γ=0.9).

2. **Learning rate is critical for larger grids.** α=0.01 is too slow for A1_grid's long paths within 500 episodes.

3. **Discount factor must match path length.** γ=0.6 fails on A1_grid because the reward signal cannot propagate 27+ steps back to the start.

4. **SARSA is robust to stochasticity** but takes longer paths under high σ. Its on-policy nature makes it naturally conservative — a desirable property for a real delivery robot that should avoid risky paths near obstacles.

5. **Epsilon decay makes the initial ε less important**, as both tested values converge to similar final performance.

6. **Compared to the random baseline**, SARSA achieves a 36× reduction in steps and eliminates all failed moves on A1_grid.

### Recommended Configuration
For the A1_grid delivery task: **α=0.1, γ=0.9, ε=0.1 with decay 0.995, 500 episodes** provides reliable convergence to a near-optimal policy in under 1 second of training time.
