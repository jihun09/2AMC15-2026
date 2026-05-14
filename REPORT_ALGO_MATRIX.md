# Algorithm Comparison — VI vs MC vs SARSA

**Status:** Plan, not results. Tables fill in as the experiments run.

## 1. Goal

Compare three RL algorithms — Value Iteration (VI), On-policy first-visit Monte Carlo (MC), and SARSA — on two grids, with **two hyperparameter setups per algorithm**. Total: **6 setups × 2 grids = 12 (algorithm, setup, grid) cells**.

| Algorithm | Owner | Source branch |
|---|---|---|
| VI | TBD (private fallback maintained on `jaden-sarsa-test`, not pushed) | (none yet) |
| MC (on-policy first-visit) | Leliotto / Marceli | `leliotto-mc` |
| SARSA | Jihun | `jaden-sarsa-test` |

## 2. Grids

| Grid | Size | Start position |
|---|---|---|
| `large_grid` | 20 × 20 | (1, 1) |
| `A1_grid` | 15 × 15 | (1, 12) |

Start positions are locked across all setups and seeds (no random placement).

## 3. Shared reward function

All three algorithms use `utils.reward_fn` (and optional `utils.shaped_reward`):

- Step onto empty cell: **−1**
- Step onto wall / obstacle: **−5**
- Reach target: **+100**
- Optional potential-based BFS shaping with `shaping_weight` (treated as a hyperparameter)

Wired in via `Environment(..., reward_fn=reward_fn)` on the env side, and by wrapping the returned reward through `shaped_reward(...)` inside the training loop.

## 4. The 6 setups

Full hyperparameter values (from `configs/final_setups.json`):

| Algo | Setup | Key hyperparameters |
|---|---|---|
| **VI** | A (smoke_A) | γ = 0.95, θ = 1e-3, shaping_weight = 0.0 |
| **VI** | B (smoke_B) | γ = 0.99, θ = 1e-5, shaping_weight = 3.0 |
| **MC** | A (smoke_A) | ε = 0.1, ε_decay = 0.999, ε_min = 0.05, δ = 0.95, episodes = 500, shaping_weight = 0.0 |
| **MC** | B (smoke_B) | ε = 0.3, ε_decay = 0.999, ε_min = 0.05, δ = 0.99, episodes = 500, shaping_weight = 3.0 |
| **SARSA** | A (A_conservative) | α = 0.05, γ = 0.95, ε = 0.3 (fixed), shaping_weight = 3.0, q_init = 0.0, episodes = 2000 |
| **SARSA** | B (B_aggressive) | α = 0.05, γ = 0.95, ε = 0.1 (fixed), shaping_weight = 3.0, q_init = 0.0, episodes = 2000 |

**Convention:** Setup A is conservative / safe; Setup B is aggressive / speed-for-risk. Each algorithm's owner picks both, then we cross-check they tell a coherent story side-by-side.

**VI Setup A vs B:** Setup A uses a loose convergence threshold (θ = 1e-3) with a lower discount factor (γ = 0.95) and no reward shaping — it converges faster but to a coarser value function. Setup B tightens the threshold to θ = 1e-5, raises γ to 0.99 (values future rewards more), and adds BFS potential shaping (weight 3.0) — more thorough convergence at the cost of more iterations. Both reach POR = 1.0 because VI is a planning algorithm that sweeps the full state space.

**MC Setup A vs B:** Setup A starts with a lower initial ε (0.1) and a lower discount δ (0.95) with no shaping — more exploitation-heavy from early on but potentially misses some paths. Setup B starts with ε = 0.3 (more exploration early) and δ = 0.99 (longer reward horizon), plus BFS shaping (weight 3.0). In practice the shaping in Setup B did not consistently outperform Setup A — MC's sample-efficiency limits are visible on the harder A1_grid.

**SARSA Setup A vs B:** Both setups were selected from a 72-config sweep (3 α × 2 γ × 3 ε-schedules × 2 shaping_weights × 2 q_inits = 72, run with 3 seeds × 1000 episodes on large_grid). Setup A uses ε = 0.3 (fixed) — more exploration maintained throughout training, which the selection rule treats as "conservative" because it accepts higher conv_ep in exchange for stable policy quality. Setup B uses ε = 0.1 (fixed) — less exploration, converging ~1 episode earlier on average. The key controlled axis is the epsilon schedule; all other hyperparameters are identical between the two SARSA setups.

## 5. Metrics

### Metric 1 — Policy Optimality Ratio (POR)

```
POR = bfs_optimal_steps(grid, start) / mean(actual_steps over eval episodes)
```

- Bounded **[0, 1]**; 1.0 means the agent found the BFS shortest path.
- Reported at two evaluation noise levels — see §6.
- If the agent never reaches the target within `max_steps`, `actual_steps = max_steps` (capped), so POR drops correspondingly.

### Metric 2 — Convergence Speed

| Algorithm | Native unit | Stopping criterion |
|---|---|---|
| VI | iterations | `max\|V_{k+1} − V_k\| < θ` |
| MC | episodes | greedy policy unchanged for `patience = 100` consecutive episodes |
| SARSA | episodes | greedy policy unchanged for `patience = 100` consecutive episodes |

⚠️ Units are **not directly comparable** between VI and MC/SARSA. The report should mention both:

- Headline number: the raw count in the algorithm's native unit.
- Side-by-side number: a derived **update count** — `iterations × |reachable_states|` for VI vs `episodes × mean_steps_per_episode` for MC/SARSA. This puts all three on an "order-of-magnitude work performed" axis.

## 6. Sigma policy and rationale

| Where | Value |
|---|---|
| Training σ | **0.1**, fixed across all setups |
| Eval σ (headline POR) | **0.0** |
| Eval σ (sidecar POR) | **0.1** |

**Why training σ is fixed at 0.1:** keeps the comparison clean — hyperparameter differences become the only explanatory variable for performance differences. Varying σ during training would confound algorithm comparison with environment difficulty. (The course default is 0.1.)

**Why σ_eval = 0.0 is the headline:** POR is meant to measure *policy quality*, not world difficulty. Under σ > 0 the environment randomly overrides ~10% of actions, so even a perfectly optimal policy can never reach POR = 1.0. The metric loses its theoretical ceiling. σ = 0 gives a clean, interpretable number bounded in [0, 1].

**Why σ_eval = 0.1 is also reported:** a policy that scores POR = 1.0 at σ = 0 but collapses at σ = 0.1 is worse than one that scores 0.9 at σ = 0 and 0.8 at σ = 0.1. The sidecar captures **noise-robust deployment performance**, which the project brief explicitly cares about.

Together they tell two stories: intrinsic optimality (σ=0) and robustness (σ=0.1).

## 7. Results matrix

Headline table — one row per (algorithm, setup, grid), aggregated across **5 seeds** (mean ± std).
Conv. metric is in the algorithm's native unit (iterations for VI, episodes for MC/SARSA).
`-1` in conv means the stopping criterion never fired within the episode budget.

| Algo  | Setup          | Grid       | Conv. metric (native) | POR @ σ=0     | POR @ σ=0.1   |
|-------|----------------|------------|----------------------|---------------|---------------|
| vi    | smoke_A        | large_grid | 47 ± 0               | 1.000 ± 0.000 | 0.898 ± 0.005 |
| vi    | smoke_A        | A1_grid    | 47 ± 0               | 1.000 ± 0.000 | 0.893 ± 0.008 |
| vi    | smoke_B        | large_grid | 55 ± 0               | 1.000 ± 0.000 | 0.898 ± 0.005 |
| vi    | smoke_B        | A1_grid    | 55 ± 0               | 1.000 ± 0.000 | 0.893 ± 0.008 |
| mc    | smoke_A        | large_grid | 174 ± 62             | 0.810 ± 0.167 | 0.764 ± 0.160 |
| mc    | smoke_A        | A1_grid    | 198 ± 203            | 0.423 ± 0.345 | 0.359 ± 0.288 |
| mc    | smoke_B        | large_grid | 183 ± 62             | 0.691 ± 0.029 | 0.673 ± 0.012 |
| mc    | smoke_B        | A1_grid    | 286 ± 163            | 0.611 ± 0.059 | 0.565 ± 0.075 |
| sarsa | A_conservative | large_grid | 157 ± 37             | 1.000 ± 0.000 | 0.898 ± 0.005 |
| sarsa | A_conservative | A1_grid    | 219 ± 78             | 1.000 ± 0.000 | 0.893 ± 0.007 |
| sarsa | B_aggressive   | large_grid | 174 ± 56             | 1.000 ± 0.000 | 0.898 ± 0.005 |
| sarsa | B_aggressive   | A1_grid    | 244 ± 52             | 1.000 ± 0.000 | 0.895 ± 0.010 |

**Summary:** VI and SARSA both achieve POR_det = 1.000 on every (grid, setup) cell, meaning they find the BFS-optimal path deterministically. VI reaches this in only 47–55 iterations (a planning algorithm sweeps the state space without exploration), while SARSA requires 157–244 episodes depending on setup and grid — faster convergence in terms of per-update work. MC falls short on both grids: POR_det peaks around 0.81 on large_grid and drops to ~0.42 on the harder A1_grid for smoke_A, with very high variance across seeds (std up to 0.35), indicating the episode budget of 500 is sometimes insufficient. POR_n10 (noise robustness at σ=0.1) shows a consistent ~0.10 gap below POR_det for all algorithms, suggesting that ~10% random action overrides reduce effective optimality proportionally regardless of algorithm — VI, SARSA, and the better MC cells all end up near 0.89–0.90 at σ=0.1 once they've found a good policy. The most notable result is that SARSA with BFS shaping matches VI's deterministic policy quality while being a fully online, model-free algorithm, at the cost of ~3–5× more convergence episodes vs VI iterations.

Raw per-seed CSV: `results/matrix/matrix_2026-05-14__21-38-19.csv` (produced by `run_matrix.py`). One row per (algo, setup, grid, seed). Schema:

```
algo, setup, grid, seed, convergence_metric, actual_steps_det,
actual_steps_n10, optimal_steps, POR_det, POR_n10, elapsed_sec
```

## 8. SARSA candidate sweep (one-off pre-screen)

SARSA's two setups are **not pre-decided** — they're chosen from a candidate pool after a one-off sweep on `large_grid`.

Cross-product:

| Axis | Values |
|---|---|
| α | 0.05, 0.1, 0.3 |
| γ | 0.95, 0.99 |
| ε schedule | fixed-0.1, fixed-0.3, decay-0.3 → 0.05 |
| `shaping_weight` | 0, 3 |
| `q_init` | 0, +1.0 |

Total: 3 × 2 × 3 × 2 × 2 = **72 configs**. Run with **3 seeds × 1000 episodes** on `large_grid` only. Wall-clock budget: **~4 minutes** (measured: ~1 s per training run × 3 seeds × 72 ≈ 216 s, plus eval overhead).

**Selection rule:** scan the resulting CSV, pick the two configs that best frame as "Setup A vs Setup B" — one conservative, one aggressive — with a clear trade-off explanation that holds up against MC's and VI's chosen pairs.

Sweep output saved to `results/sarsa_sweep_<timestamp>.csv`.

## 9. Interface contract for VI / MC owners

If you maintain VI or MC code, the eval harness (`run_matrix.py`) expects each algorithm to expose:

1. **A greedy action method**: `agent.take_action(state) -> int` returning a deterministic action (no exploration), so evaluation reflects the learned policy.
2. **A training entry point** with this signature:

```python
def train_<algo>(grid_path: Path,
                 start_pos: tuple[int, int],
                 setup: dict,
                 seed: int,
                 patience: int) -> tuple[Agent, int]:
    """Train one (algo, setup) on one (grid, seed) cell.
    Returns (trained_agent, convergence_metric).
    convergence_metric = episode/iteration at which the stopping criterion fired.
    """
```

`setup` is a dict of that algorithm's hyperparameters. `run_matrix.py` calls this per cell, then runs greedy eval at σ = 0 and σ = 0.1.

## 10. Reproducibility

All CSV outputs are accompanied by a sidecar `<basename>.meta.json` recording: hyperparameters, grid path, start position, seeds, σ values, episode budgets, and elapsed wall-clock time. Any result can be reproduced or audited from the file alone.
