---
layout: default
title: Status
---

## Project Summary

FlyToSky trains a quadrotor drone to autonomously fly through sequences of gate waypoints using deep reinforcement learning. Rather than relying on hand-tuned cascaded PID controllers, the agent learns a direct mapping from raw sensor observations — body-frame velocities, angular rates, rotor speeds, and relative waypoint positions — to continuous motor commands. The environment simulates a Crazyflie-class nano-quadrotor with realistic physics, including a quadratic thrust curve and first-order motor delay dynamics. Training uses Proximal Policy Optimization (PPO) with a curriculum that progressively increases track difficulty from straight-line gates to random 3D walks, while simultaneously ramping up dynamics randomization to encourage robustness.

## Approach

### Algorithm: Proximal Policy Optimization (PPO)

We use PPO  with Generalized Advantage Estimation (GAE). The policy is a diagonal Gaussian: given observation $s_t$, the actor outputs a mean $\mu_\theta(s_t) \in \mathbb{R}^4$ and a learned (state-independent) log-standard deviation, so actions are sampled as $a_t \sim \mathcal{N}(\mu_\theta(s_t), \text{diag}(\sigma^2))$.

**Clipped surrogate objective:**

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left( r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the probability ratio and $\hat{A}_t$ is the GAE advantage estimate.

**Combined loss** (policy + value + entropy):

$$L(\theta) = -L^{\text{CLIP}}(\theta) + c_v L^{\text{VF}}(\theta) - c_e H[\pi_\theta]$$

where $L^{\text{VF}}$ is a clipped MSE value loss and $H$ is the policy entropy bonus.

**GAE advantage** with discount $\gamma$ and trace $\lambda$:

$$\hat{A}_t = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### Network Architecture

Both actor and critic share a common encoder: two fully-connected layers of 256 hidden units each with Tanh activations, initialized with orthogonal weight matrices (gain $\sqrt{2}$). The actor head is a linear projection to 4 outputs (std 0.01 init) and the critic head is a linear projection to a scalar (std 1.0 init). The $\log \sigma$ vector is a free parameter initialized to zero.

### Environment and Physics

The simulation models a Crazyflie 2.x nano-quadrotor (mass 36 g, ~28 mm arm length). Physics is integrated in PyTorch on GPU with Euler steps at $\Delta t = 0.01$ s, with 2 decimation sub-steps per policy step. Key elements:

- **Thrust model:** Each rotor produces thrust via a quadratic polynomial in RPM: $F_i = c_0 + c_1 \omega_i + c_2 \omega_i^2$ (coefficients fit from hardware data).
- **Motor delay:** Rotor speed tracks a commanded value with separate first-order time constants for spin-up ($\tau_r \approx 83$ ms) and spin-down ($\tau_f \approx 284$ ms).
- **Torques:** Roll/pitch torques from cross products of rotor thrust vectors with arm positions; yaw torque from signed reaction torque constants.
- **Attitude integration:** Quaternion kinematics $\dot{q} = \tfrac{1}{2} q \otimes \omega_q$, normalized each step.

### Observations (25-dimensional)

| Component | Dim | Description |
|---|---|---|
| Body-frame velocity | 3 | $v$ rotated into body frame |
| Angular velocity | 3 | $\omega$ in body frame (rad/s) |
| Gravity direction | 3 | $[0,0,-1]^\top$ rotated into body frame |
| Orientation error | 3 | Axis-angle error between current and desired yaw quaternion |
| Scaled RPM | 4 | $\omega_i / \omega_\text{max}$ for each rotor |
| Relative waypoints | 9 | Body-frame offset to next 3 gates (3D each) |

### Actions (4-dimensional)

Actions are continuous values in $[-1, 1]$, linearly mapped to RPM commands in $[0, 21679]$ for each of the four rotors.

### Reward Function

The total reward per step is:

$$r = r_\text{progress} + r_\text{gate} + r_\text{smooth} + r_\text{ang} + r_\text{orient}$$

| Component | Formula | Scale |
|---|---|---|
| Progress | $\Delta d = \|p_t - g\| - \|p_{t+1} - g\|$ | $\times 100$ |
| Gate passage | $\mathbf{1}[\text{crossed gate within 1.5 m}]$ | $\times 5$ |
| Action smoothness | $-\|\Delta u\| \cdot \Delta t$ | $\times -0.7$ |
| Angular velocity | $-\|\omega\|^2 \cdot \Delta t$ | $\times -0.01$ |
| Orientation | $(1 - \tanh(\|\text{err}\|/0.5)) \cdot \Delta t$ | $\times 100$ |

Episodes terminate when the drone crashes (z < 0.1 m or z > 5 m), drifts more than 8 m from the current target gate, or completes all waypoints on the track.

### Curriculum and Track Generation

Training uses a 6-level curriculum managed by `CurriculumScheduler`. The level advances when the 100-episode rolling mean reward exceeds a threshold for 5 consecutive evaluations. Track difficulty and dynamics randomization increase together:

| Level | Track types (distribution) | Dynamics $\delta$ |
|---|---|---|
| 0 | Straight (100%) | 0% |
| 1 | Straight flat/height (50/50%) | 2% |
| 2 | Circle 50%, circle+height 25%, straight+height 25% | 4% |
| 3 | Circle+height 35%, circle 25%, straight mixes 40% | 6% |
| 4 | Random walk 40%, circle mixes 45%, straight+height 15% | 8% |
| 5 | Random walk mixes 60%, circle mixes 30%, straight+height 10% | 10% |

At each episode reset, thrust coefficients, torque constants, rotor positions, and motor time constants are each independently perturbed by up to $\pm\delta$ of their nominal values.

### Hyperparameters

TODO: REPLACE WITH ACTUAL USED PARAMETERS



| Hyperparameter | Value |
|---|---|
| Parallel environments | 64 |
| Rollout steps per update | 128 |
| Total timesteps | 50,000,000 |
| Learning rate | 3e-4 (linearly annealed to 0) |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\varepsilon$ | 0.2 |
| Entropy coefficient $c_e$ | 0.01 |
| Value coefficient $c_v$ | 0.5 |
| Update epochs per batch | 4 |
| Minibatches per epoch | 4 |
| Gradient norm clip | 0.5 |
| Hidden layer size | 256 |

Batch size = 64 × 128 = 8,192 transitions per update; minibatch size = 2,048.

## Evaluation

### Quantitative Evaluation

Training is monitored via rolling episode reward and episode length (averaged over the last 100 completed episodes). The primary metrics are:

- **Mean episode reward:** Measures overall policy quality integrating progress, gate passage, and penalty terms.
- **Mean episode length:** A proxy for survival and stability. Longer episodes indicate the drone stays airborne and continues progressing.
- **Gates passed per episode:** Directly quantifies navigation success on the track.
- **Curriculum level:** Tracks skill progression from easy to hard tracks.
- **Explained variance:** Measures value function quality; values near 1.0 indicate accurate return prediction.

The following plots will be populated once the full training run completes (target: 50M environment steps):

- Episode reward curve vs. training step
- Curriculum level over time
- Reward component breakdown (progress, gates passed, orientation, smoothness, angular velocity penalties)

### Qualitative Evaluation

We use a custom real-time renderer (via `Log.render`) to visualize the first environment during training. This outputs per-step telemetry including position, orientation, rotor speeds, thrust vectors, and waypoint positions, which is visualized using **Rerun**. Key qualitative checks:

- Does the drone point toward the next gate? (orientation component)
- Does it slow action changes between steps? (smoothness component)
- Does it recover from a poor spawn position and still reach the first gate?

Qualitative results (screen captures / Rerun recordings) will be embedded here once collected.

### Curriculum Validation

We verify each curriculum level independently by checking that the policy achieving threshold reward on level $k$ visually succeeds on the corresponding track geometry before the level advances. This prevents the curriculum from advancing prematurely due to reward hacking.

## Remaining Goals and Challenges

### Current Limitations

The current prototype implements a complete end-to-end PPO pipeline with a realistic physics simulator, curriculum learning, and dynamics randomization — but has not yet been trained to convergence. The main outstanding item is completing a full 50M-step training run and collecting quantitative results at each curriculum level. Additionally, the track generator's levels 4 and 5 (random walk) are functional but have not yet been stress-tested to confirm they present a smooth difficulty increase from level 3.

The environment currently terminates an episode as soon as the drone leaves a ±8 m radius of the current target gate, which can make early training sparse if the agent consistently overshoots. We plan to investigate whether a shaped "closest approach" fallback reward or a longer tolerated deviation radius improves sample efficiency at curriculum level 0.

### Goals for the Remainder of the Quarter

1. **Complete a full training run** to 50M steps and produce learning curves for all reward components and the curriculum level schedule.
2. **Gate passage evaluation:** Log the number of unique gates passed per episode as a clean task-success metric, separate from the dense reward signal.
3. **Baseline comparison:** Implement or import a simple PID hover-and-navigate controller and compare tracking error (RMSE to gate centers) against the trained RL policy on a fixed test track at each difficulty level.
4. **Sim-to-real considerations:** Identify which physical parameters are most sensitive under dynamics randomization, and discuss whether the trained policy's robustness bounds are plausible for real hardware deployment.
5. **Visualization and demo:** Record Rerun replay logs of the trained policy on held-out tracks at curriculum levels 0–5 and embed them or link them from the website.

## Resources Used

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Huang, S., Dossa, R. F. J., Ye, C., Braga, J., et al. (2022). *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms.* JMLR 23(274). [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
- Bitcraze AB. *Crazyflie 2.x hardware specifications.* [https://www.bitcraze.io](https://www.bitcraze.io)
- PufferLib: [https://github.com/PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)
- Rerun visualization SDK: [https://rerun.io](https://rerun.io)

- AI tools: Generative AI tools used for designing team website and generating boilplate code.