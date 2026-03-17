from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from flytosky.logging import Log
from flytosky.environment.quadcopter_env import QuadcopterEnv


# Maximum curriculum level supported by VectorizedTrackGenerator
MAX_CURRICULUM_LEVEL = 5

"""CleanRL-style PPO training loop for the FlyToSky QuadcopterEnv.

Supports:
- Curriculum-based learning: automatically advances track difficulty
  (levels 0-5) based on rolling mean episode reward thresholds.
- Dynamics randomization schedule: gradually increases randomization
  delta as the curriculum progresses.
- Full reward-component logging from the vectorized environment.
- Checkpoint save / resume with curriculum state preserved.
"""

def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared-encoder actor-critic with diagonal Gaussian policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs))

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs)
        action_mean = self.actor_mean(hidden)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(hidden)
        return action, log_prob, entropy, value

class CurriculumScheduler:
    """Advance curriculum level when rolling mean reward exceeds a threshold.

    Each level has a reward threshold.  Once the agent sustains a rolling
    mean episode reward above the threshold for ``patience`` consecutive
    evaluations the level advances.  The scheduler also linearly ramps
    dynamics randomisation delta with the level.
    """

    # Default reward thresholds per level #TODO adjust
    DEFAULT_THRESHOLDS: dict[int, float] = {
        0: 40.0,    # straight lines mastered
        1: 40.0,    # straight + variable height
        2: 40.0,   # circles
        3: 50.0,   # circles + variable height
        4: 40.0,   # random walk mixes
        # level 5 is terminal — no further promotion
    }

    def __init__(
        self,
        start_level: int = 0,
        max_level: int = MAX_CURRICULUM_LEVEL,
        thresholds: dict[int, float] | None = None,
        patience: int = 5,
        max_dynamics_delta: float = 0.1,
    ):
        self.level = start_level
        self.max_level = min(max_level, MAX_CURRICULUM_LEVEL)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.patience = patience
        self.max_dynamics_delta = max_dynamics_delta
        self._above_count = 0

    @property
    def completed(self) -> bool:
        """True when the target max level has been reached."""
        return self.level >= self.max_level

    def step(self, mean_reward: float) -> bool:
        """Returns True if the level was just advanced."""
        if self.level >= self.max_level:
            return False
        thresh = self.thresholds.get(self.level, float("inf"))
        if mean_reward >= thresh:
            self._above_count += 1
        else:
            self._above_count = 0
        if self._above_count >= self.patience:
            self.level = min(self.level + 1, self.max_level)
            self._above_count = 0
            return True
        return False

    @property
    def dynamics_delta(self) -> float:
        """Linearly scale dynamics randomisation with curriculum level."""
        if self.max_level == 0:
            return 0.0
        return self.max_dynamics_delta * (self.level / self.max_level)

    def state_dict(self) -> dict:
        return {"level": self.level, "max_level": self.max_level, "_above_count": self._above_count}

    def load_state_dict(self, d: dict) -> None:
        self.level = d["level"]
        # Allow overriding max_level from the CLI rather than the checkpoint
        self._above_count = d.get("_above_count", 0)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for QuadcopterEnv")

    # --- PPO hyper-parameters ---
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=128,
                   help="Rollout horizon per update")
    p.add_argument("--total-timesteps", type=int, default=500_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--anneal-lr", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--anneal-ent-coef", type=lambda x: x.lower() == "true", default=True)

    # --- Environment ---
    p.add_argument("--max-episode-length", type=int, default=2000)
    p.add_argument("--num-track-points", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.01)

    # --- Rewards ----
    p.add_argument("--progress-reward-scale", type=float, default=1.0)
    p.add_argument("--wp-passed-reward-scale", type=float, default=12.0)
    p.add_argument("--action-smoothness-reward-scale", type=float, default=-0.1)
    p.add_argument("--action-magnitude-reward-scale", type=float, default=-0.2)
    p.add_argument("--ang-vel-reward-scale", type=float, default=-0.1)
    p.add_argument("--orientation-reward-scale", type=float, default=-0.5)
    p.add_argument("--alive-reward-scale", type=float, default=-0.5)
    p.add_argument("--crash-penalty", type=float, default=-10.0)

    # --- Curriculum ---
    p.add_argument("--curriculum-start-level", type=int, default=2,
                   help="Initial curriculum level (0-5)")
    p.add_argument("--curriculum-patience", type=int, default=10,
                   help="Consecutive above-threshold evals to advance")
    p.add_argument("--max-dynamics-delta", type=float, default=0.1,
                   help="Max dynamics randomisation delta (at highest level)")
    p.add_argument("--max-curriculum-level", type=int, default=3, #TODO: set to MAX_CURRICULUM_LEVEL when thresholds are tuned
                   help=f"Stop training once this curriculum level is reached (0-{MAX_CURRICULUM_LEVEL})")

    # --- Infra ---
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--checkpoint-freq", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default=f'checkpoints/{time.strftime("%Y-%m-%d_%H-%M-%S")}')
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--pretrained", type=str, default="",
                   help="Path to a .pt checkpoint to load model weights from "
                        "(transfer learning: fresh optimizer, step count, and curriculum)")
    p.add_argument("--config-path", type=str, default="micro.json")
    p.add_argument("--tensorboard-dir", type=str,
                   default=f'runs/{time.strftime("%Y-%m-%d_%H-%M-%S")}',
                   help="TensorBoard log directory")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    Log.init()
    Log.debug("Starting PPO training with arguments:")
    for k, v in vars(args).items():
        Log.debug(f"  {k}: {v}")
    Log.debug(f"Max curriculum level supported by environment: {MAX_CURRICULUM_LEVEL}")
    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Log.info(f"Using device: {device}")

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve config path
    env_dir = Path(__file__).resolve().parent.parent / "environment"
    config_path = str(env_dir / "quads" / args.config_path)
    Log.info(f"Config path: {config_path}")

    # Curriculum scheduler
    curriculum = CurriculumScheduler(
        start_level=args.curriculum_start_level,
        max_level=args.max_curriculum_level,
        patience=args.curriculum_patience,
        max_dynamics_delta=args.max_dynamics_delta,
    )

    env = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=config_path,
        max_episode_length=args.max_episode_length,
        dt=args.dt,
        progress_reward_scale=args.progress_reward_scale,
        wp_passed_reward_scale=args.wp_passed_reward_scale,
        action_smoothness_reward_scale=args.action_smoothness_reward_scale,
        action_magnitude_reward_scale=args.action_magnitude_reward_scale,
        ang_vel_reward_scale=args.ang_vel_reward_scale,
        orientation_reward_scale=args.orientation_reward_scale,
        alive_reward_scale=args.alive_reward_scale,
        crash_penalty=args.crash_penalty,
        num_track_points=args.num_track_points,
        dynamics_randomization_delta=curriculum.dynamics_delta,
        device=str(device),
    )

    # Read spaces dynamically from the env
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    # Derived sizes
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    Log.info(
        f"obs_dim={obs_dim}  act_dim={act_dim}  batch_size={batch_size}  "
        f"minibatch_size={minibatch_size}  num_updates={num_updates}"
    )

    # Policy
    agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage  (all tensors on device — env already returns GPU tensors)
    obs_buf = torch.zeros(args.num_steps, args.num_envs, obs_dim, device=device)
    actions_buf = torch.zeros(args.num_steps, args.num_envs, act_dim, device=device)
    logprobs_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    rewards_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    dones_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    values_buf = torch.zeros(args.num_steps, args.num_envs, device=device)

    # Resume from checkpoint
    global_step = 0
    start_update = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        start_update = ckpt["update"] + 1
        if "curriculum" in ckpt:
            curriculum.load_state_dict(ckpt["curriculum"])
            # Apply restored curriculum to env
            env.curriculum_level = curriculum.level
            env.dynamics_randomization_delta = curriculum.dynamics_delta
        Log.info(
            f"Resumed from {args.resume}  (global_step={global_step}, "
            f"update={start_update - 1}, curriculum={curriculum.level})"
        )
    elif args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        Log.info(
            f"Loaded pretrained weights from {args.pretrained}  "
            f"(fresh optimizer, global_step=0, curriculum={curriculum.level})"
        )

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(args.tensorboard_dir)
    writer.add_text("hyperparameters",
                    "|".join(f"{k}={v}" for k, v in vars(args).items()))
    Log.info(f"TensorBoard logging to: {args.tensorboard_dir}")

    # Initial reset
    next_obs, _ = env.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    # Rolling episode stats (from env info dict)
    ep_reward_history: deque[float] = deque(maxlen=100)
    ep_length_history: deque[float] = deque(maxlen=100)

    start_time = time.time()

    train_pbar = tqdm(range(start_update, num_updates + 1), desc="Training",
                      initial=start_update - 1, total=num_updates,
                      unit="update", dynamic_ncols=True)
    for update in train_pbar:
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = args.learning_rate * frac
        if args.anneal_ent_coef:
            frac = 1.0 - (update - 1) / num_updates
            ent_coef = args.ent_coef * frac
        else:
            ent_coef = args.ent_coef

        # ---- Rollout phase ----
        for step in tqdm(range(args.num_steps), desc="  Rollout",
                         leave=False, dynamic_ncols=True):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # env.step returns tensors already on device
            next_obs, reward, terminated, truncated, infos = env.step(action)
            rewards_buf[step] = reward
            next_done = (terminated | truncated).float()

            # Collect completed-episode stats from the env info dict.
            # infos is a list with a single aggregated dict.
            if infos and len(infos) > 0:
                info = infos[0]
                ep_rew = info.get("episode_reward_mean", float("nan"))
                ep_len = info.get("episode_length_mean", float("nan"))
                if not math.isnan(ep_rew) and not math.isnan(ep_len) and ep_len > 0:
                    ep_reward_history.append(ep_rew)
                    ep_length_history.append(ep_len)

        # ---- GAE ---- (generalized advantage estimation)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # ---- Flatten batches ----
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ---- PPO update ----
        clipfracs = []
        ppo_pbar = tqdm(total=args.update_epochs * args.num_minibatches,
                        desc="  PPO", leave=False, dynamic_ncols=True)
        for _epoch in range(args.update_epochs):
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                ppo_pbar.update(1)
        ppo_pbar.close()

        # ---- Curriculum update ---- #TODO, right now leveling up too early
        rolling_mean_reward = float(np.mean(ep_reward_history)) if ep_reward_history else float("nan")
        levelled_up = False
        if not math.isnan(rolling_mean_reward):
            levelled_up = curriculum.step(rolling_mean_reward)
            if levelled_up:
                env.curriculum_level = curriculum.level
                env.dynamics_randomization_delta = curriculum.dynamics_delta
                Log.info(
                    f">>> CURRICULUM LEVEL UP -> {curriculum.level}  "
                    f"(dynamics_delta={curriculum.dynamics_delta:.3f})"
                )

        # ---- Logging ----
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed) if elapsed > 0 else 0
        rolling_mean_length = float(np.mean(ep_length_history)) if ep_length_history else float("nan")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Grab latest reward components from env info (last step of rollout)
        info = infos[0] if infos else {}

        # Update main progress bar postfix with key metrics
        train_pbar.set_postfix(
            rew=f"{rolling_mean_reward:.2f}",
            sps=sps,
            lvl=curriculum.level,
            pi=f"{pg_loss.item():.3f}",
            vf=f"{v_loss.item():.3f}",
        )

        if not math.isnan(rolling_mean_reward):
            Log.log_episode_reward_mean(rolling_mean_reward)
        # Log reward components when available
        component_keys = ["mean_progress", "mean_wp_passed", "mean_action_smoothness",
                          "mean_action_magnitude","mean_ang_vel", "mean_orientation", "mean_alive"]
        scaled_component_keys = ["mean_scaled_progress", "mean_scaled_wp_passed", "mean_scaled_action_smoothness",
                                 "mean_scaled_action_magnitude", "mean_scaled_ang_vel", "mean_scaled_orientation", "mean_scaled_alive"]
        # ---- TensorBoard ----
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/curriculum_level", curriculum.level, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if not math.isnan(rolling_mean_reward):
            writer.add_scalar("charts/episode_reward_mean", rolling_mean_reward, global_step)
        if not math.isnan(rolling_mean_length):
            writer.add_scalar("charts/episode_length_mean", rolling_mean_length, global_step)
        for k in component_keys:
            v = info.get(k, float("nan"))
            if not math.isnan(v):
                writer.add_scalar(f"reward_components/{k}", v, global_step)
        for k in scaled_component_keys:
            v = info.get(k, float("nan"))
            if not math.isnan(v):
                writer.add_scalar(f"reward_components_scaled/{k}", v, global_step)

        Log.debug(
            f"update {update:4d} | step {global_step:10d} | SPS {sps:6d} | "
            f"ep_rew {rolling_mean_reward:8.2f} | ep_len {rolling_mean_length:6.1f} | "
            f"pi_loss {pg_loss.item():7.4f} | v_loss {v_loss.item():7.4f} | "
            f"entropy {entropy_loss.item():6.3f} | approx_kl {approx_kl:6.4f} | "
            f"clipfrac {np.mean(clipfracs):5.3f} | expl_var {explained_var:5.3f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | curriculum {curriculum.level}"
        )
        parts = []
        for k in component_keys:
            v = info.get(k, float("nan"))
            short = k.replace("mean_", "")
            parts.append(f"{short}={v:+.4f}")
        if parts:
            Log.debug("  reward components: " + "  ".join(parts))

        # ---- Checkpointing ----
        if update % args.checkpoint_freq == 0 or update == num_updates:
            ckpt_data = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "curriculum": curriculum.state_dict(),
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.checkpoint_dir, f"update_{update:06d}.pt")
            torch.save(ckpt_data, ckpt_path)
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            torch.save(ckpt_data, latest_path)
            Log.info(f"Saved checkpoint: {ckpt_path}")

    train_pbar.close()
    writer.close()
    elapsed = time.time() - start_time
    Log.info(
        f"Training complete. {global_step} steps in {elapsed:.1f}s "
        f"({global_step / elapsed:.0f} SPS)  final curriculum level: {curriculum.level}"
    )
    env.close()


if __name__ == "__main__":
    main()
