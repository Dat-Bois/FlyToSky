"""CleanRL-style PPO training loop for the FlyToSky QuadcopterEnv."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

log = logging.getLogger("flytosky.train")


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 19, act_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for QuadcopterEnv")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=50_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--anneal-lr", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--checkpoint-freq", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--config-path", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve config path
    if args.config_path:
        config_path = args.config_path
    else:
        env_dir = Path(__file__).resolve().parent.parent / "environment"
        config_path = str(env_dir / "quad_params.json")
    log.info("Config path: %s", config_path)

    # Create environment
    from flytosky.environment.quadcopter_env import QuadcopterEnv

    env = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=config_path,
        device=str(device),
    )

    # Derived sizes
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    log.info(
        "batch_size=%d  minibatch_size=%d  num_updates=%d",
        batch_size, minibatch_size, num_updates,
    )

    # Policy
    agent = ActorCritic(
        obs_dim=env.single_observation_space.shape[0],
        act_dim=env.single_action_space.shape[0],
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage
    obs_buf = torch.zeros(args.num_steps, args.num_envs, *env.single_observation_space.shape, device=device)
    actions_buf = torch.zeros(args.num_steps, args.num_envs, *env.single_action_space.shape, device=device)
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
        log.info("Resumed from %s  (global_step=%d, update=%d)", args.resume, global_step, start_update - 1)

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initial reset
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Episode tracking
    ep_rewards: list[float] = []
    ep_lengths: list[float] = []

    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = args.learning_rate * frac

        # ---- Rollout phase ----
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, terminated, truncated, infos = env.step(action)
            next_obs = next_obs.to(device)
            rewards_buf[step] = reward.to(device)
            next_done = (terminated | truncated).float().to(device)

            # Track completed episodes from env info
            if infos and isinstance(infos, list) and len(infos) > 0:
                info = infos[0]
                if "episode_reward_mean" in info and info.get("episode_length_mean", 0) > 0:
                    ep_rewards.append(info["episode_reward_mean"])
                    ep_lengths.append(info["episode_length_mean"])

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf, device=device)
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
        b_obs = obs_buf.reshape(-1, *env.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, *env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ---- PPO update ----
        clipfracs = []
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

        # ---- Logging ----
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed) if elapsed > 0 else 0
        mean_ep_reward = np.mean(ep_rewards[-50:]) if ep_rewards else float("nan")
        mean_ep_length = np.mean(ep_lengths[-50:]) if ep_lengths else float("nan")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log.info(
            "update %4d | step %10d | SPS %6d | ep_rew %8.2f | ep_len %6.1f | "
            "pi_loss %7.4f | v_loss %7.4f | entropy %6.3f | approx_kl %6.4f | "
            "clipfrac %5.3f | explained_var %5.3f | lr %.2e",
            update,
            global_step,
            sps,
            mean_ep_reward,
            mean_ep_length,
            pg_loss.item(),
            v_loss.item(),
            entropy_loss.item(),
            approx_kl,
            np.mean(clipfracs),
            explained_var,
            optimizer.param_groups[0]["lr"],
        )

        # ---- Checkpointing ----
        if update % args.checkpoint_freq == 0 or update == num_updates:
            ckpt_data = {
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.checkpoint_dir, f"update_{update:06d}.pt")
            torch.save(ckpt_data, ckpt_path)
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            torch.save(ckpt_data, latest_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    elapsed = time.time() - start_time
    log.info("Training complete. %d steps in %.1fs (%.0f SPS)", global_step, elapsed, global_step / elapsed)
    env.close()


if __name__ == "__main__":
    main()
