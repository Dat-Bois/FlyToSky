# from __future__ import annotations

# import json
# import torch
# import numpy as np
# import gymnasium as gym
# from pathlib import Path

# from typing import Optional, Tuple, Dict, Any

# from flytosky.environment.math_utils import *
# from flytosky.environment.track_gen import VectorizedTrackGenerator, TrackSettings

import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

import gymnasium as gym
import pufferlib #WHAT THE ACTUAL FUCK IS THIS??????????????????
# for some reason pufferlib has to be imported AFTER gymnasium for logging not to break??

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from flytosky.logging import Log



class ComponentB:
    """Simulates a separate component that logs through its own caller identity."""

    def __init__(self, name: str) -> None:
        self.name = name
        Log.info(f"[{self.name}] ComponentB initialised")

    async def run(self, iterations: int = 5, delay: float = 0.6) -> None:
        """Log messages at various levels with a short delay between each."""
        for i in range(iterations):
            Log.debug(f"[{self.name}] tick {i} — debug detail")
            Log.info(f"[{self.name}] tick {i} — processing data")
            if i == 2:
                Log.warning(f"[{self.name}] tick {i} — simulated warning")
            if i == 4:
                Log.error(f"[{self.name}] tick {i} — simulated error")
            await asyncio.sleep(delay)

        Log.info(f"[{self.name}] finished all iterations")


class ComponentBPuffer(pufferlib.PufferEnv):
    """A minimal PufferEnv that exercises the Log system during reset/step,
    mirroring the pattern used in QuadcopterEnv."""

    def __init__(
        self,
        name: str = "PufferB",
        num_envs: int = 1,
        max_steps: int = 10,
        **kwargs,
    ):
        self.name = name
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.max_steps = max_steps

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        Log.info(f"[{self.name}] ComponentBPuffer init — {num_envs} envs, max_steps={max_steps}")
        super().__init__()

        self._step_count = np.zeros(num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # PufferEnv interface
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, dict]:
        Log.info(f"[{self.name}] reset (seed={seed})")
        if seed is not None:
            np.random.seed(seed)

        self._step_count[:] = 0
        obs = np.random.randn(self.num_envs, 4).astype(np.float32)
        Log.debug(f"[{self.name}] reset obs sample: {obs[0]}")
        return obs, [{}]

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        self._step_count += 1
        obs = np.random.randn(self.num_envs, 4).astype(np.float32)
        rewards = np.random.randn(self.num_envs).astype(np.float32)
        dones = self._step_count >= self.max_steps
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        Log.debug(f"[{self.name}] step {self._step_count[0]} — reward {rewards[0]:.3f}")
        if self._step_count[0] == self.max_steps // 2:
            Log.warning(f"[{self.name}] halfway done (step {self._step_count[0]})")
        if dones.any():
            Log.info(f"[{self.name}] episode(s) finished at step {self._step_count[0]}")

        return obs, rewards, dones, truncated, infos

    def close(self):
        Log.info(f"[{self.name}] close")

    # ------------------------------------------------------------------
    # Async helper for the interleaved logging test
    # ------------------------------------------------------------------

    async def run_episode(self, delay: float = 0.3) -> None:
        """Reset → step until done, yielding between steps for async interleaving."""
        self.reset(seed=42)
        done = False
        while not done:
            action = np.random.uniform(-1, 1, (self.num_envs, 2)).astype(np.float32)
            _obs, _rew, dones, _trunc, _info = self.step(action)
            done = dones.all()
            await asyncio.sleep(delay)
