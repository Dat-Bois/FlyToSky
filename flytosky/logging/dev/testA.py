"""
testA - entry point for the async logging integration test.

Creates the shared Log session, instantiates ComponentB from testB,
and runs both A and B coroutines concurrently so their log messages
interleave across separate per-file loggers and a shared Rerun session.
"""
# from __future__ import annotations

# import argparse
# import math
# import os
# import time
# from collections import deque
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Normal

import asyncio
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from flytosky.logging import Log
from testB import ComponentB, ComponentBPuffer
from flytosky.environment.quadcopter_env import QuadcopterEnv  # For reference logging pattern


async def component_a_loop(iterations: int = 5, delay: float = 0.4) -> None:
    """Log messages from the 'testA' caller context."""
    for i in range(iterations):
        Log.info(f"[A] step {i} — doing work")
        Log.debug(f"[A] step {i} — verbose detail")
        if i == 1:
            Log.warning(f"[A] step {i} — something looks off")
        if i == 3:
            Log.error(f"[A] step {i} — simulated failure")
        await asyncio.sleep(delay)

    Log.info("[A] all steps complete")


async def main() -> None:
    # Initialise the shared logging / Rerun session once
    Log.init("test_logging")
    Log.info("[A] Logger initialised — starting test")

    # ComponentB gets its own per-file logger automatically
    # comp_b = ComponentB("B1")

    # ComponentBPuffer acts like a PufferEnv (similar to QuadcopterEnv)
    puffer_env = ComponentBPuffer(name="PufB", num_envs=2, max_steps=8)

    # Run all three loops concurrently so messages interleave
    await asyncio.gather(
        component_a_loop(iterations=5, delay=0.4),
        # comp_b.run(iterations=5, delay=0.6),
        puffer_env.run_episode(delay=0.35),
    )

    Log.info("[A] Test finished — check Logs/ for .rrd and .log files")


if __name__ == "__main__":
    asyncio.run(main())
