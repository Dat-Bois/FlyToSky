---
layout: default
title:  Proposal
---

## Summary
The goal of this project is to develop a robust neural network controller capable of stabilizing a quadrotor and tracking arbitrary 3D trajectories using model-free reinforcement learning. Unlike traditional cascaded PID controllers that separate position and attitude control, this project aims to learn a direct mapping from the system state (pose, velocity, and target error) to the low-level control inputs (four rotor speeds). The system takes the quadrotor's current kinematic state and a desired target state as input, and produces continuous motor commands as output. The ultimate objective is to enable the agent to execute complex maneuvers that require aggressive attitude changes, which are often difficult to tune manually with linear controllers.

## Project Goals
Minimum goal: Successfully train a policy that can stabilize the quadrotor from a random initial orientation and hover at a fixed target point in space without crashing.

Realistic goal: Train an agent capable of tracking a moving target or a pre-defined 3D trajectory (such as a figure-8 or helical path) with low tracking error, outperforming a poorly tuned PID baseline.

Moonshot goal: Generalize the controller to a multi-agent "swarm" scenario where multiple drones utilize the same policy to cooperatively cover or map a defined 3D volume in minimum time.

## AI/ML Algorithms
We anticipate using Deep Reinforcement Learning with continuous control, specifically Proximal Policy Optimization (PPO), due to its stability and ease of use for robotics tasks.

## Evaluation Plan
Quantitative Evaluation: To quantify success, we will execute a set of standard test trajectories (e.g., step response, circle, and spline) and calculate the Root Mean Square Error (RMSE) of the position tracking against the desired path. We will compare these metrics against a standard PID controller baseline (provided by Drake or implemented manually). We hope that the RL approach could achieve a 10-20% improvement in tracking aggressive maneuvers where linear assumptions break down. Additionally, we will measure control effort (sum of squared motor commands) to evaluate the energy efficiency of the learned policy compared to the baseline.

Qualitative Evaluation: We will verify the project's success through 3D visualizations using the Drake visualizer (Meshcat) and log replay tools such as Rerun. We expect to see smooth, stable flight behavior where the drone anticipates turns rather than reacting jerkily. For "sanity checks," We will visualize simple cases: ensuring the drone rights itself when dropped upside down and maintains altitude when pushed. We will also plot the $x, y, z$ trajectories of the agent overlaid on the reference path to visually inspect overshoot and settling time. For the stretch goal, a successful qualitative result would be the swarm agents spreading out to cover the volume rather than clustering in one spot and not colliding.

## AI Tool Usage
We will likely be using AI tools to help generate boilerplate code for setting up the RL training loop, environment wrappers, logging tools, and possibly for hyperparameter tuning scripts. However, the core algorithm implementation, environment design, and evaluation metrics will be developed manually to ensure a deep understanding of the underlying principles.