# cs138-proj

# Curriculum Learning in Galaxian with PPO

This project implements a curriculum learning environment for the Atari game Galaxian using Proximal Policy Optimization (PPO). The agent is trained using either standard PPO or a custom curriculum strategy that replays past game states with softmax-based sampling and optional Dirichlet noise for exploration.

---

## Features

- Custom Gymnasium environment (`CurriculumLearningEnv`) that enables curriculum-based replay of previously seen states.
- Options for softmax sampling and Dirichlet noise to guide the curriculum.
- Custom CNN feature extractor for image-based Atari observations.
- Logging and evaluation with TensorBoard.
- Plotting scripts to visualize evaluation performance.

---

## Installation

This is built with stable-baselines3 and Gymnasium. This has been tested with Python

```sh
pip install -r requirements.txt
```

## How to Use

The `agent.py` script is the main entry point for training the agent. You may specify whether to use curriculum learning or standard PPO training. The logs directory for the Tensorboard logs is set to the environment variable `LOG_DIR`, and the checkpoints directory is set to the environment variable `CHECKPOINT_DIR`. Slurm scripts `learn-curriculum.sh` and `learn-base.sh` for training with and without curriculum learning, respectively, are provided for convenience.

### Train a PPO Agent
```
python agent.py train curriculum
```

### Train a Curriculum Agent
```
python agent.py train
```





