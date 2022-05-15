# Agent57: Outperforming the human Atari benchmark

This repository cotains the following RL agents:
* DQN
* R2D2
* NGU
* Agent57

## Note

* Agent57 required powerful machines and GPUs to run the Atari games, so it was not possible to run on my Macbook.
* Tested DQN, R2D2 and NGU on basic Atari games like Breakout.
* Reproduced code from a GitHub repo and made a few changes as per requirement.

## Environment properties

* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* openAI Gym    0.23.1
* tensorboard   2.8.0
* numpy         1.22.2

## Installation of Packages on Mac

```
# upgrade pip
python3 -m pip install --upgrade pip setuptools

# install snappy for compress numpy.array on M1 mac
brew install snappy
CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip3 install python-snappy

pip3 install -r requirements.txt
```

## Directory Structure:

* Each directory in the `agents` directory contains the algorithms for each agent:
    - `agent.py` in each directory contains the agent class that includes `reset()` , `step()` methods, and for agent that supports parallel training, we have `Actor` and `Learner` classes for the specific agent.
    - `run_atari.py` uses Conv2d networks to solve Atari games, the default environment_name is set to Pong.
    - `eval_aggent.py` is used for testing agents by using a greedy actor and loading model state from checkpoint file.
* `main_loop.py` contains function to run single thread and parallel training loops.
* `networks` conatins q netwroks used by the agents.
* `tracker.py` is used to accumulate statistics during training and testing/evaluation. Also helps to write log to tensorboard if required.
* `value_learning.py` contains functions to calculate q learning loss.
* `gym_env.py` contains components to standard Atari environment processing.
* `greedy_actors.py` contains all the greedy actors for testing/evaluation like `EpsilonGreedyActor` for DQN agents.
* `replay.py` contains functions and classes relating to experience replay.
* `policy_gradient.py` contains functions to calculate policy gradient loss.

## Training Agents

Default parameters for `run_atari.py` are:
* `num_iterations: 10`
* `num_train_steps: 1000000`
* `num_eval_steps: 100000`

To run the file or to change environment:
```
python3 -m agents.dqn.run_atari

# change environment to Breakout
python3 -m agent.dqn.run_atari --environment_name=Breakout
```

## Testing Agents

To test an agent, you already need to have a checkpoint file created for that agent and environment by running the file `run_atari.py` in any given agent directory. By default, it will record a single episode of agent's self-play at the `recordings` directory.

To run the file:
```
python3 -m agents.dqn.eval_agent
```

## Acknowledgments
 
 This project is based on the work of DeepMind's projects.