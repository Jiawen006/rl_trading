# Automated Trading in Financial Markets

This project developed an ensemble approach that consist of three reinforcement learning agents (A2C, PPO, DDPG). The result shows that this ensemble approach provides more robust result than any single reinforcement learning agents.

<p align="center">
<img src = "Flow_diagram.jpeg" width=80% />
</p>

**Key features:**
+ Develops a daily trading environment for ten anonymized financial time series.
+ Supports order execution for both market order and limit order.
+ Supports `Stable-Baseline3` algorithms including A2C, PPO and DDPG.
+ Visualizes the backtesting result based on the ensemble approach.
+ Provides analysis tools for model evaluation.

## Installation

```bash
# In the directory that you want to try the project
git clone git@github.com:Jiawen006/DRL-Trading-Draft.git
```

Contents in the requirement.txt

```
numpy==1.24.3
pandas==2.1.1
stable_baselines3==2.1.0
stockstats==0.6.2
```

## Quick Start

You can directly start to train your own model using the default parameters. 

```bash
python3 main.py
```

After training the model, you can start to evaluate the model in the testing dataset and then visualize the equity curve.

```bash
python3 evaluate.py
```

## Arguments

The default configurations are specified in [config.py](config/config.py)

+ `device`: The device (gpu/cpu) that we want to use. Default: `'cuda:0'`
+ `a2ctimesteps`: The training period of the A2C agent. Default: `'3000'`
+ `ppotimesteps`: The training period of the A2C agent. Default: `'1000000'`
+ `ddpgtimestep`: The training period of the A2C agent. Default: `'10000'`

+ `A2C_PARAMS`
    + [`'n_steps'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`: Number of steps used in A2C algorithm. Deafult: 10
    + [`'ent_coef'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`:  Entropy coefficient for the loss calculation. Default: 0.001
    + [`'verbose'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`: 0 (By deafult)
    + [`'learning_rate'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`:  The learning rate. Default: 1e-4

+ `PPO_PARAMS`
    + [`'n_steps'`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)`: The number of steps to run for each environment per update . Deafult: 2048
    + [`'ent_coef'`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)`:  Entropy coefficient for the loss calculation. Default: 1e-2
    + [`'batch_size'`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)`: Minibatch size. Deafult: 64
    + [`'learning_rate'`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)`:  The learning rate. Default: 2.5e-5

+ `DDPG_PARAMS`
    + [`'buffer_size'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`: Minibatch size for each gradient update. Deafult: 128
    + [`'ent_coef'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`:  Entropy coefficient for the loss calculation. Default: 1e-2
    + [`'gamma'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`: The discount factor. Deafult: 0.99
    + [`'learning_rate'`](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)`:  The learning rate. Default: 1e-3
    + [`'gradient_steps'`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)`:  Number of graident steps to do after each rollout. Default: -1



## To do list

- [ ] Try to use online dataset
- [ ] Try to build more variety with the system, for example supporting more RL agents or customize agents in ensemble strategy
- [ ] Build analysis tools for the model
- [ ] Complete the arguments
- [ ] Write a pretty flow chart with GIF