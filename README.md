# Asynchronous 1-step Deep Reinforcement Learning Methods

_Work in progress_

In this repo, you'll find two implementations from the paper _Asynchronous Deep Reinforcement Learning Methods_ by Mnih et., al 2016


Run using ```python asynchronous_1step.py```

## Parameters

### General settings
* ```game``` - ```Breakout-v0``` - Name of the atari game to play. Full list [here](https://gym.openai.com/envs/).
* ```use_gpu``` - ```False``` - If TensorFlow operations should run on GPU rather than CPU.
* ```random_seed``` - ```123``` - Sets the random seed.
* ```log``` - ```False``` - If log level should be verbose.

#### Training settings
* ```parallel_agents``` - ```8``` - Number of asynchronous agents (threads) to train with.
* ```global_max_steps``` - ```80 000 000``` - Maximum training steps.
* ```local_max_steps``` - ```5``` - Frequency with which each agent network is updated (I_target).
* ```target_network_update``` - ```40 000``` - Frequency with which the shared target network is updated (I_AsyncUpdate).
* ```frame_skip``` - ```0``` - How many frames to skip on each step.
* ```no_op_max``` -  ```0``` - 

#### Method settings
* ```method``` - ```q``` - Training algorithm to use [q, sarsa]. Defaults to Q-learning.
* ```gamma``` - ```0.99``` - Discount factor for rewards.
* ```epsilon_anneal``` - ```4 000 000``` - Number of steps to anneal epsilon.

#### Optimizer settings
* ```learning_rate``` - ```0.0001``` - Initial learning rate.
* ```optimizer``` - ```rmsprop``` - If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to RMSProp.
* ```rms_decay``` - ```0.99``` - RMSProp decay parameter.

#### Testing settings
* ```evaluate_model``` - ```False``` - It model should run through OpenAIs Gym evaluation.
* ```display``` - ```False``` - If it should display the agent.
*```test_runs``` - ```100```- Number of times to run the evaluation.
*```test_epsilon``` - ```0.0``` - Epsilon to use on test run.
