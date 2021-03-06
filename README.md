# Asynchronous 1-step Deep Reinforcement Learning

_Work in progress_

In this repo, you'll find two [TensorFlow](https://www.tensorflow.org) implementations from the paper [Asynchronous Deep Reinforcement Learning Methods](https://arxiv.org/abs/1602.01783) by Mnih et., al 2016: **Asynchronous 1-step Q-learning** and **Asynchronous 1-step SARSA**. By default, they run on OpenAIs [Gym enviroment](https://gym.openai.com/), but you can easily play around with other examples through minor edits in ```game_state.py```. 


To get started, simply run ```python asynchronous_1step.py```

<p align="center">
  <img src="resources/networks.png", width="100%"/>
</p>

## Methods
### Asynchronous 1-step Q-learning
In this method, each parallel _worker_ (or _thread_) interacts with its own copy of the enviroment. Each worker computes a gradient of the Q-learning loss at each state, which it accumulates over multiple timesteps before it applies them, making a similar effect to using minibatches. Each worker is given a different exploration rate, which add diversity of the exploration and helps to improve the robustness of the algorithm.

### Asynchonous 1-step SARSA
This method is very similar to 1-step Q-learning, with the exception of using a different target value for ```Q(s,a)```. While Q-learning uses ```r + ɣmaxQ(s',a'; θ')```, 1-step SARSA uses ```r + ɣQ(s',a'; θ')``` where ```a'``` represents the action taken in ```s'```.

### Pseudocode
```
// Algorithm for one worker.
// Assume global shared θ, θ', and the counter global_max = 0.
Initialize worker step counter ĺocal_step ← 0
Initialize target network weights θ' ← θ
Initialize network gradients dθ ← 0
Get initial state s
while global_step > global_max_steps do
    Take action a with ε-greedy policy based on Q(s,a;θ)
    Receive new state s' and reward
    for terminal s' do
        y = reward
    for non-terminal s' do
        for Q-learning do
            y = reward * ɣmaxQ(s',a';θ')
        for SARSA do
            y = reward * ɣQ(s',a';θ')
    Accumulate gradients wrt θ: dθ ← dθ + ∂(y−Q(s,a;θ)) / ∂θ
    s ← s'
    global_step ← global_step + 1 
    local_step ← local_step + 1
    if global_step % target_network_update == 0 then
        Update the target network θ' ← θ
    end if
    if local_step % local_max_steps == 0 or s' is terminal then
        Perform asynchronous update of θ using dθ.
        Clear gradients dθ ← 0.
    end if
```

#### General settings
* ```game``` - ```Breakout-v0``` - Name of the Atari game to play. Full list [here](https://gym.openai.com/envs/).
* ```histrogram_summary```- ```500``` - How many episodes to average histogram summary over.
* ```load_checkpoint``` - ```True``` - If it should should from available checkpoints.
* ```save_checkpoint``` - ```True``` - If it should should save checkpoints when break is triggered.
* ```save_stats``` - ```True``` If it should save stats for Tensorboard.
* ```random_seed``` - ```123``` - Sets the random seed.
* ```use_gpu``` - ```False``` - If TensorFlow operations should run on GPU rather than CPU.
* ```display``` - ```False``` - If it you want to render the game.
* ```log``` - ```False``` - For a verbose log.


#### Training settings
* ```parallel_agents``` - ```8``` - Number of asynchronous agents (threads) to train with.
* ```global_max_steps``` - ```80 000 000``` - Maximum training steps.
* ```local_max_steps``` - ```5``` - Frequency with which each agent network is updated (```I_target```).
* ```target_network_update``` - ```10 000``` - Frequency with which the shared target network is updated (```I_AsyncUpdate```).
* ```frame_skip``` -  ```3``` - How many frames to skip (or actions to repeat) for each step.


#### Method settings
* ```method``` - ```q``` - Training algorithm to use ```[q, sarsa]```. Defaults to Q-learning.
* ```gamma``` - ```0.99``` - Discount factor for rewards.
* ```epsilon_anneal``` - ```1 000 000``` - Number of steps to anneal epsilon.

#### Optimizer settings
* ```optimizer``` - ```rmsprop``` - Which optimizer to use ```[adam, gradientdescent, rmsprop]```. Defaults to ```rmsprop```.
* ```rms_decay``` - ```0.99``` - RMSProp decay parameter.
* ```rms_epsilon``` - ```0.1``` RMSProp epsilon parameter.
* ```learning_rate``` - ```0.0007``` - Initial learning rate.
* ```anneal_learning_rate``` - ```True``` - If learning rate should be annealed over global max steps.

#### Evaluation settings
* ```evaluate``` - ```True``` - If it should run continous evaluation throughout the training session.
* ```evaluation_episodes``` - ```10``` - How many evaluation episodes to run (and average the evaluation over).
* ```evaluation_frequency``` - ```100 000``` - The frequency of evaluation runs.