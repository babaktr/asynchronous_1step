# Asynchronous 1-step Deep Reinforcement Learning

_Work in progress_

In this repo, you'll find two [TensorFlow](https://www.tensorflow.org) implementations from the paper [Asynchronous Deep Reinforcement Learning Methods](https://arxiv.org/abs/1602.01783) by Mnih et., al 2016: **Asynchronous 1-step Q-learning** and **Asynchronous 1-step SARSA**. By default, they run on OpenAIs [Gym enviroment](https://gym.openai.com/), but you can easily play around with other examples through minor edits in ```game_state.py```. 


Run using ```python asynchronous_1step.py```

## Methods
### Asynchronous 1-step Q-learning
In this method, each parallel _worker_ (or _thread_) interacts with its own copy of the enviroment. Each worker computes a gradient of the Q-learning loss at each state, which it accumulates over multiple timesteps before it applies them, making a similar effect to using minibatches. Each worker is given a different exploration rate, which add diversity of the exploration and helps to improve the robustness of the algorithm.

### Asynchonous 1-step SARSA
This method is very similar to 1-step Q-learning, with the exception of using a different target value for ```Q(s,a)```. While Q-learning uses ```r + ɣmaxQ(s',a'; θ')```, 1-step SARSA uses ```r + ɣQ(s',a'; θ')``` where ```a'``` represents the action taken in ```s```.

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
    if local_step % local_max_steps == 0 or s is terminal then
        Perform asynchronous update of θ using dθ.
        Clear gradients dθ ← 0.
    end if
```

#### General settings
* ```game``` - ```Breakout-v0``` - Name of the Atari game to play. Full list [here](https://gym.openai.com/envs/).
* ```use_gpu``` - ```False``` - If TensorFlow operations should run on GPU rather than CPU.
* ```average_summary```- ```20``` - How many episodes to average histogram summary over.
* ```log``` - ```False``` - If log level should be verbose.
* ```random_seed``` - ```123``` - Sets the random seed.

#### Training settings
* ```parallel_agents``` - ```8``` - Number of asynchronous agents (threads) to train with.
* ```global_max_steps``` - ```80 000 000``` - Maximum training steps.
* ```local_max_steps``` - ```5``` - Frequency with which each agent network is updated (I_target).
* ```target_network_update``` - ```40 000``` - Frequency with which the shared target network is updated (I_AsyncUpdate).
* ```frame_skip``` - ```0``` - How many frames to skip on each step.
* ```no_op_max``` -  ```0``` - How many no-op actions to take at the beginning of each episode.

#### Method settings
* ```method``` - ```q``` - Training algorithm to use [q, sarsa]. Defaults to Q-learning.
* ```gamma``` - ```0.99``` - Discount factor for rewards.
* ```epsilon_anneal``` - ```4 000 000``` - Number of steps to anneal epsilon.

#### Optimizer settings
* ```optimizer``` - ```rmsprop``` - Which optimizer to use ```[adam, gradientdescent, rmsprop]```. Defaults to ```rmsprop```.
* ```rms_decay``` - ```0.99``` - RMSProp decay parameter.
* ```learning_rate``` - ```0.0001``` - Initial learning rate.

#### Testing settings
**Not yet used**
* ```evaluate_model``` - ```False``` - It model should run through OpenAIs Gym evaluation.
* ```display``` - ```False``` - If it should display the agent.
* ```test_runs``` - ```100```- Number of times to run the evaluation.
* ```test_epsilon``` - ```0.0``` - Epsilon to use on test run.
