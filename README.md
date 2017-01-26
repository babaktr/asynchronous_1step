# Asynchronous 1-step Deep Reinforcement Learning

_Work in progress_

In this repo, you'll find two implementations from the paper _Asynchronous Deep Reinforcement Learning Methods_ by Mnih et., al 2016: **Asynchronous 1-step Q-learning** and **Asynchronous 1-step SARSA**.


Run using ```python asynchronous_1step.py```

## Methods
### Asynchronous 1-step Q-learning
In this method, each parallel _worker_ (or _thread_) interacts with its own copy of the enviroment. Each worker computes a gradient of the Q-learning loss at each state, which it accumulates over multiple timesteps before it applies them, making a similar effect to using minibatches. Each worker is given a different exploration rate, which add diversity of the exploration and helps to improve the robustness of the algorithm.

### Asynchonous 1-step SARSA
This method is very similar to 1-step Q-learning, with the exception of using a different target value for ```Q(s,a)```. While Q-learning uses ```r + γmaxQ(s',a'; θ')```, 1-step SARSA uses ```r + γQ(s',a'; θ')``` where ```a'``` represents the action taken in ```s```.

### Pseudocode
```
# Algorithm for one worker.
# Assume global shared θ, θ', and counter global_max = 0.
Initialize worker step counter ĺocal_step ← 0
Initialize target network weights θ' ← θ
Initialize network gradients dθ ← 0
Get initial state s
repeat
    Take action a with ε-greedy policy based on Q(s,a;θ)
    Receive new state s' and reward r
    for terminal s' do
        y = r
    for non-terminal s' do
        for Q-learning do
            y = r * γmaxQ(s',a';θ')
        for SARSA do
            y = r * γQ(s',a';θ')
    Accumulate gradients wrt θ: dθ ← dθ + ∂(y−Q(s,a;θ)) / ∂θ
    s = s'
    global_step ← global_step + 1 
    local_step ← local_step + 1
    if global_step % target_network_update == 0 then
        Update the target network θ' ← θ
    end if
    if local_step % local_max_steps == 0 or s is terminal then
        Perform asynchronous update of θ using dθ.
        Clear gradients dθ ← 0.
    end if
    until global_step > global_max_steps
```

## General settings
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
* ```test_runs``` - ```100```- Number of times to run the evaluation.
* ```test_epsilon``` - ```0.0``` - Epsilon to use on test run.
