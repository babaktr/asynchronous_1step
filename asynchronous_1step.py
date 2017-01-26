import numpy as np
import tensorflow as tf

from network import DeepQNetwork
from game_state import GameState
from stats import Stats

from threading import Thread

import time 

flags = tf.app.flags

# General settings
flags.DEFINE_string('game', 'Breakout-v0', 'Name of the Atari game to play. Full list: https://gym.openai.com/envs/')
flags.DEFINE_boolean('use_gpu', False, 'If it should run on GPU rather than CPU.')
flags.DEFINE_integer('histogram_summary', 20, 'How many episodes to plot histogram summary over.')
flags.DEFINE_boolean('log', False, 'If log level should be verbose.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')

# Training settings
flags.DEFINE_integer('parallel_agents', 8, 'Number of asynchronous agents (threads) to train with.')
flags.DEFINE_integer('global_max_steps', 80000000, 'Maximum training steps.')
flags.DEFINE_integer('local_max_steps', 5, 'Frequency with which each agent network is updated (I_target).')
flags.DEFINE_integer('target_network_update', 40000, 'Frequency with which the shared target network is updated (I_AsyncUpdate).')
flags.DEFINE_integer('frame_skip', 0, 'How many frames to skip on each step.')
flags.DEFINE_integer('no_op_max', 0, 'How many no-op actions to take at the beginning of each episode.')

# Method settings
flags.DEFINE_string('method', 'q', 'Training algorithm to use [q, sarsa].')
flags.DEFINE_float('gamma', 0.99, 'Discount factor for rewards.')
flags.DEFINE_integer('epsilon_anneal', 400000, 'Number of steps to anneal epsilon.')

# Optimizer settings
flags.DEFINE_string('optimizer', 'rmsprop', 'Which optimizer to use [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')
flags.DEFINE_float('rms_decay', '0.99', 'RMSProp decay parameter.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

# Testing settings
# ---- Not yet used ----
flags.DEFINE_boolean('evaluate_model', False, 'It model should run through OpenAIs Gym evaluation.')
flags.DEFINE_boolean('display', False, 'If it should display the agent.')
flags.DEFINE_integer('test_runs', 100, 'Number of times to run the evaluation.')
flags.DEFINE_float('test_epsilon', 0.0, 'Epsilon to use on test run.')

settings = flags.FLAGS

'''
Sample final epsilon as paper by Mnih et. al. 2016.
'''
def sample_final_epsilon():
    final_epsilons_array = np.array([0.5, 0.1, 0.01])
    probabilities = np.array([0.3, 0.4, 0.3])
    return np.random.choice(final_epsilons_array, 1, p=list(probabilities))[0]

'''
Anneal epsilon value.
'''
def anneal_epsilon(epsilon, final_epsilon):
    if epsilon > final_epsilon:
        epsilon -= (1.0 - final_epsilon) / settings.epsilon_anneal
    else:
        epsilon = final_epsilon
    return epsilon

'''
Select action according to exploration epsilon.
'''
def select_action(epsilon, q_values, action_size):
    if np.random.random() < epsilon:
        return np.argmax(q_values)
    else:
        return np.random.randint(0, action_size)

'''
Create one-hot vector from action.
'''
def onehot_vector(action, action_size):
    vector = np.zeros(action_size)
    vector[action] = 1
    return vector

'''
Return empty arrays to reset gradients.
'''
def reset_gradient_arrays():
    return [], [], []

'''
Return empty arrays to reset stats.
'''
def reset_stat_arrays():
    return [], [], [], [], [], []

'''
Vertically stack batches to match network structure
'''
def stack_batches(state_batch, action_batch, y_batch):
    return np.vstack(state_batch), np.vstack(action_batch), np.vstack(y_batch)

'''
Worker thread that runs an agent training in a local game enviroment with its own local network.
'''
def worker_thread(thread_index, local_network, local_game_state): #sess, summary_writer, summary_op, score_input):
    global global_max_steps, global_step, target_network, sess, stats

    # Set worker's initial and final epsilons
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    # Prepare gradiets
    y_batch, state_batch, action_batch = reset_gradient_arrays()

    # Prepare stats
    action_arr, q_max_arr, reward_arr, epsilon_arr, loss_arr, acc_arr = reset_stat_arrays()

    time.sleep(1*thread_index)
    print("Starting agent " + str(thread_index) + " with final epsilon: " + str(final_epsilon))

    local_step = 0
    while global_step < global_max_steps:
        # Reset counters and values
        local_step = 0
        terminal = False

        # Get initial game observation
        state = local_game_state.reset()

        while not terminal:
            # Get the Q-values of the current state
            q_values = local_network.predict(sess, [state])

            # Anneal epsilon and select action
            epsilon = anneal_epsilon(epsilon, final_epsilon)
            action = select_action(epsilon, q_values, local_game_state.action_size)
            action = np.argmax(q_values)

            # Make action an observe 
            new_state, reward, terminal = local_game_state.step(action)
            
            # Get the new state's Q-values
            q_values_new = target_network.predict(sess, [new_state])

            if settings.method.lower() == 'sarsa':
                # Get Q(s',a') for selected action to update Q(s,a)
                q_value_new = q_values_new[action]
            else:
                # Get max(Q(s',a')) to update Q(s,a)
                q_value_new = np.max(q_values_new)

            if not terminal: 
                # Q-learning: update with reward + gamma * max(Q(s',a')
                # SARSA: update with reward + gamma * Q(s',a') for the action taken in s'
                update = reward + (settings.gamma * q_value_new)
            else:
                # Terminal state, update using reward
                update = reward

            # Fill batch
            y_batch.append([update])
            state_batch.append([state])
            action_batch.append(onehot_vector(action, local_game_state.action_size))

            # Update counters and values
            local_step += 1
            global_step += 1

            # Save for stats
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))
            action_arr.append(action)

            # Update target network on I_target
            if global_step % settings.target_network_update == 0:
                print 'Updated target network on step: {}'.format(global_step)
                sess.run(target_network.sync_from(local_network))

            # Update online network on I_AsyncUpdate
            if local_step % settings.local_max_steps == 0 or terminal:
                state_batch, action_batch, y_batch = stack_batches(state_batch, action_batch, y_batch)
                # Measure accuracy of the network
                acc = local_network.get_accuracy(sess, state_batch, y_batch)
                # Train local network with gradient batches
                loss = local_network.train(sess, state_batch, action_batch, y_batch)

                # Save values for stats
                epsilon_arr.append(epsilon)
                loss_arr.append(loss)
                acc_arr.append(acc)

                # Clear gradients
                y_batch, state_batch, action_batch = reset_gradient_arrays()
      
            if terminal:
                print 'Thread: {}, Global step: {}, Episode steps: {}, Reward: {}, Qmax: {}, Loss: {}, Accuracy: {}, Epsilon: {}'.format(thread_index, 
                    global_step, local_step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'),  format(np.average(loss_arr), '.4f'), 
                    format(np.average(acc_arr), '.3f'), format(np.average(epsilon_arr), '.2f'))

                # Update stats
                stats.update({'loss': np.average(loss_arr), 
                            'accuracy': np.average(acc_arr),
                            'qmax': np.average(q_max_arr),
                            'epsilon': np.average(epsilon_arr),
                            'episode_actions': action_arr,
                            'reward': np.sum(reward_arr),
                            'steps': local_step,
                            'step': global_step
                            }) 

                # Reset stats
                action_arr, q_max_arr, reward_arr, epsilon_arr, loss_arr, acc_arr = reset_stat_arrays()


            else:
                # Update current state from s_t to s_t1
                state = new_state
                local_game_state.update_state()

global_max_steps = settings.global_max_steps
global_step = 0

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

# Prepare game environments
local_game_states = []
for n in range(settings.parallel_agents):
    local_game_state = GameState(settings.random_seed + n, 
                                settings.log, 
                                settings.game, 
                                settings.frame_skip, 
                                settings.display, 
                                settings.no_op_max)
    local_game_states.append(local_game_state)

# Prepare local networks and game enviroments
local_networks = []
game = local_game_states[0]
for n in range(settings.parallel_agents):
    name = 'local_network_' + str(n)
    local_network = DeepQNetwork(n, 
                                name,
                                device, 
                                settings.random_seed + n, 
                                game.action_size, 
                                settings.learning_rate, 
                                settings.optimizer,
                                settings.rms_decay)
    local_networks.append(local_network)

# Set target Deep Q Network
target_network = DeepQNetwork(-1, 
                            'target_network',
                            device, 
                            settings.random_seed, 
                            game.action_size,
                            settings.learning_rate, 
                            settings.optimizer,
                            settings.rms_decay)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

# Statistics summary writer
summary_dir = './logs/asynchronous-1step-{}_game-{}_parallel_agents-{}_global-max-{}_frame-skip{}/'.format(settings.method, 
    settings.game, settings.parallel_agents, settings.global_max_steps, settings.frame_skip)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, settings.histogram_summary)

init = tf.global_variables_initializer()
sess.run(init)

# Prepare parallel workers
workers = []
for n in range(settings.parallel_agents):
    worker = Thread(target=worker_thread,
                    args=(n, local_networks[n], local_game_states[n]))
    workers.append(worker)

# Start and join workers to start training
for t in workers:
    t.start()
for t in workers:
    t.join()