import numpy as np
import tensorflow as tf

from network import DeepQNetwork
from game_state import GameState
from stats import Stats

from threading import Thread

import time 

flags = tf.app.flags

# General settings
flags.DEFINE_string('game', 'Breakout-v0', 'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
flags.DEFINE_boolean('use_gpu', False, 'If it should run on GPU rather than CPU.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_boolean('log', False, 'If log level should be verbose.')

# Training settings
flags.DEFINE_integer('parallel_agents', 8, 'Number of asynchronous agents (threads) to train with.')
flags.DEFINE_integer('global_max_steps', 80000000, 'Maximum training steps')
flags.DEFINE_integer('local_max_steps', 5, 'Frequency with which each agent network is updated (I_target).')
flags.DEFINE_integer('target_network_update', 40000, 'Frequency with which the shared target network is updated (I_AsyncUpdate)')
# ---- Not yet used ----
flags.DEFINE_integer('frame_skip', 0, 'Beskr frame skip.')
flags.DEFINE_integer('no_op_max', 0, 'Beskr noop.')

# Method settings
flags.DEFINE_string('method', 'q', 'Training algorithm to use [q, sarsa].')
flags.DEFINE_float('gamma', 0.99, 'Discount factor for rewards')
flags.DEFINE_integer('epsilon_anneal', 400000, 'Number of steps to anneal epsilon.')

# Optimizer settings
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to gradient descent.')

# Testing settings
flags.DEFINE_boolean('evaluate_model', False, 'It model should run through OpenAIs Gym evaluation.')
flags.DEFINE_boolean('display', False, 'If it should display the agent.')

flags.DEFINE_integer('test_runs', 100, 'Number of times to run the evaluation.')
flags.DEFINE_float('test_epsilon', 0.0, 'Epsilon to use on test run.')

settings = flags.FLAGS

global_max_steps = settings.global_max_steps
global_step = 0

target_network = None

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


def worker_thread(thread_index, device, stats): #sess, summary_writer, summary_op, score_input):
    global global_max_steps, global_step

    # Load local game enviroments
    local_game_state = GameState(settings.random_seed + thread_index, 
                                settings.log, 
                                settings.game, 
                                settings.frame_skip, 
                                settings.display, 
                                settings.no_op_max)
    
    # Create local network
    local_network = DeepQNetwork(thread_index, 
                                device, 
                                settings.random_seed + thread_index, 
                                local_game_state.action_size, 
                                settings.learning_rate, 
                                settings.optimizer)

    # Set worker's initial and final epsilons
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Starting agent " + str(thread_index) + " with final epsilon: " + str(final_epsilon))

    # Prepare gradiets
    y_batch = []
    state_batch = []
    action_batch = [] 

    # Prepare stats
    q_max_arr = []
    reward_arr = []
    epsilon_arr = []
    loss_arr = []
    acc_arr = []

    time.sleep(3*thread_index)
    local_step = 0

    while global_step < global_max_steps:
        # Reset counters and values
        local_step = 0
        episode_reward = 0
        average_q_max = 0

        terminal = False

        # Get initial game observation
        state = local_game_state.reset()

        while not terminal:
            # Get the Q-values of the current state
            q_values = local_network.predict([state])

            # Anneal epsilon
            epsilon = anneal_epsilon(epsilon, final_epsilon)

            # Select action
            action = select_action(epsilon, q_values, local_game_state.action_size)

            # Make action an observe 
            new_state, reward, terminal = local_game_state.step(action)
            
            # Get the new state's Q-values
            q_values_new = local_network.predict([new_state])

            if settings.method == 'sarsa':
                # Get Q(s',a') for selected action a to update Q(s,a)
                q_value_new = q_values_new[action]
            else:
                # Get max(Q(s',a')) to update Q(s,a)
                q_value_new = np.max(q_values_new)

            if not terminal: 
                # Non-terminal state, update with reward + gamma * max(Q(s'a')
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
            episode_reward += reward
            average_q_max += np.max(q_values)

            # Save for stats
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))

            # Update target network on I_target
            if global_step % settings.target_network_update == 0:
                print global_step
                session.run(reset_target_network_params)
                target_network.sync_from(local_network)

            # Update online network on I_AsyncUpdate
            if local_step % settings.local_max_steps == 0 or terminal:
                # Stack batches
                state_batch = np.vstack(state_batch)
                action_batch = np.vstack(action_batch)
                y_batch = np.vstack(y_batch)

                acc = local_network.get_accuracy(state_batch, y_batch)

                loss = local_network.train(state_batch, action_batch, y_batch)

                # Save values for stats
                epsilon_arr.append(epsilon)
                loss_arr.append(loss)
                acc_arr.append(acc)

                # Clear gradients
                y_batch = []
                state_batch = []
                action_batch = []
                
            if terminal:
                print('global_step: {}, thread: {}, episode_reward: {}, steps: {}, avg_acc: {}'.format(global_step, thread_index, episode_reward, local_step, np.average(acc_arr)))
                stats.update({'loss': np.average(loss_arr), 
                            'accuracy': np.average(acc_arr),
                            'qmax': np.average(q_max_arr),
                            'epsilon': np.average(epsilon_arr),
                            'reward': np.sum(reward_arr),
                            'steps': local_step,
                            'step': global_step
                            }) 
            else:
                # Update current state from s_t to s_t1
                state = new_state
                local_game_state.update_state()

def train():
    if settings.use_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'

    # Start game environment to get action_size
    game = GameState(settings.random_seed, 
                    settings.log, 
                    settings.game, 
                    settings.frame_skip, 
                    settings.display, 
                    settings.no_op_max)

    # Set target Deep Q Network
    target_network = DeepQNetwork(-1, 
                                device, 
                                settings.random_seed, 
                                game.action_size,
                                settings.learning_rate, 
                                settings.optimizer)

    # Statistics summary writer
    summary_dir = './logs/asynchronous-1step-{}_game-{}/'.format(settings.method, settings.game)
    summary_writer = tf.summary.FileWriter(summary_dir, target_network.sess.graph)
    stats = Stats(target_network.sess, summary_writer)

    # Prepare parallel workers
    workers = []
    for n in range(settings.parallel_agents):
        worker = Thread(target=worker_thread,
                            args=(n, device, stats))
        workers.append(worker)

    # Start workers
    for t in workers:
        t.start()
    # Join workers
    for t in workers:
        t.join()

def main():
    if settings.evaluate_model:
        evaluate()
    else:
        train()

if __name__ == '__main__':
    main()