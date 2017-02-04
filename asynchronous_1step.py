import numpy as np
import tensorflow as tf

from network import DeepQNetwork
from game_state import GameState
from stats import Stats

from threading import Thread
from threading import Lock

import time 
import signal
import os

flags = tf.app.flags

# General settings
flags.DEFINE_string('game', 'Breakout-v0', 'Name of the Atari game to play. Full list: https://gym.openai.com/envs/')
flags.DEFINE_integer('histogram_summary', 500, 'How many episodes to plot histogram summary over.')
flags.DEFINE_boolean('load_checkpoint', True, 'If it should should from available checkpoints.')
flags.DEFINE_boolean('save_checkpoint', True, 'If it should should save checkpoints when break is triggered.')
flags.DEFINE_boolean('save_stats', True, 'If it should save stats for Tensorboard.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_boolean('use_gpu', False, 'If it should run on GPU rather than CPU.')
flags.DEFINE_boolean('display', False, 'If it you want to render the game.')
flags.DEFINE_boolean('log', False, 'For a verbose log.')

# Training settings
flags.DEFINE_integer('parallel_agents', 8, 'Number of asynchronous agents (threads) to train with.')
flags.DEFINE_integer('global_max_steps', 80000000, 'Maximum training steps.')
flags.DEFINE_integer('local_max_steps', 5, 'Frequency with which each agent network is updated (I_target).')
flags.DEFINE_integer('target_network_update', 10000, 'Frequency with which the shared target network is updated (I_AsyncUpdate).')
flags.DEFINE_integer('frame_skip', 3, 'How many frames to skip (or actions to repeat) for each step,')

# Method settings
flags.DEFINE_string('method', 'q', 'Training algorithm to use [q, sarsa].')
flags.DEFINE_float('gamma', 0.99, 'Discount factor for rewards.')
flags.DEFINE_integer('epsilon_anneal', 1000000, 'Number of steps to anneal epsilon.')

# Optimizer settings
flags.DEFINE_string('optimizer', 'rmsprop', 'Which optimizer to use [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')
flags.DEFINE_float('rms_decay', 0.99, 'RMSProp decay parameter.')
flags.DEFINE_float('rms_epsilon', 0.1, 'RMSProp epsilon parameter.')
flags.DEFINE_float('learning_rate', 0.0007, 'Initial learning rate.')
flags.DEFINE_boolean('anneal_learning_rate', True, 'If learning rate should be annealed over global max steps.')

# Evaluation settings
flags.DEFINE_boolean('evaluate', True, 'If it should run continous evaluation throughout the training session.')
flags.DEFINE_integer('evaluation_episodes', 10, 'How many evaluation episodes to run (and average the evaluation over).')
flags.DEFINE_integer('evaluation_frequency', 100000, 'The frequency of evaluation runs.')


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
def anneal_epsilon(epsilon, final_epsilon, step):
    if epsilon > final_epsilon:
        return 1.0 - step * ((1.0 - final_epsilon) / settings.epsilon_anneal)
    else:
        return final_epsilon

'''
Anneal learning rate.
'''
def anneal_learning_rate(step):
    if settings.anneal_learning_rate:
        return settings.learning_rate - (step * (settings.learning_rate / settings.global_max_steps))

'''
Select action according to exploration epsilon.
'''
def select_action(epsilon, q_values, action_size):
    if np.random.random() > epsilon:
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
Vertically stack batches to match network structure
'''
def stack_batches(state_batch, action_batch, y_batch):
    return np.vstack(state_batch), np.vstack(action_batch), np.vstack(y_batch)

'''
Handles the loading any available checkpoint.
'''
def load_checkpoint(sess, saver, checkpoint_path):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'Checkpoint loaded:', checkpoint.model_checkpoint_path
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_step = int(tokens[len(tokens)-1])
        print 'Global step set to: ', global_step
        # set wall time
        wall_t_fname = checkpoint_path + '/' + 'wall_t.' + str(global_step)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
    else:
        print 'Could not find old checkpoint'
        global_step = 0
        wall_t = 0.0
    return wall_t, global_step

'''
Catches the break signal input from the user.
'''
def signal_handler(signal, frame):
    global stop_requested
    print 'You pressed Ctrl+C!'
    stop_requested = True

def push_stats_updates(stats, loss_arr, acc_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, local_step, global_step):
    stats.update({'loss': np.average(loss_arr), 
                                'accuracy': np.average(acc_arr),
                                'learning_rate': learning_rate,
                                'qmax': np.average(q_max_arr),
                                'epsilon': np.average(epsilon_arr),
                                'episode_actions': action_arr,
                                'reward': np.sum(reward_arr),
                                'steps': local_step,
                                'step': global_step
                                }) 

'''
Runs evaluation of the current network.
'''
def run_evaluation(sess, evaluation_network, stats, game_state, episodes, at_step):
    global stop_requested
    print '>>>>>> Starting evaluation at step {}'.format(at_step)
    rewards = 0
    reward_arr = []
    step_arr = []
    for n in range(episodes):
        local_step = 0
        terminal = False
        state = game_state.reset()
        while not terminal and not stop_requested: 
            q_values = evaluation_network.predict(sess, [state])
            action = select_action(0.01, q_values, game_state.action_size)
            new_state, reward, terminal = game_state.step(action)
            rewards += reward
            local_step += 1
            if terminal:
                reward_arr.append(rewards)
                step_arr.append(local_step)
                print '>>>>>> Evaluation episode {}/{} finished with reward {} on step {}.'.format(n+1, episodes, rewards, local_step)
                rewards = 0
            else:
                state = new_state
                game_state.update_state()

    stats.update_eval({'rewards': np.average(reward_arr), 
                        'steps': np.average(step_arr),
                        'step': at_step
                        }) 
    print '>>>>>> Evaluation done.'


'''
Worker thread that runs an agent training in a local game enviroment.
'''
def worker_thread(thread_index, local_game_state): 
    global stop_requested, global_step, increase_global_step, target_network, online_network, evaluation_network, sess, stats, lock, eval_lock

    # Set worker's initial and final epsilons
    final_epsilon = sample_final_epsilon()
    epsilon = 1.0

    # Prepare gradiets
    y_batch, state_batch, action_batch = [], [], []

    # Prepare stats
    action_arr, q_max_arr, reward_arr, epsilon_arr, loss_arr, acc_arr = [], [], [], [], [], []

    time.sleep(0.5*thread_index)
    g_step = sess.run(global_step)
    print("Starting agent " + str(thread_index) + " with final epsilon: " + str(final_epsilon))

    local_step = 0
    while g_step < settings.global_max_steps and not stop_requested:
        # Reset counters and values
        local_step = 0
        terminal = False
        run_eval = False

        # Get initial game state (s_t)
        state = local_game_state.reset()

        while not terminal:
            # Get the Q-values of the current state (s_t)
            q_values = online_network.predict(sess, [state])

            # Anneal epsilon and select action (a_t)
            epsilon = anneal_epsilon(epsilon, final_epsilon, g_step)
            action = select_action(epsilon, q_values, local_game_state.action_size)
            
            # Make action (a_t) an observe (s_t1)
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
                # SARSA: update with reward + gamma * Q(s',a') for the action taken in s' - not yet fully  tested
                update = reward + (settings.gamma * q_value_new)
            else:
                # Terminal state, update using reward
                update = reward

            # Fill batch
            y_batch.append([update])
            state_batch.append([state])
            action_batch.append(onehot_vector(action, local_game_state.action_size))

            # Save for stats
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))
            action_arr.append(action)

            # Update counters and values
            local_step += 1
            g_step = sess.run(increase_global_step)

            if g_step % settings.evaluation_frequency == 0:
                run_eval = True
                at_step = g_step

            # Update target network on I_target
            if g_step % settings.target_network_update == 0:
                if lock.acquire(False):
                    try:
                        sess.run(target_network.sync_variables_from(online_network))
                        print 'Thread {} updated target network on step: {}'.format(thread_index, g_step)
                    finally:
                        lock.release()

            # Update online network on I_AsyncUpdate
            if local_step % settings.local_max_steps == 0 or terminal:
                state_batch, action_batch, y_batch = stack_batches(state_batch, action_batch, y_batch)
                # Measure accuracy of the network given the batches
                acc = online_network.get_accuracy(sess, state_batch, y_batch)
                # Train online network with gradient batches
                learning_rate = anneal_learning_rate(g_step)
                loss = online_network.train(sess, state_batch, action_batch, y_batch, learning_rate)

                # Save values for stats
                epsilon_arr.append(epsilon)
                loss_arr.append(loss)
                acc_arr.append(acc)

                # Clear gradients
                y_batch, state_batch, action_batch = [], [], []

            if run_eval and settings.evaluate:
                if eval_lock.acquire(False):
                    try:
                        sess.run(evaluation_network.sync_variables_from(online_network))
                        run_evaluation(sess, evaluation_network, stats, local_game_state, settings.evaluation_episodes, at_step)
                    finally:
                        eval_lock.release()
          
            if terminal:
                print 'Thread: {}  /  Global step: {}  /  Local steps: {}  /  Reward: {}  /  Qmax: {}  /  Epsilon: {}'.format(str(thread_index).zfill(2), 
                    g_step, local_step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'), format(np.average(epsilon_arr), '.2f'))

                # Update stats
                if settings.save_stats:
                    push_stats_updates(stats, loss_arr, acc_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, local_step, g_step)

                # Reset stats
                action_arr, q_max_arr, reward_arr, epsilon_arr, loss_arr, acc_arr =  [], [], [], [], [], []
            else:
                # Update current state from s_t to s_t1
                state = new_state
                local_game_state.update_state()

            if stop_requested:
                break

global_step = tf.Variable(0, name='global_step', trainable=False)
increase_global_step = global_step.assign_add(1, use_locking=True)
stop_requested = False

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

lock = Lock()
eval_lock = Lock()

# Prepare game environments
local_game_states = []
for n in range(settings.parallel_agents):
    local_game_state = GameState(settings.random_seed + n, 
                                settings.log, 
                                settings.game, 
                                settings.display,
                                settings.frame_skip)
    local_game_states.append(local_game_state)

# Prepare online network
game = local_game_states[0]
online_network = DeepQNetwork(0, 
                            'online_network',
                            device, 
                            settings.random_seed, 
                            game.action_size, 
                            initial_learning_rate=settings.learning_rate, 
                            optimizer=settings.optimizer,
                            rms_decay=settings.rms_decay,
                            rms_epsilon=settings.rms_epsilon)

# Set target Deep Q Network
target_network = DeepQNetwork(1, 
                            'target_network',
                            device, 
                            settings.random_seed, 
                            game.action_size,
                            initial_learning_rate=settings.learning_rate, 
                            optimizer=settings.optimizer,
                            rms_decay=settings.rms_decay,
                            rms_epsilon=settings.rms_epsilon)


evaluation_network = DeepQNetwork(-1, 
                            'evaluation_network',
                            device, 
                            settings.random_seed, 
                            game.action_size,
                            initial_learning_rate=settings.learning_rate, 
                            optimizer=settings.optimizer,
                            rms_decay=settings.rms_decay,
                            rms_epsilon=settings.rms_epsilon)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

experiment_name = 'asynchronous-1step-{}_game-{}_global-max-{}'.format(settings.method, 
    settings.game, settings.global_max_steps)

# Statistics summary writer
summary_dir = './logs/{}/'.format(experiment_name)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, settings.histogram_summary)

init = tf.global_variables_initializer()
sess.run(init)

wall_t = 0

# Checkpoint handler
if settings.load_checkpoint:
    checkpoint_dir = './checkpoints/{}/'.format(experiment_name)
    saver = tf.train.Saver(max_to_keep=1)
    wall_t, g_step = load_checkpoint(sess, saver, checkpoint_dir)
    sess.run(global_step.assign(g_step))

# Prepare parallel workers
workers = []
for n in range(settings.parallel_agents):
    worker = Thread(target=worker_thread,
                    args=(n, local_game_states[n]))
    workers.append(worker)

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t
print 'Press Ctrl+C to stop'

time.sleep(2)
for t in workers:
    t.start()

signal.pause()
    
for t in workers:
    t.join()

if settings.save_checkpoint:
    print 'Now saving checkpoint. Please wait'
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')  
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)  

    g_step = sess.run(global_step)
    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = checkpoint_dir + '/' + 'wall_t.' + str(g_step)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))

    saver.save(sess, checkpoint_dir + '/' 'checkpoint', global_step=g_step)
