import numpy as np
import tensorflow as tf

from network import DeepQNetwork
from game_state import GameState
from stats import Stats

from threading import Thread
from threading import Lock

from rmsprop_applier import RMSPropApplier

import time 
import signal
import os

flags = tf.app.flags

# General settings
flags.DEFINE_string('game', 'BreakoutDeterministic-v0', 'Name of the Atari game to play. Full list: https://gym.openai.com/envs/')
flags.DEFINE_integer('histogram_summary', 500, 'How many episodes to plot histogram summary over.')
flags.DEFINE_boolean('load_checkpoint', True, 'If it should should from available checkpoints.')
flags.DEFINE_boolean('save_checkpoint', True, 'If it should should save checkpoints when break is triggered.')
flags.DEFINE_boolean('save_stats', True, 'If it should save stats for Tensorboard.')
flags.DEFINE_integer('random_seed', 1, 'Sets the random seed.')
flags.DEFINE_boolean('use_gpu', True, 'If it should run on GPU rather than CPU.')
flags.DEFINE_boolean('display', False, 'If it you want to render the game.')
flags.DEFINE_boolean('log', False, 'For a verbose log.')

# Training settings
flags.DEFINE_integer('parallel_agents', 16, 'Number of asynchronous agents (threads) to train with.')
flags.DEFINE_integer('global_max_steps', 50000000, 'Maximum training steps.')
flags.DEFINE_integer('local_max_steps', 5, 'Frequency with which each agent network is updated (I_target).')
flags.DEFINE_integer('target_network_update', 10000, 'Frequency with which the shared target network is updated (I_AsyncUpdate).')
flags.DEFINE_integer('frame_skip', 0, 'How many frames to skip (or actions to repeat) for each step.')

# Method settings
flags.DEFINE_string('method', 'q', 'Training algorithm to use [q, sarsa].')
flags.DEFINE_float('gamma', 0.99, 'Discount factor for rewards.')
flags.DEFINE_integer('epsilon_anneal', 1000000, 'Number of steps to anneal epsilon.')

# Optimizer settings
flags.DEFINE_string('optimizer', 'rmsprop', 'Which optimizer to use [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')
flags.DEFINE_float('rms_decay', 0.99, 'RMSProp decay parameter.')
flags.DEFINE_float('rms_epsilon', 0.1, 'RMSProp epsilon parameter.')
flags.DEFINE_float('learning_rate', 0.0016, 'Initial learning rate.')
flags.DEFINE_boolean('anneal_learning_rate', True, 'If learning rate should be annealed over global max steps.')

# Evaluation settings
flags.DEFINE_boolean('evaluate', True, 'If it should run continous evaluation throughout the training session.')
flags.DEFINE_integer('evaluation_episodes', 10, 'How many evaluation episodes to run (and average the evaluation over).')
flags.DEFINE_integer('evaluation_frequency', 200000, 'The frequency of evaluation runs.')


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

def push_stats_updates(stats, loss_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, l_step, g_step):
    stats.update({'loss': np.average(loss_arr), 
                'learning_rate': learning_rate,
                'qmax': np.average(q_max_arr),
                'epsilon': np.average(epsilon_arr),
                'episode_actions': action_arr,
                'reward': np.sum(reward_arr),
                'steps': l_step,
                'step': g_step
                }) 

'''
Runs evaluation of the current network.
'''
def run_evaluation(sess, thread_id, evaluation_network, stats, game_state, episodes, at_step):
    global stop_requested
    print '>>>>>> Starting evaluation with thread {} at step {}'.format(thread_id, at_step)
    rewards = 0
    reward_arr = []
    score_arr = []
    step_arr = []
    for n in range(episodes):
        local_step = 0
        rewards = 0
        scores = 0
        terminal = False
        state = game_state.reset()
        while not terminal and not stop_requested: 
            q_values = evaluation_network.predict(sess, [state])
            action = select_action(0.01, q_values, game_state.action_size)
            new_state, reward, terminal = game_state.step(action)
            if reward > 0.0:
                scores += reward
            rewards += reward
            local_step += 1
            if terminal:
                reward_arr.append(rewards)
                step_arr.append(local_step)
                score_arr.append(scores)
                print '>>>>>> Evaluation episode {}/{} finished with reward {} on step {}.'.format(n+1, episodes, rewards, local_step)
            else:
                state = new_state
                game_state.update_state()
    r_avg = np.average(reward_arr)
    st_avg = np.average(step_arr)
    sc_avg = np.average(score_arr)
    if not stop_requested:
        stats.update_eval({'rewards': np.average(r_avg), 
                            'score': np.average(sc_avg),
                            'steps': np.average(st_avg),
                            'step': at_step
                            }) 
        print '>>>>>> Evaluation done with average reward: {}, score {}, step {}.'.format(r_avg, sc_avg, st_avg)


'''
Worker thread that runs an agent training in a local game enviroment.
'''
def worker_thread(thread_index, local_game_state): 
    global stop_requested, global_step, increase_global_step, sess, stats, lock, eval_lock   # General
    global target_network, online_network, evaluation_network               # Networks


    # Set worker's initial and final epsilons
    final_epsilon = sample_final_epsilon()
    epsilon = 1.0

    time.sleep(0.5*thread_index)
    g_step = sess.run(global_step)
    print("Starting agent " + str(thread_index) + " with final epsilon: " + str(final_epsilon))

    while g_step < settings.global_max_steps and not stop_requested:
        # Reset counters and values
        local_step = 0
        terminal = False
        run_eval = False

        state_batch = []
        action_batch = []
        target_batch = []

        # Reset stats
        action_arr, q_max_arr, reward_arr, epsilon_arr, loss_arr = [], [], [], [], []

        # Get initial game state (s_t)
        state = local_game_state.reset()

        while not terminal:
            # Get the Q-values of the current state (s_t)
            q_values = local_network.predict(sess, [state])

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

            state_batch.append([state])
            action_batch.append(onehot_vector(action, local_game_state.action_size))
            target_batch.append([update])          

            # Save for stats
            action_arr.append(action)
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))
            loss_arr.append(0)
            epsilon_arr.append(epsilon)

            # Update counters and values
            local_step += 1
            g_step = sess.run(increase_global_step)

            # Update target network on I_target
            if g_step % settings.target_network_update == 0 and lock.acquire(False):
                    try:
                        sess.run(target_network.sync_variables_from(online_network))
                    
                    except IndexError:
                        print 'INDEX ERROR!'
                    except AssertionError:
                        print 'ASSERTION ERROR'
                    finally:
                        print 'Thread {} updated target network on step: {}'.format(thread_index, g_step)
                        lock.release()

            if local_step % settings.local_max_steps == 0 or terminal:

                loss = online_network.train(sess, np.vstack(state_batch), np.vstack(action_batch), np.vstack(target_batch), anneal_learning_rate(g_step), g_step)
                loss_arr.append(loss)

                state_batch = []
                action_batch = []
                target_batch = []

            if g_step % settings.evaluation_frequency == 0 and settings.evaluate and eval_lock.acquire(False):
                    try:
                        sess.run(evaluation_network.sync_variables_from(online_network))
                        run_evaluation(sess, thread_index, evaluation_network, stats, local_game_state, settings.evaluation_episodes, g_step)
                    except IndexError:
                        print 'INDEX ERROR'
                    except AssertionError:
                        print 'ASSERTION ERROR'
                    finally:
                        eval_lock.release()

            if terminal:
                #print 'pushing stats'
                print 'Thread: {}  /  Global step: {}  /  Local steps: {}  /  Reward: {}  /  Qmax: {}  /  Epsilon: {}'.format(str(thread_index).zfill(2), 
                    g_step, local_step, np.sum(reward_arr), format(np.average(q_max_arr), '.2f'), format(np.average(epsilon_arr), '.2f'))

                # Update stats
                if settings.save_stats:
                    learning_rate = anneal_learning_rate(g_step)
                    push_stats_updates(stats, loss_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, local_step, g_step)

            else:
                # Update current state from s_t to s_t1
                state = new_state
                local_game_state.update_state()

            if stop_requested:
                break

stop_requested = False


if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'


with tf.name_scope('global_step_counter') as cope:
    with tf.device(device):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('increase_global_step') as scope:
            increase_global_step = global_step.assign_add(1, use_locking=True)

# Prepare locks
lock = Lock()
eval_lock = Lock()
update_lock = Lock()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))

# Prepare network stat savers
#acc_arr, loss_arr = [], []

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

# Prepare online network
with tf.name_scope('online_network'):
    online_network = DeepQNetwork('global_online_network', device, settings.random_seed, game.action_size, 
                                initial_learning_rate=settings.learning_rate, 
                                optimizer=settings.optimizer,
                                rms_decay=settings.rms_decay,
                                rms_epsilon=settings.rms_epsilon)

    with tf.name_scope('local_networks') as scope:
        local_networks = []
        for n in range(settings.parallel_agents):
            name = 'local_network_' + str(n)
            local_network = DeepQNetwork(name, device, settings.random_seed, game.action_size,
                                        initial_learning_rate=settings.learning_rate,
                                        optimizer=settings.optimizer,
                                        rms_decay=settings.rms_decay,
                                        rms_epsilon=settings.rms_epsilon)
            local_networks.append(local_network)


# Prepare target network
target_network = DeepQNetwork('target_network', device, settings.random_seed, game.action_size)
# Prepare evaluation network
evaluation_network = DeepQNetwork('evaluation_network', device, settings.random_seed, game.action_size)


experiment_name = 'aggrlr-asynchronous-1step-{}_game-{}_global-max-{}'.format(settings.method, 
    settings.game, settings.global_max_steps)

# Statistics summary writer
summary_dir = './logs/{}/'.format(experiment_name)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, settings.histogram_summary)

wall_t = 0

# Checkpoint handler
if settings.load_checkpoint:
    checkpoint_dir = './checkpoints/{}/'.format(experiment_name)
    saver = tf.train.Saver(max_to_keep=1)
    wall_t, g_step = load_checkpoint(sess, saver, checkpoint_dir)
    sess.run(global_step.assign(g_step))

shared_variables = online_network.get_variables()

grad_applier = RMSPropApplier(learning_rate=settings.learning_rate,
                            decay=settings.rms_decay,
                            epsilon=settings.rms_epsilon,
                            clip_norm=40.,
                            device=device)

# Prepare parallel workers
workers = []
for n in range(settings.parallel_agents):
    with tf.device(device):
        local_network = local_networks[n]
        apply_gradients_op = grad_applier.apply_gradients(shared_variables, local_network.gradients)
        sync_op = local_network.sync_variables_from(online_network)

    worker = Thread(target=worker_thread,
                    args=(n, local_game_states[n], local_network, apply_gradients_op, sync_op))
    workers.append(worker)

init = tf.global_variables_initializer()
sess.run(init)

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
