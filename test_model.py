import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from network import DeepQNetwork
from game_state import GameState

import time 
import signal
import os

flags = tf.app.flags

# General settings
flags.DEFINE_string('mode', 'play', 'What to run with the loaded model [play, visualize].')
flags.DEFINE_string('game', 'Breakout-v0', 'What game to play.')
flags.DEFINE_boolean('load_checkpoint', True, 'If it should load from available checkpoints.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_boolean('log', False, 'For a verbose log.')
flags.DEFINE_integer('frame_skip', 2, 'How many frames to skip (or actions to repeat) for each step.')


flags.DEFINE_integer('global_max_steps', 80000000, 'Set this to the same as in your experiment.')

# Method settings
flags.DEFINE_string('method', 'q', 'Training algorithm to use [q, sarsa].')
flags.DEFINE_float('epsilon', 0.01, 'Which epsilon to run with.')

settings = flags.FLAGS

def select_action(epsilon, q_values, action_size):
    if np.random.random() > epsilon:
        return np.argmax(q_values)
    else:
        return np.random.randint(0, action_size)

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

def getActivations(sess, s, layer, stimuli, filters):
  #print "stim shape: %s" % stimuli.shape
  units = layer.eval(session=sess, feed_dict=({s: [stimuli]}))
  plotNNFilter(units, filters)


def plotNNFilter(units, filters):
  filters = units.shape[3]
  test = units.shape[1]
  print test
  #plt.figure(1, figsize=(20,20))

  fig, axes = plt.subplots(1, filters, figsize=(30, 6),
             subplot_kw={'xticks': [], 'yticks': []})

  print filters

  for ax,i in zip(axes.flat, range(1*filters)):
    inch = i//filters
    outch = i%filters
    img = units[0,:,:,i]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))
  plt.show()


'''
Catches the break signal input from the user.
'''
def signal_handler(signal, frame):
    global stop_requested
    print 'You pressed Ctrl+C!'
    stop_requested = True

'''
Worker thread that runs an agent training in a game enviroment.
'''
def play(game_state): 
    global stop_requested, sess

    print 'Starting agent with epsilon: {}'.format(settings.epsilon)

    episode = 0
    while not stop_requested:
        # Reset counters and values
        step = 0
        terminal = False
        reward_arr = []
        q_max_arr = []
        # Get initial game observation
        state = game_state.reset()

        while not terminal:
            # Get the Q-values of the current state
            q_values = online_network.predict(sess, [state])

            action = select_action(settings.epsilon, q_values, game_state.action_size)

            time.sleep(0.08)
            
            # Make action an observe 
            new_state, reward, terminal = game_state.step(action)
            # Get the new state's Q-values
            #q_values_new = target_network.predict(sess, [new_state])

            # Update counters and values
            step += 1
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))
        
            if terminal:
                print 'Episode: {}  /  steps: {}  /  Reward: {}  /  Qmax: {}'.format(episode, 
                    step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'))
                episode +=1
            else:
                # Update current state from s_t to s_t1
                state = new_state
                game_state.update_state()

            if stop_requested:
                break

def visualize(game_state):
    global  online_network, sess

    state = game_state.reset()

    for n in range(10):
        state, _, _ = game.step(0)

    x_t = game.x_t

    plt.imshow(x_t, interpolation="nearest", cmap=plt.cm.gray)


    W_conv1 = sess.run(online_network.W_conv1)

    # show graph of W_conv1
    fig, axes = plt.subplots(4, 16, figsize=(12, 6),
               subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax,i in zip(axes.flat, range(4*16)):
        inch = i//16
        outch = i%16
        img = W_conv1[:,:,inch,outch]
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

    plt.show()

    W_conv2 = sess.run(online_network.W_conv2)

    # show graph of W_conv2
    fig, axes = plt.subplots(2, 32, figsize=(27, 6),
               subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax,i in zip(axes.flat, range(2*32)):
        inch = i//32
        outch = i%32
        img = W_conv2[:,:,inch,outch]
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title(str(inch) + "," + str(outch))

    plt.show()

    arr = sess.run(online_network.get_variables())

    s = tf.placeholder("float", [None, 84, 84, 4])

    b_conv1 = sess.run(online_network.b_conv1)
    b_conv2 = sess.run(online_network.b_conv2)

    inp_1 = online_network.conv2d(s, W_conv1, 2)
    h_conv1 = tf.nn.relu(inp_1 + b_conv1)

    inp_2 = online_network.conv2d(h_conv1, W_conv2, 4)
    h_conv2 = tf.nn.relu(inp_2 + b_conv2)

    getActivations(sess, s, h_conv1, state, 16)
    getActivations(sess, s, h_conv2, state, 32)


stop_requested = False

device = '/cpu:0'

if settings.mode == 'play':
    display = True
else:
    display = False

# Prepare game environments
game_state = GameState(settings.random_seed, 
                    settings.log, 
                    settings.game, 
                    display,
                    settings.frame_skip)

# Prepare online network
game = game_state
online_network = DeepQNetwork('online_network',
                            device, 
                            settings.random_seed, 
                            game.action_size)

# Set target Deep Q Network
target_network = DeepQNetwork('target_network',
                            device, 
                            settings.random_seed, 
                            game.action_size)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

experiment_name = 'asynchronous-1step-{}_game-{}_global-max-{}'.format(settings.method, 
    settings.game, settings.global_max_steps)

init = tf.global_variables_initializer()
sess.run(init)

wall_t = 0

# Checkpoint handler
if settings.load_checkpoint:
    checkpoint_dir = './checkpoints/{}/'.format(experiment_name)
    saver = tf.train.Saver(max_to_keep=1)
    wall_t, global_step = load_checkpoint(sess, saver, checkpoint_dir)

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t
print 'Press Ctrl+C to stop'

time.sleep(2)
    
if settings.mode == 'play':
    play(game_state)
elif settings.mode == 'visualize':
    visualize(game_state)
