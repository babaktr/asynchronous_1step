from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import gym

import matplotlib.pyplot as plt

import time

class GameState(object):
    def __init__(self, random_seed, log, game, frame_skip, display, no_op_max):
        np.random.seed(random_seed)
        self.display = display
        self.frame_skip = frame_skip
        self.log = log  

        # Load game environment
        self.game = gym.make(game)
        # Get minimal action set
        self.action_size = self.game.action_space.n

    '''
    Resets game environments and regenerates new internal state s_t.
    '''
    def reset(self):
        x_t_raw = self.game.reset()
        #x_t_raw = self.game.render(mode='rgb_array')


        ## Make random initial actions
        #if self.no_op_max > 0:
        #    n_no_actions = np.random.randint(0, self.no_op_max + 1)
        #    for _ in range(n_no_actions):
        #        self.ale.act(0)

        x_t = self.process_frame(x_t_raw)

        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        return self.s_t

    '''
    Processes image frame for network input.
    '''
    def process_frame(self, frame):
        return resize(rgb2gray(frame), (84, 84))

    '''
    Make action and observe enviroment return.
    '''
    def step(self, action):
        if self.display:
            self.game.render()

        reward = 0
        for n in range(self.frame_skip):
            x_t1_raw, r, terminal, info = self.game.step(action)
            reward += r
            if terminal:
                break
                
        #x_t1_raw = self.game.render(mode='rgb_array')
        x_t1 = self.process_frame(x_t1_raw)

        #plt.imshow(x_t1)
        #plt.show()
        #time.sleep(100)

        if self.log:
            print info

        # Clip reward to [-1, 1]
        reward = np.clip(reward, -1, 1)

        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1.reshape(84, 84, 1), axis=2)

        return self.s_t1, reward, terminal

    ''' 
    Update internal game state s_t to s_t1.
    '''
    def update_state(self):
        self.s_t = self.s_t1