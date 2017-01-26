from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import gym

import matplotlib.pyplot as plt

import time

class GameState(object):
    def __init__(self, random_seed, log, game, frame_skip, display, no_op_max):
        np.random.seed(random_seed)
        self.log = log
        self.frame_skip = frame_skip
        self.display = display
        self.no_op_max = no_op_max

        # Load game environment
        self.game = gym.make(game)
        self.game.seed(random_seed)
        # Get minimal action set
        self.action_size = self.game.action_space.n

    '''
    Resets game environments and regenerates new internal state s_t.
    '''
    def reset(self):
        x_t_raw = self.game.reset()
        #x_t_raw = self.game.render(mode='rgb_array') TODO: Keep?

        # Make no-op actions
        if self.no_op_max > 0:
            no_op_actions = np.random.randint(0, no_op_max + 1)
            for n in range(no_op_actions):
                x_t_raw, _, _, _ = self.game.step(0)
    
        x_t = self.process_frame(x_t_raw)
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        return self.s_t

    '''
    Processes image frame for network input.
    '''
    def process_frame(self, frame):
        return resize(rgb2gray(frame), (84, 84))
        #return resize(rgb2gray(frame), (110, 84))[17:110 - 9, :] TODO: Keep?

    '''
    Make action and observe enviroment return.
    '''
    def step(self, action):
        if self.display:
            self.game.render()

        reward = 0
        for n in range(self.frame_skip + 1):
            x_t1_raw, r, terminal, info = self.game.step(action)
            reward += r
            if terminal:
                break
                
        #x_t1_raw = self.game.render(mode='rgb_array') # TODO: Keep?
        x_t1 = self.process_frame(x_t1_raw)
        
        if False: # TODO: Keep?
            plt.imshow(x_t1)
            plt.show()
            time.sleep(100)

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