from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import gym

import matplotlib.pyplot as plt

import time

class GameState(object):
    def __init__(self, random_seed, log, game, frame_skip=0, display=False, no_op_max=7):
        print random_seed
        np.random.seed(random_seed)

        self.display = display

        self.game = gym.make(game)

        # Get minimal action set
        self.action_size = 3 #self.game.action_space.n

        self.log = log

    def reset(self):
        x_t = self.game.reset()

        ## Make random initial actions
        #if self.no_op_max > 0:
        #    n_no_actions = np.random.randint(0, self.no_op_max + 1)
        #    for _ in range(n_no_actions):
        #        self.ale.act(0)

        x_t = self.process_frame(x_t)

        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        return self.s_t

    def process_frame(self, frame):
        return resize(rgb2gray(frame), (84, 84))

    def step(self, action):
        if self.display:
            self.game.render()

        x_t1, reward, terminal, info = self.game.step(action)
        x_t1 = self.process_frame(x_t1)

        #plt.imshow(x_t1)
        #plt.show()
        #time.sleep(100)

        if self.log:
            print info

        # Clip reward to [-1, 1]
        reward = np.clip(reward, -1, 1)

        #print 's_t shape:{}'.format(self.s_t.shape)
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1.reshape(84, 84, 1), axis=2)
        #print 's_t1 shape:{}'.format(self.s_t1.shape)

        return self.s_t1, reward, terminal

    def update_state(self):
        self.s_t = self.s_t1