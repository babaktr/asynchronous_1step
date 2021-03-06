from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import time
import gym

class GameState(object):
    def __init__(self, random_seed, log, game, display, frame_skip):
        np.random.seed(random_seed)
        self.log = log
        self.display = display
        self.frame_skip = frame_skip

        # Load game environment
        self.game = gym.make(game)
        self.game.seed(random_seed)
        # Get minimal action set
        if game == 'PongDeterministic-v0' or game == 'BreakoutDeterministic-v0':
            self.action_size = 3
            # Shift action space from [0,1,2] --> [1,2,3]
            self.action_shift = 1
        else:
            # Tip: Rather than letting it pass to this case, see which 
            # actions the game you want to run uses to speed up the training
            # significantly!
            self.action_size = self.game.action_space.n
            self.action_shift = 0

    '''
    Resets game environments and regenerates new internal state s_t.
    '''
    def reset(self):
        x_t_raw = self.game.reset()
    
        self.x_t = self.process_frame(x_t_raw)
        self.s_t = np.stack((self.x_t, self.x_t, self.x_t, self.x_t), axis=2)

        return self.s_t

    '''
    Processes image frame for network input.
    '''
    def process_frame(self, frame):
        frame_cut = frame[30:195,10:150]
        self.x_t = resize(rgb2gray(frame_cut), (84, 84))
        return self.x_t

    '''
    Make action and observe enviroment return.
    '''
    def step(self, action):
        if self.display:
            self.game.render()

        accum_reward = 0
        for n in range (self.frame_skip+1):
            x_t1_raw, r, terminal, info = self.game.step(action+self.action_shift)
            accum_reward += r
            if terminal:
                break

        reward = accum_reward
        x_t1 = self.process_frame(x_t1_raw)
        
        if False: # TODO: Keep?
            plt.imshow(x_t1, cmap='gray')
            plt.savefig(str(np.random.randint(0,10000)) + '.png')
            time.sleep(10)

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