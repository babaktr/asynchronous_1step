from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import time
import gym

class GameState(object):
    def __init__(self, random_seed, log, game, display):
        np.random.seed(random_seed)
        self.log = log
        self.display = display

        # Load game environment
        self.game = gym.make(game)
        self.game.seed(random_seed)
        # Get minimal action set
        if game == 'Pong-v0' or game == 'Breakout-v0':
            self.action_size = 3
            # Shift action space from [0,1,2] --> [1,2,3]
            self.action_shift = 1
        else:
            # Tip: Rather than letting it pass to this case, see which 
            # actions the game you want to run uses to speed up the training
            # significantly!
            self.action_size = self.game.action_space.n
            seflf.action_shift = 0

    '''
    Resets game environments and regenerates new internal state s_t.
    '''
    def reset(self):
        x_t_raw = self.game.reset()
    
        x_t = self.process_frame(x_t_raw)
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

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

        x_t1_raw, reward, terminal, info = self.game.step(action+self.action_shift)
        #x_t1_raw = self.game.render(mode='rgb_array') # TODO: Keep?
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