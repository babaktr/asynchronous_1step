from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as numpy
import time

from settings import Settings
from game_state import GameState
from experience import Experience

class Agent(Process):
    def __init__(self, index, prediction_queue, training_queue, episode_log_queue):
        super(Agent, self).__init()

        self.index = index
        self.prediction_queue = prediction_queue
        self.training_queue = training_queue
        self.episode_log_queue = episode_log_queue

        self.game_state = GameState(settings.random_seed, settings.game, settings.display)

        self.gamma = settings.gamma
        self.wait_queue = Queue(maxsize=1)
        self.stop_flag = Value('i', 0)

    