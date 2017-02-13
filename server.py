from multiprocessing import Queue
import tensorflow as tf

import time
from game_state import game_state
#import networkVP
#import processAgent
#import processstats
#import threadDynamicadjustment
#import threadpredictor
#threadtrainer
#

class ParameterServer:
	def __init__(self, settings)
	#self.stats = stats
	#
	self.training_queue = Queue(max_size=settings.max_queue_size)
	self.prediction_queue = Queue(max_size=settings.max_queue_size)

	self.model = networkVP

	