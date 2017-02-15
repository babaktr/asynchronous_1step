import tensorflow as tf
import numpy as np
from settings import Settings

'''
Network structure from "Playing Atari with Deep Reinforcement Learning" by Mnih et al., 2013 
as specified in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al., 2016.
'''
class DeepQNetwork(object):
    def __init__(self, device, random_seed, action_size, initial_learning_rate=0.0007, optimizer='rmsprop', rms_decay=0.99, rms_epsilon=0.1):
        self.device = device
        self.action_size = action_size
        tf.set_random_seed(random_seed)

        self.learning_rate_value = Settings.learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._build_graph():
                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(allow_soft_replacement=True,
                        log_device_placement=False))
                init = tf.global_variables_initializer()
                self.sess.run(init)

            vars = tf.global_variables()

    '''
    Set up convolutional weight variable.
    '''
    def conv_weight_variable(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.Variable(initializer(shape=shape), name=name)

    '''
    Set up weight variable.
    '''
    def weight_variable(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape=shape), name=name)
    
    '''
    Set up bias variable.
    '''
    def bias_variable(self, shape, name):
        init_value = tf.constant(0.0, shape=shape)
        return tf.Variable(init_value, name=name)

    '''
    Set up 2D convolution.
    '''
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


    def build_network(self)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.placeholder(tf.float43, name='learning_rate', shape=[])

        with tf.name_scope('input') as scope:
            # Action input batch with shape [?, action_size]
            self.a = tf.placeholder(tf.float32, [None, action_size], name='action-input')

            # State input batch with shape [?, 84, 84, 4]
            self.s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='s-input')

            # Target Q-value batch with shape [?, 1]
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target-q_value')

        # Convolutional layer 1 weights and bias with stride=4
        # Produces 16 19x19 outputs
        with tf.name_scope('conv-1') as scope:
            self.W_conv1 = tf.placeholder(tf.float32, shape=[8, 8, 4, 16], name='w_conv1')
            self.b_conv1 = tf.placeholder(tf.float32, shape=[16], name='b_conv1')
            stride_1 = 4

            # Convolutional layer 1 output
            with tf.name_scope('conv1-out') as scope:
                self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.s, self.W_conv1, stride_1), self.b_conv1))

        # Convolutional laer 2 weights and biases with stride=2
        # Produces 32 9x9 outputs
        with tf.name_scope('conv-2') as scope:
            self.W_conv2 = tf.placeholder(tf.float32, shape=[4, 4, 16, 32], name='w_conv2')
            self.b_conv2 = tf.placeholder(tf.float32, shape=[32], name='b_conv2')
            stride_2 = 2

            # Convolutional layer 2 output 
            with tf.name_scope('conv2-out') as scope:
                self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, stride_2), self.b_conv2))

        # 256 Fully connected units with weights and biases
        # Weights total 9x9x32 (2592) from the output of the 2nd convolutional layer
        with tf.name_scope('fully_connected') as scope:
            self.W_fc1 = self.weight_variable([2592, 256], name='w_fc')
            self.b_fc1 = self.bias_variable([256], name='b_fc')

            # Fully connected layer output
            with tf.name_scope('fully-connected-out') as scope:
                h_conv2_flat = tf.reshape(self.h_conv2, [tf.negative(1), 2592])
                h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv2_flat, self.W_fc1), self.b_fc1))

        # Output layer weights and biases
        with tf.name_scope('output') as scope:
            self.W_fc2 = self.weight_variable([256, action_size], name='w_out')
            self.b_fc2 = self.bias_variable([action_size], name='b_out')

            # Output
            with tf.name_scope('q_values') as scope:
                self.q_values = tf.add(tf.matmul(h_fc1, self.W_fc2), self.b_fc2)


        with tf.name_scope('optimizer') as scope:
            self.optimizer = self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
                decay=Settings.rms_decay,
                momentum=Settings.rms_momentum,
                epsilon=Settings.rms_epsilon)

            with tf.name_scope('loss') as scope:
                target_q_value = tf.reduce_sum(tf.multiply(self.q_values, self.a), reduction_indices=1)
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, target_q_value)))

            with tf.name_scope('gradient_clipping') as scope:
                self.gradients = self.optimizer.compute_gradients(self.loss)
                self.clipped_gradients = [(tf.clip_by_average_norm(g, Settings.gradient_clip_norm), v) for g,v in self.gradient]
                self.train_op = self.optimizer.apply_gradients(self.clipped_gradients)

        with tf.name_scope('parameters') as scope:
            self.theta = {
                self.W_conv1: self.conv_weight_variable([8, 8, 4, 16], name='w_conv1'),
                self.b_conv1: self.bias_variable([16], name='b_conv1'),
                self.W_conv2: self.conv_weight_variable([4, 4, 16, 32], name='w_conv2'),
                self.b_conv: self.bias_variable([32], name='b_conv2'),
                self.W_fc1: self.weight_variable([2592, 256], name='w_fc'),
                self.b_fc1: self.bias_variable([256], name='b_fc'),
                self.W_fc2: self.weight_variable([256, self.action_size], name='w_out'),
                self.b_fc2: self.bias_variable([action_size], name='b_out')
                }

            self.theta_target = theta.copy()

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def 

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and target output.
    '''
    def train(self, s_input, a_input, y_input, learn_rate, g_step):
        feed_dict = self.theta.copy()
        feed_dict.update({
            self.s: s_input, 
            self.a: a_input, 
            self.y: y_input, 
            self.lr: learn_rate})

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, s_input, target=False):
        if target:
            feed_dict = self.theta_target.copy()
        else:
            feed_dict = self.theta.copy()
        
        feed_dict.update({self.s: s_input})
        predicted_output = sess.run(self.q_values, feed_dict=feed_dict)
        return predicted_output


    def get_variables(self):
        return [self.W_conv1, self.b_conv1, 
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1, 
                self.W_fc2, self.b_fc2]

    def sync_variables_from(self, source_network):
        with tf.device(self.device):
            source_variables = source_network.get_variables()
            own_variables = self.get_variables()
            sync_ops = []
            for(src_var, own_var) in zip(source_variables, own_variables):            
                sync_op = tf.assign(own_var, src_var)
                sync_ops.append(sync_op)
            return tf.group(*sync_ops)
    
