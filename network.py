import tensorflow as tf
import numpy as np

'''
Network structure from "Playing Atari with Deep Reinforcement Learning" by Mnih et al., 2013 
as specified in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al., 2016.
'''
class DeepQNetwork(object):
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
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], 
                            padding='VALID')

    def __init__(self, name, device, random_seed, action_size, batch_size=5, initial_learning_rate=0.0007, optimizer='rmsprop', rms_decay=0.99, rms_epsilon=0.1):
        self.device = device

        with tf.device(self.device) and tf.name_scope(name) as scope:
            # Set random seed
            tf.set_random_seed(random_seed)

            self.a_array = []
            self.s_array = []
            self.y_array = []
            self.batch_size = batch_size
            self.loss_value = 0

            with tf.name_scope('input') as scope:
                # Action input batch with shape [?, action_size]
                self.a = tf.placeholder(tf.float32, [None, action_size], name='action-input')

                # State input batch with shape [?, 84, 84, 4]
                self.s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='s-input')

                # Target Q-value batch with shape [?, 1]
                self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target-q_value')

                with tf.name_scope('gradient_clipping') as scope:
                    clipped_s = tf.clip_by_norm(self.s, 40.0)
                    clipped_a = tf.clip_by_norm(self.a, 40.0)
                    clipped_y = tf.clip_by_norm(self.y, 40.0)

            # Convolutional layer 1 weights and bias with stride=4
            # Produces 16 19x19 outputs
            with tf.name_scope('conv-1') as scope:
                self.W_conv1 = self.conv_weight_variable([8, 8, 4, 16], 'w_conv1')
                self.b_conv1 = self.bias_variable([16], 'bias-1')
                stride_1 = 4

                # Convolutional layer 1 output
                with tf.name_scope('conv-1-out') as scope:
                    self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(clipped_s, self.W_conv1, stride_1), self.b_conv1))

            # Convolutional laer 2 weights and biases with stride=2
            # Produces 32 9x9 outputs
            with tf.name_scope('conv-2') as scope:
                self.W_conv2 = self.conv_weight_variable([4, 4, 16, 32], name='w-conv2')
                self.b_conv2 = self.bias_variable([32], name='b-conv2')
                stride_2 = 2

                # Convolutional layer 2 output 
                with tf.name_scope('conv-2-out') as scope:
                    self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, stride_2), self.b_conv2))

            # 256 Fully connected units with weights and biases
            # Weights total 9x9x32 (2592) from the output of the 2nd convolutional layer
            with tf.name_scope('fully_connected') as scope:
                self.W_fc1 = self.weight_variable([2592, 256], name='w-fc')
                self.b_fc1 = self.bias_variable([256], name='b-fc')

                # Fully connected layer output
                with tf.name_scope('fully-connected-out') as scope:
                    h_conv2_flat = tf.reshape(self.h_conv2, [tf.negative(1), 2592])
                    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv2_flat, self.W_fc1), self.b_fc1))

            # Output layer weights and biases
            with tf.name_scope('output') as scope:
                self.W_fc2 = self.weight_variable([256, action_size], name='w-out')
                self.b_fc2 = self.bias_variable([action_size], name='b-out')

                # Output
                with tf.name_scope('q_values') as scope:
                    self.q_values = tf.add(tf.matmul(h_fc1, self.W_fc2), self.b_fc2)

            with tf.name_scope('optimizer') as scope:
                if optimizer.lower() == 'adam':
                    # Adam Optimizer
                    self.optimizer_function = tf.train.AdamOptimizer(initial_learning_rate)
                elif optimizer.lower() == 'gradientdecent':
                    # Gradient Descent
                    self.optimizer_function = tf.train.GradientDescentOptimizer(initial_learning_rate)
                else: 
                    # RMSProp
                    self.optimizer_function = tf.train.RMSPropOptimizer(initial_learning_rate, decay=rms_decay, epsilon=rms_epsilon)

                with tf.name_scope('loss'):
                    target_q_value = tf.reduce_sum(tf.multiply(self.q_values, clipped_a), reduction_indices=1)
                    self.loss_function = tf.reduce_mean(tf.square(tf.subtract(clipped_y, target_q_value)))

                with tf.name_scope('gradient_clipping') as scope:
                    # Compute gradients w.r.t. weights
                    variables = self.get_variables()
                    gradients = tf.gradients(self.loss_function, variables)
                    #with tf.name_scope('clipped_gradients') as scope:
                        # Apply gradient norm clipping
                        #clipped_gradients = []
                        #for variable in variables:
                            #gradient = tf.clip_by_norm(variable, 40.0)
                            #clipped_gradients.append(gradient)
                        #gradients = tf.clip_by_norm(gradients, 40.0)
                    gradient_variables = list(zip(gradients, variables))
                    with tf.name_scope('training_op') as scope:
                        self.train_op = self.optimizer_function.apply_gradients(gradient_variables)

            # Specify how accuracy is measured
            with tf.name_scope('accuracy') as scope:
                max_q_value = tf.reduce_max(self.q_values)
                estimated_value = tf.reduce_max(self.y)
                higher_value = tf.reduce_max([max_q_value, estimated_value])
                lower_value = tf.reduce_min([max_q_value, estimated_value])
                self.accuracy = tf.div(lower_value, higher_value)
    
    def clip_gradients(self, gradients):
        variables = self.get_variables()
        clipped_gradients = []
        for gradient in gradients:
            gradient = tf.clip_by_norm(variable, 40.0)
            clipped_gradients.append(gradient)
        return list(zip(clipped_gradients, variables))

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and target output.
    '''
    def train(self, sess, s_input, a_input, y_input, learn_rate):
        with tf.device(self.device):
            self.optimizer_function.learn_rate = learn_rate 
            _, loss = sess.run([self.train_op, self.loss_function], feed_dict={self.s: s_input, self.a: a_input, self.y: y_input})
            return loss
            return loss

    def accumulate_gradients(self, sess, s_input, a_input, y_input, learn_rate):
        with tf.device(self.device):
            self.s_array.append(s_input)
            self.a_array.append(a_input)
            self.y_array.append(y_input)

            print 'accum'

            if len(self.s_array) % self.batch_size == 0:
                #a = np.vstack(self.s_array)
                #print self.s_batch
                #print a.shape
                #ops = []
                #s_op = tf.assign(self.s_batch, np.vstack(self.s_array))
                #a_op = tf.assign(self.a_batch, np.vstack(self.a_array))
                #y_op = tf.assign(self.y_batch, np.vstack(self.y_array))
                #ops.append(s)
                #ops.append(a)
                #ops.append(y)

                #ops.append(self.train(sess, self.s_batch, self.a_batch, self.y_batch, learning_rate))
                #ops.append(tf.assign(self.optimizer_function.learn_rate, learn_rate))
                #ops.append(self.train_op)
                #tf.group(*ops)
                #self.s_array, self.a_array, self.y_array = [], [], []
                print 'update {}'.format(len(self.s_array))
                self.loss_value = self.train(sess, np.vstack(self.s_array), np.vstack(self.a_array), np.vstack(self.y_array), learn_rate)
                self.s_array, self.a_array, self.y_array = [], [], []
                print ' '

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, sess, s_input):
        with tf.device(self.device):
            predicted_output = sess.run(self.q_values, feed_dict={self.s: s_input})
            return predicted_output

    '''
    Measures the accuracy of the network based on the specified accuracy measure, the input and the target output.
    '''
    def get_accuracy(self, sess, s_input, target_output):
        with tf.device(self.device):
            acc = sess.run(self.accuracy, feed_dict={self.s: s_input, 
                                                        self.y: target_output})
            return acc

    def get_variables(self):
        return [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

    def sync_variables_from(self, source_network):
        with tf.device(self.device):
            source_variables = source_network.get_variables()
            own_variables = self.get_variables()
            sync_ops = []
            for(src_var, own_var) in zip(source_variables, own_variables):
                sync_op = tf.assign(own_var, src_var)
                sync_ops.append(sync_op)
                
            return tf.group(*sync_ops)
    
