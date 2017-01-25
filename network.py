import tensorflow as tf

class DeepQNetwork(object):
    '''
    Set up weight variable.
    '''
    def weight_variable(self, shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    '''
    Set up bias variable.
    '''
    def bias_variable(self, shape, name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

    '''
    Set up 2D convolution.
    '''
    def conv2d(self, x, W, stride):
      return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], 
                                padding='VALID')

    def __init__(self, index, device, random_seed, action_size, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()
        self.device = device

        with tf.device(self.device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Action input batch with shape [?, action_size]
            self.a = tf.placeholder(tf.float32, [None, action_size], name='action-input')

            # State input batch with shape [?, 84, 84, 4]
            self.s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='s-input')

            # Desired Q-value batch with shape [?, 1]
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='desired-q_value')

            # Convolutional layer 1 weights and bias with stride=4
            # Produces 16 19x19 outputs
            self.W_conv1 = self.weight_variable([8, 8, 4, 16], 'w_conv1')
            self.b_conv1 = self.bias_variable([16], 'bias-1')
            stride_1 = 4

            # Convolutional layer 1 output
            with tf.name_scope('conv-1') as scope:
                h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, stride_1) + self.b_conv1)

            # Cconvolutional laer 2 weights and biases with stride=2
            # Produces 32 9x9 outputs
            self.W_conv2 = self.weight_variable([4, 4, 16, 32], name='w-conv2')
            self.b_conv2 = self.bias_variable([32], name='b-conv2')
            stride_2 = 2

            # Convolutional layer 2 output 
            with tf.name_scope('conv-2') as scope:
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, stride_2) + self.b_conv2)

            # 256 Fully connected units with weights and biases
            # Weights total 9x9x32 (2592) from the output of the 2nd convolutional layer
            self.W_fc1 = self.weight_variable([9*9*32, 256], name='w-fc')
            self.b_fc1 = self.bias_variable([256], name='b-fc')

            # Fully connected layer output
            with tf.name_scope('fully-connected') as scope:
                h_conv2_flat = tf.reshape(h_conv2, [-1, 9*9*32])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # Output layer weights and biases
            self.W_fc2 = self.weight_variable([256, action_size], name='w-out')
            self.b_fc2 = self.bias_variable([action_size], name='b-out')

            # Output
            with tf.name_scope('output') as scope:
                self.q_values = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

            # Objective function 
            with tf.name_scope('loss') as scope:
                action_q_values = tf.reduce_sum(tf.mul(self.q_values, self.a), reduction_indices=1)
                self.obj_function = tf.reduce_mean(tf.square(self.y - action_q_values))

            with tf.name_scope('train') as scope:
                if optimizer.lower() == 'adam':
                    # Adam Optimizer
                    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.obj_function)
                elif optimizer.lower() == 'rmsprop':
                    # RMSProp
                    self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.obj_function)
                else: 
                    # Gradient Descent
                    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.obj_function)

            # Specify how accuracy is measured
            with tf.name_scope('accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self.q_values, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
            init = tf.global_variables_initializer()
            self.sess.run(init)

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and desired output.
    '''
    def train(self, s_input, a_input, desired_output):
        #print 'self.a.get_shape(): {}'.format(self.a.get_shape())
        #print 'a.input.shape: {}'.format(a_input.shape)
        #print 'self.s.get_shape(): {}'.format(self.s.get_shape())
        #print 's_input.shape: {}'.format(s_input.shape)
        #print 'self.y.get_shape(): {}'.format(self.y.get_shape())
        #print 'desired_output: {}'.format(desired_output.shape) 
        with tf.device(self.device):
            _, loss = self.sess.run([self.train_step, self.obj_function],
                                    feed_dict={self.s: s_input,
                                            self.a: a_input,
                                            self.y: desired_output})
        return loss

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, s_input):
        with tf.device(self.device):
            predicted_output = self.sess.run(self.q_values, feed_dict={self.s: s_input})
        return predicted_output

    '''
    Measures the accuracy of the network based on the specified accuracy measure, the input and the desired output.
    '''
    def get_accuracy(self, s_input, desired_output):
        with tf.device(self.device):
            acc = self.sess.run(self.accuracy, feed_dict={self.s: s_input, 
                                                        self.y: desired_output})
        return acc

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2]

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self.device):
          #with tf.name_scope(name, "GameACNetwork", []) as name:
            for(src_var, dst_var) in zip(src_vars, dst_vars):
              sync_op = tf.assign(dst_var, src_var)
              sync_ops.append(sync_op)

            return tf.group(*sync_ops) #name=name)