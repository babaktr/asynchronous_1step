import tensorflow as tf
import numpy as np

class Stats(object):  
  def __init__(self, sess, summary_writer):
    self.sess = sess

    self.writer = summary_writer
    with tf.variable_scope('summary'):
      scalar_summary_tags = ['network/loss', 'network/accuracy', 'episode/avg_q_max', 'episode/epsilon', 'episode/reward', 'episode/steps']

      self.summary_placeholders = {}
      self.summary_ops = {}

    for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
        self.summary_ops[tag]  = tf.summary.scalar(tag, self.summary_placeholders[tag])

  def update(self, dictionary):
    self.inject_summary({
          'network/loss': dictionary['loss'],
          'network/accuracy': dictionary['accuracy'],
          'episode/avg_q_max': dictionary['qmax'],
          'episode/epsilon': dictionary['epsilon'],
          'episode/reward': dictionary['reward'],
          'episode/steps':dictionary['steps']
      }, dictionary['step'])

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)
      self.writer.flush()

