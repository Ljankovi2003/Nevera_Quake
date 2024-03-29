"""nevera_quake model.

The function `get` is implemented to help prototype other models.
One can create a subclass `Proto(nevera_quake)` and overwrite the 
`_setup_prediction` method to change the network architecture.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import tflib.model
import tflib.layers as layers

def get(model_name, inputs, config, checkpoint_dir, is_training=False):
    """Returns a Model instance by model name.

    Args:
        model_name: Name of the model class to instantiate.
        inputs: Samples as returned by a DataPipeline.
        config: Dictionary of model parameters.
        checkpoint_dir: Directory to save/load model checkpoints.
        is_training: Boolean indicating whether the model is being trained.
    """
    return globals()[model_name](inputs, config, checkpoint_dir, is_training)

class Nevera_Quake(tflib.model.BaseModel):

    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        """Initialize Nevera_Quake model instance.

        Args:
            inputs: Input data for the model.
            config: Configuration parameters for the model.
            checkpoint_dir: Directory for saving/loading model checkpoints.
            is_training: Boolean indicating whether the model is being trained.
            reuse: Boolean indicating whether to reuse variables.
        """
        super(Nevera_Quake, self).__init__(inputs, checkpoint_dir, is_training=is_training, reuse=reuse)
        self.is_training = is_training
        self.config = config

    def _setup_prediction(self):
        """Set up the prediction network architecture."""
        self.batch_size = self.inputs['data'].get_shape().as_list()[0]

        current_layer = self.inputs['data']
        c = 32  # Number of channels per conv layer
        ksize = 3  # Size of the convolution kernel
        depth = 8

        # Convolution layers
        for i in range(depth):
            current_layer = layers.conv1(current_layer, c, ksize, stride=2, scope='conv{}'.format(i+1), padding='SAME')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
            self.layers['conv{}'.format(i+1)] = current_layer

        bs, width, _ = current_layer.get_shape().as_list()
        current_layer = tf.reshape(current_layer, [bs, width * c], name="reshape")

        # Fully connected layer
        current_layer = layers.fc(current_layer, self.config.n_clusters, scope='logits', activation_fn=None)
        self.layers['logits'] = current_layer
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)

        # Softmax layer
        self.layers['class_prob'] = tf.nn.softmax(current_layer, name='class_prob')
        self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 1, name='class_pred')

        # Regularization
        tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.config.regularization),
            weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))

    def validation_metrics(self):
        """Get validation metrics for the model."""
        if not hasattr(self, '_validation_metrics'):
            self._setup_loss()

            self._validation_metrics = {
                'loss': self.loss,
                'detection_accuracy': self.detection_accuracy,
                'localization_accuracy': self.localization_accuracy
            }
        return self._validation_metrics

    def validation_metrics_message(self, metrics):
        """Format validation metrics message."""
        s = 'loss = {:.5f} | det. acc. = {:.1f}% | loc. acc. = {:.1f}%'.format(metrics['loss'],
                                                                                metrics['detection_accuracy']*100,
                                                                                metrics['localization_accuracy']*100)
        return s

    def _setup_loss(self):
        """Set up loss computation for the model."""
        with tf.name_scope('loss'):
            # Change target range from -1:n_clusters-1 to 0:n_clusters
            targets = tf.add(self.inputs['cluster_id'], self.config.add)
            raw_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.layers['logits'], targets))
            self.summaries.append(tf.scalar_summary('loss/train_raw', raw_loss))

        self.loss = raw_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_losses:
            with tf.name_scope('regularizers'):
                reg_loss = sum(reg_losses)
                self.summaries.append(tf.scalar_summary('loss/regularization', reg_loss))
            self.loss += reg_loss

        self.summaries.append(tf.scalar_summary('loss/train', self.loss))

        with tf.name_scope('accuracy'):
            is_true_event = tf.cast(tf.greater(targets, tf.zeros_like(targets)), tf.int64)
            is_pred_event = tf.cast(tf.greater(self.layers['class_prediction'], tf.zeros_like(targets)), tf.int64)
            detection_is_correct = tf.equal(is_true_event, is_pred_event)
            is_correct = tf.equal(self.layers['class_prediction'], targets)
            self.detection_accuracy = tf.reduce_mean(tf.to_float(detection_is_correct))
            self.localization_accuracy = tf.reduce_mean(tf.to_float(is_correct))
            self.summaries.append(tf.scalar_summary('detection_accuracy/train', self.detection_accuracy))
            self.summaries.append(tf.scalar_summary('localization_accuracy/train', self.localization_accuracy))

    def _setup_optimizer(self, learning_rate):
        """Set up optimizer for the model."""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops, name='update_ops')
            with tf.control_dependencies([updates]):
                self.loss = tf.identity(self.loss)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, name='optimizer', global_step=self.global_step)
        self.optimizer = optim

    def _tofetch(self):
        """Specify tensors to fetch during training."""
        return {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'detection_accuracy': self.detection_accuracy,
            'localization_accuracy': self.localization_accuracy
        }

    def _summary_step(self, step_data):
        """Format summary for each training step."""
        step = step_data['step']
        loss = step_data['loss']
        det_accuracy = step_data['detection_accuracy']
        loc_accuracy = step_data['localization_accuracy']
        duration = step_data['duration']
        avg_duration = 1000 * duration / step

        if self.is_training:
            toprint = 'Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | det. acc. = {:.1f}% | loc. acc. = {:.1f}%'.format(
                step, duration, avg_duration, loss, 100 * det_accuracy, 100 * loc_accuracy)
        else:
            toprint = 'Step {} | {:.0f}s ({:.0f}ms) | accuracy = {:.1f}% | accuracy = {:.1f}%'.format(
                step, duration, avg_duration, 100 * det_accuracy, 100 * loc_accuracy)

        return toprint
