"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


EPSILON = 1e-5


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.check_output_shape()

    @abstractmethod
    def check_output_shape(self):
        """Check shape consistency"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def build_net(self, preprocessed_inputs, is_training=False):
        """Builds network and returns heatmaps and fpn features"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def bbox_clf_reg_net(self, fpn_features, is_training=False):
        """Builds bbox classifier and regressor"""
        raise NotImplementedError("Not yet implemented")

    def predict(self, inputs, is_training=False):
        images = inputs['images']
        preprocessed_inputs = self.preprocess(images)
        fpn_features = self.build_net(
            preprocessed_inputs, is_training=is_training)
        bbox_clf_logits, bbox_regs = self.bbox_clf_reg_net(
            fpn_features, is_training)
        prediction = {'bbox_clf_logits': bbox_clf_logits,
                      'bbox_regs': bbox_regs}
        return prediction

    def bbox_clf_reg_loss(self, clf_labels, clf_logits,
                          clf_weights, regs_gt, regs_pred):
        clf_labels = tf.reshape(clf_labels, [-1])
        clf_weights = tf.reshape(clf_weights, [-1])
        # n_pos_labels = tf.to_float(tf.reduce_sum(clf_labels))
        # n_labels = tf.reduce_sum(clf_weights)
        # n_neg_labels = n_labels - n_pos_labels
        # scale_factor = tf.cond(
        #     tf.greater(n_pos_labels, 0),
        #     lambda: tf.pow(n_neg_labels / n_pos_labels, .3),
        #     lambda: 1.)
        # clf_weights += tf.to_float(clf_labels) * scale_factor
        regs_gt = tf.reshape(regs_gt, [-1, 4])
        clf_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=clf_labels,
            logits=clf_logits,
            weights=clf_weights)
        reg_loss = tf.losses.huber_loss(
            labels=regs_gt,
            predictions=regs_pred,
            delta=1.,
            reduction=tf.losses.Reduction.NONE)
        # reg_loss = tf.losses.mean_squared_error(
        #     labels=regs_gt,
        #     predictions=regs_pred,
        #     reduction=tf.losses.Reduction.NONE)
        reg_loss = tf.reduce_sum(reg_loss, axis=-1)
        reg_loss = tf.reduce_mean(
            reg_loss * tf.to_float(clf_labels))
        return clf_loss, reg_loss

    def losses(self, prediction, ground_truth):
        bbox_clf_logits = prediction['bbox_clf_logits']
        bbox_regs_pred = prediction['bbox_regs']
        bbox_clf_gt = ground_truth['classes']
        bbox_regs_gt = ground_truth['regs']
        bbox_weights = ground_truth['weights']

        bbox_clf_loss, bbox_reg_loss = self.bbox_clf_reg_loss(
            clf_labels=bbox_clf_gt,
            clf_logits=bbox_clf_logits,
            clf_weights=bbox_weights,
            regs_gt=bbox_regs_gt,
            regs_pred=bbox_regs_pred)

        losses = {'bbox_clf_loss': bbox_clf_loss,
                  'bbox_reg_loss': bbox_reg_loss}
        # l2_loss = tf.losses.mean_squared_error(
        #     heatmaps_gt, heatmaps_pred,
        #     reduction=tf.losses.Reduction.NONE)
        # l2_loss = weights * tf.reduce_mean(l2_loss, axis=-1)
        # l2_loss = tf.reduce_mean(l2_loss)
        # # TODO : add regularization losses
        # losses = {'l2_loss': l2_loss}
        return losses
