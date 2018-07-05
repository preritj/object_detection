import os
import argparse
import tensorflow as tf
from model.mobilenet_obj import MobilenetPose
import functools
from dataset.data_reader import ObjectDataReader
from utils.parse_config import parse_config
from utils.bboxes import (generate_anchors, get_matches,
                          bbox_decode)
from utils.ops import non_max_suppression
import utils.visualize as vis
from tensorflow.python import pywrap_tensorflow
try:
    import horovod.tensorflow as hvd
    print("Found horovod module, will use distributed training")
    use_hvd = True
except ImportError:
    print("Horovod module not found, will not use distributed training")
    use_hvd = False
use_hvd = False

slim = tf.contrib.slim
DEBUG = False


class Trainer(object):

    def __init__(self, cfg_file):
        # Define model parameters
        cfg = parse_config(cfg_file)
        self.data_cfg = cfg['data_config']
        self.train_cfg = cfg['train_config']
        self.model_cfg = cfg['model_config']
        self.infer_cfg = cfg['infer_config']
        self.data_reader = ObjectDataReader(self.data_cfg)
        self.labels = self.data_reader.product_names
        # label_codes = self.data_reader.product_names
        # text_labels = {}
        # with open(self.infer_cfg.products_csv, 'r') as f:
        #     for line in f:
        #         prod_id, prod_name = line.split(',')
        #         prod_name = (prod_name.split('\n')[0]).strip()
        #         try:
        #             prod_id = int(prod_id)
        #         except ValueError:
        #             continue
        #         text_labels[str(prod_id)] = prod_name
        # for i, prod_id in label_codes.items():
        #
        self.hparams = tf.contrib.training.HParams(
            **self.model_cfg.__dict__,
            num_classes=len(self.labels))
        self.gpu_device = '/gpu:0'
        self.cpu_device = '/cpu:0'
        self.param_server_device = '/gpu:0'

    def generate_anchors(self):
        all_anchors = []
        for i, (base_anchor_size, stride) in enumerate(zip(
                self.model_cfg.base_anchor_sizes,
                self.model_cfg.anchor_strides)):
            grid_shape = tf.constant(
                self.model_cfg.input_shape, tf.int32) / stride
            anchors = generate_anchors(
                grid_shape=grid_shape,
                base_anchor_size=base_anchor_size,
                stride=stride,
                scales=self.model_cfg.anchor_scales,
                aspect_ratios=self.model_cfg.anchor_ratios)
            all_anchors.append(anchors)
        return tf.concat(all_anchors, axis=0)

    def get_features_labels_data(self):
        """returns dataset containing (features, labels)"""
        model_cfg = self.model_cfg
        train_cfg = self.train_cfg
        anchors = self.generate_anchors()
        # num_keypoints = len(train_cfg.train_keypoints)
        # data_reader = ObjectDataReader(self.data_cfg)
        # self.labels = data_reader.product_names
        dataset = self.data_reader.read_data(train_cfg)

        def map_fn(images, bboxes, bbox_labels):
            features = {'images': images}
            classes, regs, weights = get_matches(
                gt_bboxes=bboxes,
                gt_classes=bbox_labels,
                pred_bboxes=anchors,
                unmatched_threshold=model_cfg.unmatched_threshold,
                matched_threshold=model_cfg.matched_threshold,
                force_match_for_gt_bbox=model_cfg.force_match_for_gt_bbox,
                scale_factors=model_cfg.scale_factors)
            labels = {'classes': classes,
                      'regs': regs,
                      'weights': weights}
            return features, labels

        dataset = dataset.map(
            map_fn, num_parallel_calls=train_cfg.num_parallel_map_calls)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        dataset = dataset.batch(train_cfg.batch_size)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def prepare_tf_summary(self, features, predictions, max_display=3):
        all_anchors = self.generate_anchors()
        batch_size = self.train_cfg.batch_size
        images = tf.cast(features['images'], tf.uint8)
        images = tf.split(
            images,
            num_or_size_splits=batch_size,
            axis=0)

        bbox_clf_logits = predictions['bbox_clf_logits']
        bbox_probs = tf.nn.softmax(bbox_clf_logits)
        bbox_probs = tf.split(
            tf.squeeze(bbox_probs),
            num_or_size_splits=batch_size,
            axis=0)
        bbox_regs = tf.split(
            predictions['bbox_regs'],
            num_or_size_splits=batch_size,
            axis=0)
        out_images = []

        for i in range(max_display):
            obj_prob = 1. - bbox_probs[i][:, 0]
            indices = tf.squeeze(tf.where(
                tf.greater(obj_prob, 0.5)))

            def _draw_bboxes():
                img = tf.squeeze(images[i])
                bboxes = tf.gather(bbox_regs[i], indices)
                class_probs = tf.gather(bbox_probs[i], indices)
                # bboxes = tf.zeros_like(bboxes)
                anchors = tf.gather(all_anchors, indices)
                bboxes = bbox_decode(
                    bboxes, anchors, self.model_cfg.scale_factors)
                # bboxes = tf.expand_dims(bboxes, axis=0)
                scores = tf.gather(obj_prob, indices)
                selected_indices = tf.image.non_max_suppression(
                    bboxes, scores,
                    max_output_size=10,
                    iou_threshold=0.5)
                bboxes = tf.gather(bboxes, selected_indices)
                class_probs = tf.gather(class_probs, selected_indices)
                top_probs, top_classes = tf.nn.top_k(class_probs, 3)
                vis_fn = functools.partial(
                    vis.visualize_bboxes_on_image,
                    class_labels=self.labels
                )
                out_img = tf.py_func(
                    vis_fn,
                    [img, bboxes, top_classes, top_probs], tf.uint8)
                return tf.expand_dims(out_img, axis=0)
                # return tf.image.draw_bounding_boxes(
                #    images[i], bboxes)

            out_image = tf.cond(
                tf.greater(tf.rank(indices), 0),
                true_fn=_draw_bboxes,
                false_fn=lambda: images[i])
            out_images.append(out_image)

        out_images = tf.concat(out_images, axis=0)
        tf.summary.image('bboxes', out_images, max_display)

    def train(self):
        """run training experiment"""
        if use_hvd:
            hvd.init()
            session_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=str(hvd.local_rank())
                ))
        else:
            session_config = tf.ConfigProto(
                allow_soft_placement=True
            )

        if not os.path.exists(self.train_cfg.model_dir):
            os.makedirs(self.train_cfg.model_dir)

        model_path = os.path.join(
            self.train_cfg.model_dir,
            self.model_cfg.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.hparams.model_dir = model_path

        model_dir = model_path
        if use_hvd and (hvd.rank() != 0):
            # Horovod: save checkpoints only on worker 0
            # to prevent other workers from corrupting them.
            model_dir = None

        run_config = tf.contrib.learn.RunConfig(
            model_dir=model_dir,
            session_config=session_config
        )

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.hparams,  # HParams
            config=run_config  # RunConfig
        )

        hooks = None
        if use_hvd:
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
            # rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights or
            # restored from a checkpoint.
            bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
            hooks = [bcast_hook]

        def train_input_fn():
            """Create input graph for model.
            """
            # TODO : add multi-gpu training
            with tf.device(self.cpu_device):
                dataset = self.get_features_labels_data()
                return dataset

        # train_input_fn = self.input_fn
        estimator.train(input_fn=train_input_fn,
                        hooks=hooks)

    def get_optimizer_fn(self):
        """returns an optimizer function
        which takes as argument learning rate"""
        opt = dict(self.train_cfg.optimizer)
        opt_name = opt.pop('name', None)

        if opt_name == 'adam':
            opt_params = opt.pop('params', {})
            # remove learning rate if present
            opt_params.pop('learning_rate', None)

            def optimizer_fn(lr):
                opt = tf.train.AdamOptimizer(lr)
                if use_hvd:
                    return hvd.DistributedOptimizer(opt)
                else:
                    return opt

        else:
            raise NotImplementedError(
                "Optimizer {} not yet implemented".format(opt_name))

        return optimizer_fn

    def get_train_op(self, loss):
        """Get the training Op.
        Args:
             loss (Tensor): Scalar Tensor that represents the loss function.
        Returns:
            Training Op
        """
        # TODO: build configurable optimizer
        # optimizer_cfg = train_cfg.optimizer

        learning_rate = self.train_cfg.learning_rate
        if use_hvd:
            learning_rate *= hvd.size()
        lr_decay_params = self.train_cfg.learning_rate_decay
        if lr_decay_params is not None:
            lr_decay_fn = functools.partial(
                tf.train.exponential_decay,
                decay_steps=lr_decay_params['decay_steps'],
                decay_rate=lr_decay_params['decay_rate'],
                staircase=True
            )
        else:
            lr_decay_fn = None

        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.get_optimizer_fn(),
            learning_rate=learning_rate,
            learning_rate_decay_fn=lr_decay_fn
        )

    @staticmethod
    def get_eval_metric_ops(labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy')
        }

    def get_model_fn(self):
        """Return the model_fn.
        """
        # TODO: add multi-GPU training and CPU/GPU optimizations
        train_cfg = self.train_cfg

        def model_fn(features, labels, mode, params):
            """Model function used in the estimator.
            Args:
                model (Model): an instance of class Model
                features (Tensor): Input features to the model.
                labels (Tensor): Labels tensor for training and evaluation.
                mode (ModeKeys): Specifies if training, evaluation or prediction.
                params (HParams): hyperparameters.
            Returns:
                (EstimatorSpec): Model to be run by Estimator.
            """
            model = None
            model_name = params.model_name
            print("Using model ", model_name)
            if model_name == 'mobilenet_obj':
                model = MobilenetPose(params)
            else:
                NotImplementedError("{} not implemented".format(model_name))

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            # Define model's architecture
            # inputs = {'images': features}
            # predictions = model.predict(inputs, is_training=is_training)
            predictions = model.predict(features, is_training=is_training)
            with tf.device(self.cpu_device):
                self.prepare_tf_summary(features, predictions)
            # Loss, training and eval operations are not needed during inference.
            loss = None
            train_op = None
            eval_metric_ops = {}
            if mode != tf.estimator.ModeKeys.PREDICT:
                # labels = tf.image.resize_bilinear(
                #     labels, size=params.output_shape)
                # heatmaps = labels[:, :, :, :-1]
                # masks = tf.squeeze(labels[:, :, :, -1])
                # labels = heatmaps
                # ground_truth = {'heatmaps': heatmaps,
                #                 'masks': masks}
                ground_truth = labels
                losses = model.losses(predictions, ground_truth)
                with tf.device(self.cpu_device):
                    for loss_name, loss_val in losses.items():
                        tf.summary.scalar('loss/' + loss_name, loss_val)
                # with tf.device(self.param_server_device):
                loss = train_cfg.bbox_clf_weight * losses['bbox_clf_loss']
                loss += train_cfg.bbox_reg_weight * losses['bbox_reg_loss']
                if self.train_cfg.quantize:
                    # Call the training rewrite which rewrites the graph in-place with
                    # FakeQuantization nodes and folds batchnorm for training. It is
                    # often needed to fine tune a floating point model for quantization
                    # with this training tool. When training from scratch, quant_delay
                    # can be used to activate quantization after training to converge
                    # with the float graph, effectively fine-tuning the model.
                    tf.contrib.quantize.create_training_graph(
                        tf.get_default_graph(), quant_delay=20000)
                train_op = self.get_train_op(loss)
                eval_metric_ops = None  # get_eval_metric_ops(labels, predictions)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )

        return model_fn

    def freeze_model(self):
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(self.infer_cfg.model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph

        absolute_model_dir = os.path.dirname(input_checkpoint)
        output_graph = os.path.join(absolute_model_dir, "frozen_model.pb")

        model, output_nodes = None, None
        model_name = self.hparams.model_name
        print("Using model ", model_name)
        if model_name == 'mobilenet_obj':
            model = MobilenetPose(self.hparams)
        else:
            NotImplementedError("{} not implemented".format(model_name))

        h, w = self.infer_cfg.network_input_shape
        inputs = {'images': tf.placeholder(tf.float32, [None, h, w, 3],
                                           name='images')}
        predictions = model.predict(inputs, is_training=False)
        if self.train_cfg.quantize:
            # Call the eval rewrite which rewrites the graph in-place with
            # FakeQuantization nodes and fold batchnorm for eval.
            tf.contrib.quantize.create_eval_graph()
        bbox_clf_logits = predictions['bbox_clf_logits']
        bbox_classes = tf.nn.softmax(bbox_clf_logits, name='bbox_classes')
        bbox_regs = tf.identity(predictions['bbox_regs'], name='bbox_regs')

        output_nodes = ['bbox_classes', 'bbox_regs']

        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)

        if DEBUG:
            # TODO : load only required variables from checkpoint
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            checkpoint_vars = reader.get_variable_to_shape_map()
            checkpoint_vars = [v for v in tf.trainable_variables()
                               if v.name.split(":")[0] in checkpoint_vars.keys()]
            saver = tf.train.Saver(checkpoint_vars)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_nodes  # The output node names are used to select the useful nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            # tf.train.write_graph(output_graph_def, absolute_model_dir,
            #                      "frozen_model.pbtxt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='./config.yaml', help='Config file')
    args = parser.parse_args()
    config_file = args.config_file
    assert os.path.exists(config_file), \
        "{} not found".format(config_file)
    trainer = Trainer(config_file)
    # trainer.train()
    trainer.freeze_model()
