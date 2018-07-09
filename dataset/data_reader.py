import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.avaRetail import AVAretail
import json
from tqdm import tqdm
import functools
from utils.dataset_util import (
    random_brightness, random_contrast, random_rotate, random_hue,
    random_gaussian_noise, random_flip_left_right, keypoints_select, resize)


slim_example_decoder = tf.contrib.slim.tfexample_decoder


class ObjectDataReader(object):
    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        self.datasets = []
        self.product_names = None

        for dataset in data_cfg.datasets:
            name = dataset['name']
            weight = dataset['weight']
            data_dir = dataset['data_dir']
            train_files = glob(os.path.join(
                data_dir, dataset['train_files']))
            tfrecord_files = dataset['tfrecord_files']
            if type(tfrecord_files) is list:
                tfrecord_files = [os.path.join(
                    data_dir, f) for f in tfrecord_files]
            elif "*" in tfrecord_files:
                tfrecord_files = glob(os.path.join(
                    data_dir, tfrecord_files))
            else:
                tfrecord_files = [os.path.join(
                    data_dir, tfrecord_files)]
            if self.product_names is None:
                labels_file = os.path.dirname(train_files[0])
                labels_file = os.path.join(labels_file, 'labels.txt')
                with open(labels_file, 'r') as f:
                    self.product_names = {}
                    for line in f:
                        label, prod_name = line.split()
                        self.product_names[int(label)] = prod_name

                text_labels = {}
                with open(self.data_cfg.labels_csv, 'r') as f:
                    for line in f:
                        prod_id, prod_name = line.split(',')
                        prod_name = (prod_name.split('\n')[0]).strip()
                        try:
                            prod_id = int(prod_id)
                        except ValueError:
                            continue
                        text_labels[str(prod_id)] = prod_name
                with open(self.data_cfg.out_labels, 'w') as f:
                    for label, prod_id in self.product_names.items():
                        if label == 0:
                            f.write(str(label) + ', Background\n')
                        else:
                            prod_name = text_labels[prod_id]
                            f.write(str(label) + ', ' + prod_name + '\n')
            if (len(tfrecord_files) == 0) or dataset['overwrite_tfrecord']:
                for train_file in train_files:
                    train_file_dir = os.path.dirname(train_file)
                    tfrecord_name = os.path.basename(train_file_dir) + '.records'
                    sub_dir = os.path.dirname(dataset['tfrecord_files'])
                    tfrecord_path = os.path.join(data_dir, sub_dir, tfrecord_name)
                    tfrecord_dir = os.path.dirname(tfrecord_path)
                    if not os.path.exists(tfrecord_dir):
                        os.makedirs(tfrecord_dir)
                    ds = self.add_dataset(name, data_dir, train_file,
                                          tfrecord_path, weight)
                    ds.create_tf_record(tfrecord_path)
                    self.datasets.append({'name': name,
                                          'tfrecord_path': tfrecord_path,
                                          'weight': weight})
            else:
                for tfrecord_path in tfrecord_files:
                    self.datasets.append({'name': name,
                                          'tfrecord_path': tfrecord_path,
                                          'weight': weight})

    def add_dataset(self, name, data_dir, train_file, tfrecord_file, weight):
        if name == 'ava':
            ds = AVAretail(self.data_cfg, data_dir, train_file)
        else:
            raise RuntimeError('Dataset not supported')
        return ds

    def _get_probs(self):
        probs = [ds['weight'] for ds in self.datasets]
        probs = np.array(probs)
        return probs / np.sum(probs)

    @staticmethod
    def _get_tensor(tensor):
        if isinstance(tensor, tf.SparseTensor):
            return tf.sparse_tensor_to_dense(tensor)
        return tensor

    @staticmethod
    def _image_decoder(keys_to_tensors):
        filename = keys_to_tensors['image/filename']
        image_string = tf.read_file(filename)
        # TODO: decode after crop to increase speed
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        return image_decoded

    @staticmethod
    def _bbox_decoder(keys_to_tensor):
        bboxes = keys_to_tensor['image/object/bboxes']
        # img_shape = keys_to_tensor['image/shape']
        num_instances = ObjectDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances] + [4]
        if isinstance(bboxes, tf.SparseTensor):
            bboxes = tf.sparse_reshape(bboxes, shape=shape)
            bboxes = tf.sparse_tensor_to_dense(bboxes)
        else:
            bboxes = tf.reshape(bboxes, shape=shape)
        return bboxes

    @staticmethod
    def _label_decoder(keys_to_tensor):
        labels = keys_to_tensor['image/object/labels']
        # img_shape = keys_to_tensor['image/shape']
        num_instances = ObjectDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances]
        if isinstance(labels, tf.SparseTensor):
            labels = tf.sparse_reshape(labels, shape=shape)
            labels = tf.sparse_tensor_to_dense(labels)
        else:
            labels = tf.reshape(labels, shape=shape)
        return labels

    def _decoder(self):
        keys_to_features = {
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/shape':
                tf.FixedLenFeature([2], tf.int64),
            'image/num_instances':
                tf.FixedLenFeature((), tf.int64),
            'image/object/bboxes':
                tf.VarLenFeature(tf.float32),
            'image/object/labels':
                tf.VarLenFeature(tf.int64)
        }
        items_to_handlers = {
            'image': slim_example_decoder.ItemHandlerCallback(
                'image/filename', self._image_decoder),
            'bboxes': slim_example_decoder.ItemHandlerCallback(
                ['image/object/bboxes', 'image/num_instances'],
                self._bbox_decoder),
            'labels': slim_example_decoder.ItemHandlerCallback(
                ['image/object/labels', 'image/num_instances'],
                self._label_decoder)
        }
        decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        return decoder

    def augment_data(self, dataset, train_cfg):
        aug_cfg = train_cfg.augmentation
        # preprocess_cfg = train_cfg.preprocess
        # img_size = preprocess_cfg['image_resize']
        # if aug_cfg['flip_left_right']:
        #     kp_dict = {kp_name: i for i, kp_name in
        #                enumerate(train_cfg.train_keypoints)}
        #     flipped_kp_indices = []
        #     for kp_name in train_cfg.train_keypoints:
        #         if kp_name.startswith('left'):
        #             flipped_kp_name = 'right' + kp_name.split('left')[1]
        #             flipped_kp_indices.append(kp_dict[flipped_kp_name])
        #         elif kp_name.startswith('right'):
        #             flipped_kp_name = 'left' + kp_name.split('right')[1]
        #             flipped_kp_indices.append(kp_dict[flipped_kp_name])
        #         else:
        #             flipped_kp_indices.append(kp_dict[kp_name])
        #     random_flip_left_right_fn = functools.partial(
        #         random_flip_left_right,
        #         flipped_keypoint_indices=flipped_kp_indices)
        #     dataset = dataset.map(
        #         random_flip_left_right_fn,
        #         num_parallel_calls=train_cfg.num_parallel_map_calls
        #     )
        #     dataset = dataset.prefetch(train_cfg.prefetch_size)

        if aug_cfg['random_brightness']:
            dataset = dataset.map(
                random_brightness,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_contrast']:
            dataset = dataset.map(
                random_contrast,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_hue']:
            dataset = dataset.map(
                random_hue,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_gaussian_noise']:
            dataset = dataset.map(
                random_gaussian_noise,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        if aug_cfg['random_rotate']:
            dataset = dataset.map(
                random_rotate,
                num_parallel_calls=train_cfg.num_parallel_map_calls
            )
            dataset = dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def preprocess_data(self, dataset, train_cfg):
        preprocess_cfg = train_cfg.preprocess
        img_size = preprocess_cfg['image_resize']
        resize_fn = functools.partial(
            resize,
            target_image_size=img_size)
        dataset = dataset.map(
            resize_fn,
            num_parallel_calls=train_cfg.num_parallel_map_calls
        )
        dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def read_data(self, train_config):
        probs = self._get_probs()
        probs = tf.cast(probs, tf.float32)
        decoder = self._decoder()
        filenames = [ds['tfrecord_path'] for ds in self.datasets]
        file_ids = list(range(len(filenames)))
        dataset = tf.data.Dataset.from_tensor_slices((file_ids, filenames))
        dataset = dataset.apply(tf.contrib.data.rejection_resample(
            class_func=lambda c, _: c,
            target_dist=probs,
            seed=42))
        dataset = dataset.map(lambda _, a: a[1])
        if train_config.shuffle:
            dataset = dataset.shuffle(
                train_config.filenames_shuffle_buffer_size)

        dataset = dataset.repeat(train_config.num_epochs or None)

        file_read_func = functools.partial(tf.data.TFRecordDataset,
                                           buffer_size=8 * 1000 * 1000)
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                file_read_func, cycle_length=train_config.num_readers,
                block_length=train_config.read_block_length, sloppy=True))
        if train_config.shuffle:
            dataset = dataset.shuffle(train_config.shuffle_buffer_size)

        decode_fn = functools.partial(
            decoder.decode, items=['image', 'bboxes', 'labels'])
        dataset = dataset.map(
            decode_fn, num_parallel_calls=train_config.num_parallel_map_calls)
        dataset = dataset.prefetch(train_config.prefetch_size)

        dataset = self.augment_data(dataset, train_config)

        # dataset = self.preprocess_data(dataset, train_config)
        return dataset
