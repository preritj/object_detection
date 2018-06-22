import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.dataset_util import (
    rotate, overlay_image_alpha, random_crop,
    random_flip_left_right, keypoints_select, resize)


class ObjectDataReader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        bkg_images = glob(os.path.join(cfg.bkg_data_dir, "*"))
        self.bkg_images = [b for b in bkg_images if b.lower().endswith('.jpg')
                           or b.lower().endswith('.jpeg')]
        product_dirs = glob(os.path.join(cfg.obj_data_dir, "*"))
        self.product_names = {i + 1: os.path.basename(p)
                              for i, p in enumerate(product_dirs)}
        self.images, self.labels = [], []
        for i, prod_dir in enumerate(product_dirs):
            prod_images = glob(os.path.join(product_dirs[i], "*"))
            prod_images = [p for p in prod_images if p.lower().endswith('.png')]
            self.labels += len(prod_images) * [i + 1]
            self.images += prod_images
        self.n_samples = len(self.labels)
        self.shuffled_images, self.shuffled_labels = None, None
        self.reset()

    def reset(self):
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        self.shuffled_images = np.array(self.images)[idx]
        self.shuffled_labels = np.array(self.labels)[idx]

    def generator(self):
        i_end = 0
        while True:
            i_start = i_end
            n1 = self.cfg.n_items_min
            n2 = self.cfg.n_items_max
            n = np.random.randint(n1, n2 + 1)
            i_end = i_start + n
            if i_end > self.n_samples:
                self.reset()
                i_start = 0
                i_end = n
            yield i_start, n

    def generate_image_and_labels(self, i_start, n):
        img_h, img_w = self.cfg.image_shape[0], self.cfg.image_shape[1]
        ncols = np.ceil(np.sqrt(n))
        nrows = np.ceil(n / ncols)
        prods = self.shuffled_images[i_start:i_start + n]
        labels = self.shuffled_labels[i_start:i_start + n]
        bkg_img = np.random.choice(self.bkg_images)
        image = plt.imread(bkg_img)
        h, w, _ = image.shape
        scale = max(1.5 * img_h / h, 1.5 * img_w / w)
        scale = np.random.uniform(scale, 3. * scale)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        h, w, _ = image.shape
        top = np.random.randint(0, h - img_h)
        left = np.random.randint(0, w - img_w)
        image = image[top: top + img_h, left: left + img_w]
        flip_h, flip_v = np.random.randint(0, 1, 2)
        if flip_h:
            image = cv2.flip(image, 0)
        if flip_v:
            image = cv2.flip(image, 1)
        if np.max(image) > 1.5:
            image = image / 255.
        color = np.ones((img_h, img_w, 3)) * np.random.uniform(0, 1., 3)
        image = cv2.addWeighted(image, 0.7, color, 0.3, 0)
        img_h, img_w, _ = image.shape
        indices = np.arange(n)
        np.random.shuffle(indices)
        bboxes, classes = [], []
        for i in indices:
            img = plt.imread(prods[i])
            label = labels[i]
            mask = img[:, :, 3]
            dims = self.cfg.product_dims[
                self.cfg.category[self.product_names[label]]]
            y_idx, x_idx = np.where(mask > 0.5)
            y1, x1 = np.min(y_idx), np.min(x_idx)
            y2, x2 = np.max(y_idx), np.max(x_idx)
            img = img[y1:y2, x1:x2, :]
            h, w, _ = img.shape
            volume = dims[2] * dims[0] * dims[1]
            scale = img_h / np.random.uniform(18, 24)
            scale *= (volume / (h * w * w)) ** (1. / 3)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            img = rotate(img, np.random.uniform(-180, 180))
            mask = img[:, :, 3]
            h, w, _ = img.shape
            col = i % ncols
            row = i // ncols
            delta_x, delta_y = np.random.uniform(0.2, 0.8, 2)
            x = (delta_x + col) * img_w / ncols - 0.5 * w
            y = (delta_y + row) * img_h / nrows - 0.5 * h
            y_idx, x_idx = np.where(mask > 0.9)
            y1, x1 = np.min(y_idx), np.min(x_idx)
            y2, x2 = np.max(y_idx), np.max(x_idx)
            classes.append(label)
            bboxes.append([(y + y1) / img_h, (x + x1) / img_w,
                           (y + y2) / img_h, (x + x2) / img_w])
            pos = (int(x), int(y))
            image = overlay_image_alpha(image, img[:, :, :3], pos, mask)
            # if np.random.uniform() < 0.15:
            #     x1, y1 = 0., np.random.uniform()
            #     x2, y2 = 1., np.random.uniform()
            #     x1, y1 = int(x1 * img_w), int(y1 * img_h)
            #     x2, y2 = int(x2 * img_w), int(y2 * img_h)
            #     mask = np.zeros((img_h, img_w))
            #     thickness = np.random.randint(5, 25)
            #     mask = cv2.line(mask, (x1, y1), (x2, y2), (1.), thickness)
            #     foreground_img = np.random.choice(self.bkg_images)
            #     foreground_img = plt.imread(foreground_img)
            #     foreground_img = foreground_img[:img_h, :img_w]
            #     color = np.ones((img_h, img_w, 3)) * np.random.uniform(0, 1., 3)
            #     if np.max(foreground_img) > 1.5:
            #         foreground_img = foreground_img / 255.
            #     foreground_img = cv2.addWeighted(foreground_img, 0.3, color, 0.7, 0)
            #     image = overlay_image_alpha(image, foreground_img, (0, 0), mask)
        #                 color = np.random.uniform(0., 1., 3)
        #                 thickness = np.random.randint(5, 25)
        #                 image = cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        #             if np.random.uniform() < 0.1:
        #                 hand_idx = np.random.randint(len(hand_images))
        #                 hand_img_file = hand_images[hand_idx]
        #                 img = plt.imread(hand_img_file)
        #                 h, w, _ = img.shape
        #                 scale = img_h / max(h, w)
        #                 if np.max(img) > 1.001:
        #                     img = img / 255.
        #                 if not hand_img_file.lower().endswith('png'):
        #                     new_img = np.zeros((h, w, 4))
        #                     new_img[:, :, :3] = img
        #                     new_img[:, :, 3] = np.prod(img, axis=2) < .9
        #                     img = new_img
        #                 img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #                 img = rotate(img, np.random.uniform(-180, 180))
        #                 mask = img[:, :, 3]
        #                 h, w, _ = img.shape
        #                 x, y = np.random.uniform(-0.2, 0.2, 2)
        #                 pos = (int(x * img_w), int(y * img_h))
        #                 image = overlay_image_alpha(image, img[:, :, :3], pos, mask)
        blur = 2 * np.random.randint(0, 2) + 1
        image = cv2.GaussianBlur(image, (blur, blur), 0)
        image = np.clip(image, 0., 1.)
        bboxes = np.clip(bboxes, 0., 1.)
        return image.astype(np.float32), bboxes.astype(np.float32), labels.astype(np.int32)

    def read_data(self, train_config):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.int32, tf.int32))

        def py_fn(i_start, n):
            return tuple(tf.py_func(
                func=self.generate_image_and_labels,
                inp=[i_start, n],
                Tout=[tf.float32, tf.float32, tf.int32]))

        dataset = dataset.map(
            py_fn,
            num_parallel_calls=train_config.num_parallel_map_calls)
        return dataset
