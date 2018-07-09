import tensorflow as tf
import numpy as np
import cv2


def random_int(maxval, minval=0):
    return tf.random_uniform(
        shape=[], minval=minval, maxval=maxval, dtype=tf.int32)


def normalize_bboxes(bboxes, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4, axis=1)
    return tf.concat([ymin / img_h, xmin / img_w, ymax / img_h, xmax / img_w], 1)


def normalize_keypoints(keypoints, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=2)
    v = tf.minimum(v, 1)
    return tf.concat([x / img_w, y / img_h, v], 2)


def keypoints_select(img, keypoints, bboxes, mask, keypoints_to_keep):
    keypoints_subset = tf.gather(keypoints,
                                 keypoints_to_keep,
                                 axis=1)
    return img, keypoints_subset, bboxes, mask


def flip_left_right_keypoints(keypoints, flipped_keypoint_indices):
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=2)
    flipped_keypoints = tf.concat([1. - x, y, v], 2)
    flipped_keypoints = tf.gather(flipped_keypoints,
                                  flipped_keypoint_indices,
                                  axis=1)
    return flipped_keypoints


def flip_left_right_bboxes(bboxes):
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                      axis=1)
    return tf.concat([ymin, 1. - xmax, ymax, 1. - xmin], 1)


def rotate_bboxes(bboxes, k):
    """rotate in multiples of 90 degrees"""
    y1, x1, y2, x2 = tf.split(value=bboxes, num_or_size_splits=4,
                              axis=1)
    rot_0 = lambda: bboxes
    rot_90 = lambda: tf.concat([1. - x2, y1, 1. - x1, y2], 1)
    rot_180 = lambda: tf.concat([1. - y2, 1. - x2, 1. - y1, 1. - x1], 1)
    rot_270 = lambda: tf.concat([x1, 1. - y2, x2, 1. - y1], 1)

    rot_bboxes = tf.case(
        {tf.equal(k, 0): rot_0,
         tf.equal(k, 1): rot_90,
         tf.equal(k, 2): rot_180,
         tf.equal(k, 3): rot_270},
        default=rot_0, exclusive=True)
    return rot_bboxes


def random_rotate(image, bboxes, labels):
    k = random_int(4)
    image = tf.image.rot90(image, k)
    bboxes = rotate_bboxes(bboxes, k)
    return image, bboxes, labels


def random_flip_left_right(img, keypoints, bboxes, mask,
                           flipped_keypoint_indices):
    random_var = random_int(2)
    random_var = tf.cast(random_var, tf.bool)
    flipped_img = tf.cond(random_var,
                          true_fn=lambda: tf.image.flip_left_right(img),
                          false_fn=lambda: tf.identity(img))
    mask = tf.expand_dims(mask, axis=2)
    flipped_mask = tf.cond(random_var,
                           true_fn=lambda: tf.image.flip_left_right(mask),
                           false_fn=lambda: tf.identity(mask))
    flipped_mask = tf.squeeze(flipped_mask)
    flipped_keypoints = tf.cond(
        random_var,
        true_fn=lambda: flip_left_right_keypoints(
            keypoints, flipped_keypoint_indices),
        false_fn=lambda: tf.identity(keypoints))
    flipped_bbox = tf.cond(
        random_var,
        true_fn=lambda: flip_left_right_bboxes(bboxes),
        false_fn=lambda: tf.identity(bboxes))
    return flipped_img, flipped_keypoints, flipped_bbox, flipped_mask


def prune_bboxes_keypoints(bboxes, keypoints, crop_box):
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                      axis=1)
    crop_ymin, crop_xmin, crop_ymax, crop_xmax = tf.unstack(crop_box)
    crop_h, crop_w = crop_ymax - crop_ymin, crop_xmax - crop_xmin
    ymin, xmin = (ymin - crop_ymin) / crop_h, (xmin - crop_xmin) / crop_w
    ymax, xmax = (ymax - crop_ymin) / crop_h, (xmax - crop_xmin) / crop_w
    is_outside = tf.concat([
        tf.greater(ymin, 1.), tf.greater(xmin, 1.),
        tf.less(ymax, 0.), tf.less(xmax, 0.)
    ], 1)
    is_outside = tf.reduce_any(is_outside, 1)
    valid_indices = tf.reshape(tf.where(tf.logical_not(is_outside)), [-1])
    valid_bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
    valid_bboxes = tf.gather(valid_bboxes, valid_indices)
    valid_bboxes = tf.clip_by_value(valid_bboxes,
                                    clip_value_min=0.,
                                    clip_value_max=1.)
    valid_keypoints = tf.gather(keypoints, valid_indices)
    x, y, v = tf.split(value=valid_keypoints, num_or_size_splits=3, axis=2)
    x, y = (x - crop_xmin) / crop_w, (y - crop_ymin) / crop_h
    is_outside = tf.concat([
        tf.greater_equal(x, 1.), tf.greater_equal(y, 1.),
        tf.less_equal(x, 0.), tf.less_equal(y, 0.)
    ], 2)
    is_outside = tf.reduce_any(is_outside, 2)
    is_outside = tf.cast(tf.logical_not(is_outside), tf.float32)
    v = v * tf.expand_dims(is_outside, 2)
    valid_keypoints = tf.concat([x, y, v], axis=2)
    return valid_bboxes, valid_keypoints


def random_gaussian_noise(image, bboxes, labels, std=0.015):
    noise = tf.random_normal(shape=tf.shape(image), mean=0., stddev=std)
    new_image = tf.to_float(image) + 255. * noise
    new_image = tf.clip_by_value(new_image,
                                 clip_value_min=0,
                                 clip_value_max=255)
    return tf.cast(new_image, tf.uint8), bboxes, labels


def random_brightness(image, bboxes, labels):
    image = tf.image.random_brightness(
        image,
        max_delta=0.25)
    return image, bboxes, labels


def random_contrast(image, bboxes, labels):
    image = tf.image.random_contrast(
        image,
        lower=0.75,
        upper=1.25)
    return image, bboxes, labels


def random_hue(image, bboxes, labels):
    image = tf.image.random_hue(
        image,
        max_delta=0.025)
    return image, bboxes, labels


def resize(image, keypoints, bbox, mask,
           target_image_size=(224, 224),
           target_mask_size=None):
    img_size = list(target_image_size)
    if target_mask_size is None:
        target_mask_size = img_size
    mask_size = list(target_mask_size)
    new_image = tf.image.resize_images(image, size=img_size)
    new_mask = tf.expand_dims(mask, axis=2)
    new_mask.set_shape([None, None, 1])
    new_mask = tf.image.resize_images(new_mask, size=mask_size)
    new_mask = tf.squeeze(new_mask)
    return new_image, keypoints, bbox, new_mask


def rotate(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # c = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    img[y1:y2, x1:x2, :] = (
        np.expand_dims(alpha, 2) * img_overlay[y1o:y2o, x1o:x2o, :] +
        np.expand_dims(alpha_inv, 2) * img[y1:y2, x1:x2, :])
    return img
