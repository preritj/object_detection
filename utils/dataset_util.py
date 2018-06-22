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


def random_crop(image, keypoints, bboxes, mask,
                crop_size=(224, 224), scale_range=(1.5, 4.)):
    bboxes = tf.clip_by_value(
        bboxes, clip_value_min=0.0, clip_value_max=1.0)
    n_bboxes = tf.shape(bboxes)[0]
    img_shape = tf.cast(tf.shape(image), tf.float32)
    img_h, img_w = img_shape[0], img_shape[1]
    random_bbox = tf.cond(
        tf.greater(n_bboxes, 0),
        true_fn=lambda: tf.random_shuffle(bboxes)[0],
        false_fn=lambda: tf.constant([0., 0., 1., 1.]))
    bbox_area = ((random_bbox[2] - random_bbox[0])
                 * (random_bbox[3] - random_bbox[1]))
    img_aspect_ratio = img_w / img_h
    aspect_ratio = 1. * crop_size[1] / crop_size[0]
    crop_aspect_ratio = tf.constant(aspect_ratio, tf.float32)

    def target_height_fn():
        return tf.to_int32(tf.round(img_w / crop_aspect_ratio))

    crop_h = tf.cond(img_aspect_ratio >= aspect_ratio,
                     true_fn=lambda: tf.to_int32(img_h),
                     false_fn=target_height_fn)

    def target_width_fn():
        return tf.to_int32(tf.round(img_h * crop_aspect_ratio))

    crop_w = tf.cond(img_aspect_ratio <= aspect_ratio,
                     true_fn=lambda: tf.to_int32(img_w),
                     false_fn=target_width_fn)

    max_crop_shape = tf.stack([crop_h, crop_w])

    crop_area = 1. * crop_size[0] * crop_size[1]
    scale_ratio = tf.sqrt(bbox_area * img_h * img_w / crop_area)

    crop_size = tf.constant(list(crop_size), tf.float32)
    # cap min scale at 0.5 to prevent bad resolution
    scale_min = tf.maximum(scale_range[0] * scale_ratio, 0.5)
    # max scale has to be greater than min scale (1.1 * min scale)
    scale_max = tf.maximum(scale_range[1] * scale_ratio,
                           1.1 * scale_min)
    size_min = tf.minimum(max_crop_shape - 1,
                          tf.to_int32(scale_min * crop_size))
    size_max = tf.minimum(max_crop_shape,
                          tf.to_int32(scale_max * crop_size))
    crop_h = random_int(maxval=tf.to_int32(size_max[0]),
                        minval=tf.to_int32(size_min[0]))
    crop_w = tf.to_int32(aspect_ratio * tf.to_float(crop_h))
    crop_shape = tf.stack([crop_h, crop_w])

    bbox_min, bbox_max = random_bbox[:2], random_bbox[2:]
    bbox_min = tf.cast(tf.round(bbox_min * img_shape[:2]), tf.int32)
    bbox_max = tf.cast(tf.round(bbox_max * img_shape[:2]), tf.int32)
    bbox_min = tf.maximum(bbox_min, 0)

    offset_min = tf.maximum(0, bbox_max - crop_shape)
    offset_max = tf.minimum(
        tf.cast(img_shape[:2], tf.int32) - crop_shape + 1,
        bbox_min + 1)
    offset_min = tf.where(tf.less_equal(offset_max, offset_min),
                          tf.constant([0, 0]),
                          offset_min)

    offset_h = random_int(maxval=offset_max[0], minval=offset_min[0])
    offset_w = random_int(maxval=offset_max[1], minval=offset_min[1])

    new_image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.expand_dims(mask, 2)
    new_mask = tf.image.crop_to_bounding_box(
        new_mask, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.squeeze(new_mask)
    crop_box = tf.stack([
        tf.to_float(offset_h) / img_h,
        tf.to_float(offset_w) / img_w,
        tf.to_float(offset_h + crop_h) / img_h,
        tf.to_float(offset_w + crop_w) / img_w
    ])
    new_bboxes, new_keypoints = prune_bboxes_keypoints(
        bboxes, keypoints, crop_box)
    return new_image, new_keypoints, new_bboxes, new_mask


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
