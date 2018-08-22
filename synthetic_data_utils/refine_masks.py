import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from bilateral_solver import apply_bilateral
from multiprocessing import Pool
from functools import partial
import signal
from scipy.interpolate import UnivariateSpline


def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def warm_image(image):
    incr_ch_lut0 = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 70, 140, 210, 256])
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 84, 168, 224, 256])
    # decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
    #                               [0, 30, 80, 120, 192])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 38, 76, 120, 192])
    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_g = cv2.LUT(c_g, incr_ch_lut0).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    # c_r = np.clip(1.2 * c_r - 10, 0, 255).astype(np.uint8)
    # c_g = np.clip(1.4 * c_g - 20, 0, 255).astype(np.uint8)
    # c_b = np.clip(0.85 * c_b, 0, 255).astype(np.uint8)
    image_warm = cv2.merge((c_b, c_g, c_r))

    # c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)

    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(image_warm,
                                           cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut0).astype(np.uint8)
    # c_s = np.clip(1.4 * c_s, 0, 255).astype(np.uint8)

    image_warm = cv2.cvtColor(cv2.merge(
        (c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
    return image_warm


def increase_saturation(image):
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 96, 192, 236, 256])
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(image,
                                           cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)


def reduce_red_in_image(image):
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 90, 180, 230, 256])
    decr_ch_lut0 = create_LUT_8UC1([0, 64, 128, 192, 256],
                                   [0, 48, 100, 188, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
                                  [0, 40, 90, 180, 256])
    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    c_g = cv2.LUT(c_g, decr_ch_lut0).astype(np.uint8)
    # # c_r = np.clip(1.2 * c_r - 10, 0, 255).astype(np.uint8)
    # # c_g = np.clip(1.4 * c_g - 20, 0, 255).astype(np.uint8)
    # # c_b = np.clip(0.85 * c_b, 0, 255).astype(np.uint8)
    # c_r = np.clip(0.5 * c_r, 0, 255).astype(np.uint8)
    image = cv2.merge((c_b, c_g, c_r))

    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(image,
                                           cv2.COLOR_BGR2HSV))
    c_v = cv2.LUT(c_v, incr_ch_lut).astype(np.uint8)
    # c_s = np.clip(1.4 * c_s, 0, 255).astype(np.uint8)

    image = cv2.cvtColor(cv2.merge(
        (c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
    return image


def get_mask_v0(filename, crop=None, scale=1, flip=False,
             skip_contours=False, skip_bilateral=False):
    img_mask = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if crop is not None:
        y1, x1, y2, x2 = crop
        img_mask = img_mask[y1:y2, x1:x2]
    if scale != 1:
        img_mask = cv2.resize(img_mask, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
    if flip:
        img_mask = cv2.flip(img_mask, 1)
    img = img_mask[:, :, :3]  # BGR
    score = img_mask[:, :, 3] / 255.
    mask = np.copy(score)
    # remove shadows from mask
    mask[mask < 0.98] = 0
    mask[mask >= 0.98] = 1
    # remove noise
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # cook up some scores
    score[score < 0.3] = .8  # background gets high score
    score[score > 0.7] = 1.  # foreground gets high score
    score[(score >= 0.3) & (score <= 0.7)] = 0.4   # low score elsewhere
    score = cv2.GaussianBlur(score, (11, 11), 11, 11)
    if not skip_contours:
        mask = (255 * mask).astype(np.uint8)
        # fill the holes in mask
        _, contour, _ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
        mask = (mask / 255).astype(np.uint8)
    if not skip_bilateral:
        # apply bilateral solver
        mask = apply_bilateral(img, mask, score, thresh=.7)
    return img, mask


def get_approx_mask(image, bkg_median, threshold=30):
    h, w = image.shape[:2]
    b = cv2.inRange(image, bkg_median - threshold, bkg_median + threshold)
    b = (b / 255).astype(np.uint8)
    kernel = np.ones((5, 5),np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    fill_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(b, fill_mask, (0, 0), 1)
    fill_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(b, fill_mask, (w-1, 0), 1)
    mask = 1 - b
    fill_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(b, fill_mask, (0, 0), 0)
    mask = cv2.bitwise_or(mask, b)
    return mask


def get_mask(filename, crop=None, scale=1, flip=False, reflective=False,
             warm=False, reduce_red=False, saturate=False):
    img = cv2.imread(filename)
    if crop is not None:
        y1, x1, y2, x2 = crop
        img = img[y1:y2, x1:x2]
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    if flip:
        img = cv2.flip(img, 1)
    if reduce_red:
        img = reduce_red_in_image(img)
    if saturate:
        img = increase_saturation(img)

    h, w = img.shape[:2]
    # extract background color from bottom
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ab = img_lab[:, :, 1:]
    bkg_patch = ab[int(0.8 * h):, :]
    ab_median = np.median(bkg_patch, axis=(0, 1)).astype(np.uint8)
    mask = cv2.GC_PR_BGD * np.ones(img.shape[:2], np.uint8)

    # get masks with varying thresholds
    thresh1 = 20 if reflective else 40  # 40
    thresh2 = 10 if reflective else 25  # 25
    obj_mask = get_approx_mask(ab, ab_median, threshold=thresh1)
    pr_obj_mask = get_approx_mask(ab, ab_median, threshold=thresh2)
    pr_bkg_mask = 1 - get_approx_mask(ab, ab_median, threshold=4)
    bkg_mask = 1 - get_approx_mask(ab, ab_median, threshold=3)

    # Run grabcut algorithm
    mask[pr_bkg_mask > 0] = cv2.GC_PR_BGD
    mask[bkg_mask > 0] = cv2.GC_BGD
    mask[pr_obj_mask > 0] = cv2.GC_PR_FGD
    mask[obj_mask > 0] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel,
                                           fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask = (255 * mask).astype(np.uint8)
    kernel = np.ones((13, 13), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = (mask / 255).astype(np.uint8)

    # mask = pr_obj_mask
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # score = 0.2 * np.ones_like(mask, dtype=np.float32)
    # # score[pr_obj_mask > 0] = .8
    # score[obj_mask > 0] = 1.
    # score[bkg_mask > 0] = 1.
    # mask = apply_bilateral(img, mask, score, thresh=.7)
    if warm:
        img = warm_image(img)
    return img, mask


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def create_mask_wrapper(args, crop=None, scale=1, flip=False,
                        reflective=False, warm=False, reduce_red=False,
                        saturate=False):
    """Wrapper used to pass params to workers"""
    input_filename, output_filename = args[0], args[1]
    try:
        img, mask = get_mask(input_filename, crop, scale, flip, reflective,
                             warm, reduce_red, saturate)
    except:
        return
    mask = np.expand_dims(mask, axis=2)
    mask = (255 * mask).astype(np.uint8)
    img = np.concatenate((img, mask), axis=2)
    cv2.imwrite(output_filename, img)


def refine_masks(args_):
    prods = glob(os.path.join(args_.raw_dir, args_.glob_string))
    crop = args_.crop
    if np.any(np.array(crop) < 0):
        crop = None
    scale = args_.scale
    flip = args_.flip
    reflective = args_.reflective
    warm = args_.warm
    reduce_red = args_.reduce_red
    saturate = args_.saturate
    skip_contours = args_.skip_contours
    skip_bilateral = args_.skip_bilateral
    for p in prods:
        prod_name = os.path.basename(p)
        print("Refining product ", prod_name)
        out_prod_dir = os.path.join(args_.out_dir, prod_name)
        if os.path.exists(out_prod_dir):
            print("Product directory already exists:")
            if args_.overwrite:
                print("Overwriting.")
                files = glob(os.path.join(out_prod_dir, "*"))
                files.sort()
                for f in files:
                    os.remove(f)
            else:
                print("Skipping.")
                continue
        else:
            print("Writing in ", out_prod_dir)
            os.makedirs(out_prod_dir)
        prod_images = glob(os.path.join(p, "*.png"))
        params_list = []
        for i, img_file in tqdm(enumerate(prod_images)):
            out_file = os.path.join(out_prod_dir, str(i).zfill(5)) + '.png'
            params = (img_file, out_file)
            params_list.append(params)
        partial_func = partial(
            create_mask_wrapper,
            crop=crop, scale=scale, flip=flip, reflective=reflective,
            warm=warm, reduce_red=reduce_red, saturate=saturate)
        p = Pool(args_.number_of_workers, init_worker)
        try:
            p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()


def parse_args():
    """Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Refines raw masks")
    parser.add_argument(
        "raw_dir", help="The root directory which contains the raw images in png format")
    parser.add_argument(
        "out_dir", help="The directory where refined masks are created in png format")
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing files. Default is not overwrite.", action="store_true")
    parser.add_argument(
        "--flip",
        help="Flip image left-right. Default is not flip.", action="store_true")
    parser.add_argument(
        "--skip_contours",
        help="Skips contour post-processing step. Default is to apply contour processing.", action="store_true")
    parser.add_argument(
        "--skip_bilateral",
        help="Skips bilateral solver post-processing step. Default is to apply this processing.", action="store_true")
    parser.add_argument(
        "--reflective",
        help="Setting for reflective objects. Default is to not use this setting.", action="store_true")
    parser.add_argument(
        "--warm",
        help="Increase color temperature. Default is to not use this setting.", action="store_true")
    parser.add_argument(
        "--reduce_red",
        help="Decrease R in RGB image. Default is to not use this setting.", action="store_true")
    parser.add_argument(
        "--saturate",
        help="Increase saturation. Default is to not use this setting.", action="store_true")
    parser.add_argument(
        "--scale",
        help="Scaling applied to images. Defaults to 1.", default=1, type=float)
    parser.add_argument(
        '--crop', nargs=4, type=int,
        help="Initial crop to be applied to image (y1, x1, y2, x2).", default=[-1, -1, -1, -1])
    parser.add_argument(
        "--glob_string",
        help="Glob string to make selections inside raw_dir. Defaults to *", default="*")
    parser.add_argument(
        "--number_of_workers",
        help="Number of workers for pooling. Defaults to 4.", default=4, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    refine_masks(args)
