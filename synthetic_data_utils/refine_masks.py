import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from bilateral_solver import apply_bilateral


def get_mask(filename):
    img_mask = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
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
    mask = (255 * mask).astype(np.uint8)
    # fill the holes in mask
    _, contour, _ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    mask = (mask / 255).astype(np.uint8)
    # apply bilateral solver
    mask = apply_bilateral(img, mask, score, thresh=.7)
    return img, mask


def refine_masks(args_):
    prods = glob(os.path.join(args_.raw_dir, '*'))
    for p in prods:
        prod_name = os.path.basename(p)
        print("Refining product ", prod_name)
        out_prod_dir = os.path.join(args_.out_dir, prod_name)
        if os.path.exists(out_prod_dir):
            print("Product directory already exists:")
            if args_.overwrite:
                print("Overwriting.")
                files = glob(out_prod_dir).sort()
                for f in files:
                    os.remove(f)
            else:
                print("Skipping.")
                continue
        else:
            print("Writing in ", out_prod_dir)
            os.makedirs(out_prod_dir)
        prod_images = glob(os.path.join(p, "*.png"))
        for i, img_file in tqdm(enumerate(prod_images)):
            out_file = os.path.join(out_prod_dir, str(i).zfill(5)) + '.png'
            img, mask = get_mask(img_file)
            mask = np.expand_dims(mask, axis=2)
            mask = (255 * mask).astype(np.uint8)
            img = np.concatenate((img, mask), axis=2)
            cv2.imwrite(out_file, img)


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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    refine_masks(args)
