#!/bin/bash

RAW_DIR=/media/storage/datasets/products/MasterCard/raw_final
OUT_DIR=/media/storage/datasets/products/MasterCard/out_final/objects

python3 refine_masks.py $RAW_DIR $OUT_DIR --overwrite --scale .35 --glob_string "155" --number_of_workers 12
#--reflective
# python3 refine_masks.py $RAW_DIR $OUT_DIR --flip --crop 480 910 860 1290 --glob_string "34"

# python3 refine_masks.py $RAW_DIR $OUT_DIR --skip_contours --glob_string "57"
