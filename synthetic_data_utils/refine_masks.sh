#!/bin/bash

RAW_DIR=/media/storage/datasets/products/MasterCard/raw_final
OUT_DIR=/media/storage/datasets/products/MasterCard/out_final/objects

python3 refine_masks.py $RAW_DIR $OUT_DIR
