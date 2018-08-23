#!/bin/bash

RAW_DIR=/media/storage/datasets/products/MasterCard/raw_final_cardboard
OUT_DIR=/media/storage/datasets/products/MasterCard/out_final_cardboard/objects

python3 refine_masks.py $RAW_DIR $OUT_DIR --overwrite --scale .35 --glob_string "78" --number_of_workers 12 --crop 100 200 1600 2000 --saturate --reflective #--reduce_red
#--crop 150 600 1500 2400 --saturate #--reflective #--reduce_red
#--warm
# python3 refine_masks.py $RAW_DIR $OUT_DIR --flip --crop 480 910 860 1290 --glob_string "34"

# python3 refine_masks.py $RAW_DIR $OUT_DIR --skip_contours --glob_string "57"
