#!/bin/bash

INPUT_DIR=/media/storage/datasets/products/MasterCard/out_final/objects
OUTPUT_DIR=/media/storage/datasets/products/TrainData/train_data/sample_006

python3 dataset_generator.py $INPUT_DIR $OUTPUT_DIR --dontocclude --add_distractors --selected
