#!/bin/bash

INPUT_DIR=/media/storage/datasets/products/TrainData/objects
OUTPUT_DIR=/media/storage/datasets/products/TrainData/train_data/sample_010

python3 dataset_generator.py $INPUT_DIR $OUTPUT_DIR
