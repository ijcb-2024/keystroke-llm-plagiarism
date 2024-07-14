#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRAIN_DIR="$SCRIPT_DIR/../train"
FILEPATH="$TRAIN_DIR/user_specific.py"
DATASET_ROOT="$SCRIPT_DIR/../../Datasets"

# Default values for hyperparameters
learning_rate=0.001
batch_size=512
num_epochs=1
seq_len=100
gpu_device=5
dropout=0.5
dataset_root=$DATASET_ROOT

# Call the main python script with the parsed arguments
python "$FILEPATH" \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --num_epochs $num_epochs \
  --seq_len $seq_len \
  --gpu_device $gpu_device \
  --dropout $dropout \
  --dataset_root $dataset_root
