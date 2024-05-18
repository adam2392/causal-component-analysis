#!/bin/bash

# Directory containing the Python script
SCRIPT_NAME="02-script-collidergraph.py"

# Change to the directory containing the script
# cd "$SCRIPT_DIR"

# Loop over the training seeds and submit a job for each seed
for TRAINING_SEED in $(seq 1 10000 1000)
do
  # Construct the command to run the Python script with the current training seed
  CMD="python3 $SCRIPT_NAME --training-seed $TRAINING_SEED"
  
  # Optionally, you can use a job scheduler like `nohup` to run the command in the background
  # or `&` to run the command in the background
  nohup $CMD > output_seed_$TRAINING_SEED.log 2>&1 &
  
  echo "Submitted job for training seed: $TRAINING_SEED"
done
