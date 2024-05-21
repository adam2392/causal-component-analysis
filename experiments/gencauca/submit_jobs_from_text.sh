#!/bin/bash

# Read the parameters from the file
# Read the parameters from the file line by line
while IFS= read -r line; do
  # Call the Python script with the line as arguments
  echo $line;
  python run.py $line
  exit
done < parameters_cdrl.txt

# Call the Python script with the constructed arguments
# python your_python_script.py $args
