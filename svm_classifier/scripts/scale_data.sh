#!/bin/bash

# Define input and output file paths
INPUT_FILE="data/data.txt"
RANGE_FILE="data/range_file"
SCALED_FILE="data/scaled_data.txt"

# Run svm-scale to scale the data
echo "Scaling data in $INPUT_FILE..."
svm-scale -l -1 -u 1 -s "$RANGE_FILE" "$INPUT_FILE" > "$SCALED_FILE"

# Check if data has been saved
if [ $? -eq 0 ]; then
    echo "Data successfully scaled."
    echo "Range file saved at $RANGE_FILE."
    echo "Scaled data saved at $SCALED_FILE."
else
    echo "Error occurred during scaling."
    exit 1
fi