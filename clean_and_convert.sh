#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 path/to/notebook.ipynb"
    exit 1
}

# Check if a notebook path is provided
if [ -z "$1" ]; then
    echo "Error: No notebook path provided."
    usage
fi

NOTEBOOK_PATH="$1"

# Check if the notebook file exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: File '$NOTEBOOK_PATH' not found."
    exit 1
fi

# Clear the notebook outputs using nbconvert
echo "Clearing the notebook outputs..."
jupyter nbconvert --clear-output --inplace "$NOTEBOOK_PATH"

# Convert the notebook to a Python script using nbconvert
echo "Converting the notebook to a Python script..."
jupyter nbconvert --to script "$NOTEBOOK_PATH"

echo "Notebook outputs cleared and converted successfully."
