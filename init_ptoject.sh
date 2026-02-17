#!/bin/bash

PROJECT_NAME="penny"

echo "Creating project structure: $PROJECT_NAME"

# Create root directory
#mkdir -p $PROJECT_NAME

#cd $PROJECT_NAME || exit

# Create main files
touch README.md
touch requirements.txt
touch dataset.py

# Create src directory and files
mkdir -p src
touch src/config.py
touch src/data.py
touch src/features.py
touch src/model.py
touch src/viz.py

# Create models directory and placeholder model file
mkdir -p models
touch models/model.joblib

echo "Project structure created successfully."

echo ""
echo "Structure:"
tree .
