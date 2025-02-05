## Medical Dataset Feature Selection Using KNN and Optimization Algorithms

# Overview

This project implements feature selection for a medical dataset using k-Nearest Neighbors (k-NN) combined with heuristic optimization techniques like Hill Climbing and Variable Neighborhood Search (VNS). The goal is to improve classification accuracy by selecting the most relevant features.


    1. Loads a medical dataset and preprocesses it.
    2. Uses k-NN (k=3) for classification.
    3. Implements a fitness function to evaluate feature subsets.
    4. Uses Hill Climbing and VNS to optimize feature selection.
    5. Outputs the best-selected feature subset and classification accuracy.

# Installation

To set up the project, clone the repository and install the required dependencies:

git clone https://github.com/marij0na/Medical-Dataset-Feature-Selection-Using-KNN-and-Optimization-Algorithms.git

cd Medical-Dataset-Feature-Selection-Using-KNN-and-Optimization-Algorithms

pip install -r requirements.txt

# Usage

    Place your dataset file (medical_dataset.data) in the project directory.
    Run the script:

    python ScriptK-NN.py

    The script will:
        Load and preprocess the dataset.
        Apply feature selection using k-NN.
        Optimize the feature subset using heuristic algorithms.
        Output the best feature subset and its classification accuracy.

# File Structure

Medical-Dataset-Feature-Selection-Using-KNN-and-Optimization-Algorithms/

│── ScriptK-NN.py             # Main script for feature selection and classification

│── medical_dataset.data      # Medical dataset (input data)

│── requirements.txt          # Dependencies for the project

│── README.md                 # Project documentation (this file)

# Dependencies

This project requires the following Python libraries:

    scikit-learn
    numpy
    pandas

Install them using:

pip install -r requirements.txt

# Contribution

Feel free to fork this repository, create a branch, and submit a pull request if you’d like to contribute improvements.
License

This project is licensed under the MIT License.
