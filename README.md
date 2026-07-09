# Iris Species Classification

A Decision Tree classifier that distinguishes between three Iris species (*setosa*, *versicolor*, *virginica*) using the classic Iris dataset, with full visual interpretation of model behavior.

## Overview
This project trains and evaluates a `DecisionTreeClassifier` on the 150-sample Iris dataset (4 features: sepal length/width, petal length/width), then visualizes exactly how the model makes its decisions.

## Results
- **Test Accuracy:** 96.7%
- **Train/Test Split:** 80/20, stratified by species

## What's Included
- `iris_classifier.py` — data loading, train/test split, model training, evaluation, and visualization generation
- `confusion_matrix.png` — per-class prediction accuracy breakdown
- `decision_boundary.png` — 2D visualization (petal length vs. petal width) of the regions the model assigns to each species
- `decision_tree.png` — full visual structure of the trained decision tree, showing every split condition

## Tech Stack
- Python
- scikit-learn (`DecisionTreeClassifier`, `train_test_split`, `classification_report`)
- pandas
- Matplotlib

## How to Run
```bash
pip install scikit-learn pandas matplotlib numpy
python iris_classifier.py
```

## Key Insight
The model separates *setosa* perfectly using a single feature (petal length ≤ 2.45 cm), while distinguishing *versicolor* from *virginica* requires deeper splits on petal width and length — visible directly in `decision_tree.png`.
