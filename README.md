# Machine Learning Lab - Experiments

This repository contains implementations of 8 fundamental machine learning algorithms and concepts using scikit-learn and related Python libraries.

## Syllabus

### 1. Adaptive Linear Neuron (ADALINE)
Implement an Adaptive Linear Neuron in Python. ADALINE is a single-layer neural network that uses a linear activation function and learns using the Widrow-Hoff learning rule.

### 2. FIND-S Algorithm
Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a CSV file.

### 3. Candidate-Elimination Algorithm
For a given set of training data examples stored in a CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.

### 4. Decision Tree - ID3 Algorithm
Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.

### 5. Naïve Bayesian Classifier
Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a CSV file. Compute the accuracy of the classifier, considering few test data sets.

### 6. k-Nearest Neighbor (k-NN) - Iris Dataset
Write a program to implement k-Nearest Neighbor algorithm to classify the iris data set. Print both correct and wrong predictions.

### 7. Naïve Bayesian Classifier for Document Classification
Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Calculate the accuracy, precision, and recall for your data set.

### 8. Bayesian Network - Heart Disease Diagnosis
Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using the Heart Disease Dataset.

## Getting Started

### Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiments

Each experiment is available as a Jupyter notebook in the `experiments/` directory:

```bash
jupyter notebook experiments/1_adaline.ipynb
jupyter notebook experiments/2_finds_algorithm.ipynb
jupyter notebook experiments/3_candidate_elimination.ipynb
jupyter notebook experiments/4_id3_decision_tree.ipynb
jupyter notebook experiments/5_naive_bayes_classifier.ipynb
jupyter notebook experiments/6_knn_iris.ipynb
jupyter notebook experiments/7_naive_bayes_text_classification.ipynb
jupyter notebook experiments/8_bayesian_network_heart_disease.ipynb
```

Or launch Jupyter Lab:
```bash
jupyter lab
```

## Structure

```
.
├── README.md
├── requirements.txt
└── experiments/
    ├── 1_adaline.ipynb
    ├── 2_finds_algorithm.ipynb
    ├── 3_candidate_elimination.ipynb
    ├── 4_id3_decision_tree.ipynb
    ├── 5_naive_bayes_classifier.ipynb
    ├── 6_knn_iris.ipynb
    ├── 7_naive_bayes_text_classification.ipynb
    └── 8_bayesian_network_heart_disease.ipynb
```

## Dependencies

- scikit-learn: Machine learning library
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- jupyter: Interactive notebooks
- pgmpy: Bayesian network implementation

## Notes

- All experiments use scikit-learn (sklearn) for standard algorithms
- Sample datasets are either generated or obtained from sklearn.datasets
- Each notebook includes explanations, visualizations, and performance metrics
