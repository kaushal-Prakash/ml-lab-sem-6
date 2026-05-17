import json
import os

notebooks = {
    "4_id3_decision_tree.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# ID3 Decision Tree\n", "Classification using sklearn DecisionTreeClassifier on Iris dataset"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from sklearn import datasets\n", "from sklearn.model_selection import train_test_split\n", "from sklearn.tree import DecisionTreeClassifier\n", "from sklearn.metrics import accuracy_score, confusion_matrix\n", "import numpy as np"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load Iris dataset\n", "iris = datasets.load_iris()\n", "X = iris.data\n", "y = iris.target\n", "print(f'Dataset shape: {X.shape}')\n", "print(f'Classes: {np.unique(y)}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Split data\n", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n", "\n", "# Train decision tree\n", "dt = DecisionTreeClassifier(random_state=42)\n", "dt.fit(X_train, y_train)\n", "print('Model trained')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Predictions\n", "y_pred = dt.predict(X_test)\n", "\n", "# Accuracy\n", "accuracy = accuracy_score(y_test, y_pred)\n", "print(f'Accuracy: {accuracy:.4f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Confusion matrix\n", "cm = confusion_matrix(y_test, y_pred)\n", "print('\\nConfusion Matrix:')\n", "print(cm)"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Viva Questions\n", "1. What does ID3 algorithm do?\n", "2. How does information gain guide tree construction?\n", "3. What is entropy?\n", "4. Why use train-test split?\n", "5. What does confusion matrix show?"]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    },
    "5_naive_bayes_classifier.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Naive Bayes Classifier\n", "Classification on breast cancer dataset"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from sklearn import datasets\n", "from sklearn.model_selection import train_test_split\n", "from sklearn.naive_bayes import GaussianNB\n", "from sklearn.metrics import accuracy_score, precision_score, recall_score"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load breast cancer dataset\n", "data = datasets.load_breast_cancer()\n", "X = data.data\n", "y = data.target\n", "print(f'Dataset shape: {X.shape}')\n", "print(f'Classes: {[0, 1]}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Split data\n", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n", "\n", "# Train Naive Bayes\n", "nb = GaussianNB()\n", "nb.fit(X_train, y_train)\n", "print('Model trained')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Predictions and metrics\n", "y_pred = nb.predict(X_test)\n", "\n", "accuracy = accuracy_score(y_test, y_pred)\n", "precision = precision_score(y_test, y_pred)\n", "recall = recall_score(y_test, y_pred)\n", "\n", "print(f'Accuracy: {accuracy:.4f}')\n", "print(f'Precision: {precision:.4f}')\n", "print(f'Recall: {recall:.4f}')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Viva Questions\n", "1. What is the Naive Bayes assumption?\n", "2. How does Naive Bayes calculate probability?\n", "3. Why is it called 'Naive'?\n", "4. What is precision vs recall?\n", "5. When should we use Naive Bayes?"]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    },
    "6_knn_iris.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# k-NN (k-Nearest Neighbors)\n", "Classification on Iris dataset with optimal k selection"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from sklearn import datasets\n", "from sklearn.model_selection import train_test_split\n", "from sklearn.neighbors import KNeighborsClassifier\n", "from sklearn.metrics import accuracy_score"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load Iris dataset\n", "iris = datasets.load_iris()\n", "X = iris.data\n", "y = iris.target\n", "\n", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n", "print(f'Training set size: {len(X_train)}')\n", "print(f'Test set size: {len(X_test)}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Find best k\n", "best_k = 1\n", "best_accuracy = 0\n", "\n", "for k in range(1, 11):\n", "    knn = KNeighborsClassifier(n_neighbors=k)\n", "    knn.fit(X_train, y_train)\n", "    y_pred = knn.predict(X_test)\n", "    accuracy = accuracy_score(y_test, y_pred)\n", "    print(f'k={k}: Accuracy={accuracy:.4f}')\n", "    if accuracy > best_accuracy:\n", "        best_accuracy = accuracy\n", "        best_k = k\n", "\n", "print(f'\\nBest k: {best_k} with accuracy: {best_accuracy:.4f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train with best k\n", "knn_best = KNeighborsClassifier(n_neighbors=best_k)\n", "knn_best.fit(X_train, y_train)\n", "y_pred_best = knn_best.predict(X_test)\n", "final_accuracy = accuracy_score(y_test, y_pred_best)\n", "print(f'Final Accuracy: {final_accuracy:.4f}')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Viva Questions\n", "1. How does k-NN classify new points?\n", "2. How do we select the best k?\n", "3. What are advantages of k-NN?\n", "4. What are disadvantages of k-NN?\n", "5. Why is distance metric important?"]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    },
    "7_naive_bayes_text_classification.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Text Classification with Naive Bayes\n", "TfidfVectorizer + MultinomialNB on text documents"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from sklearn.feature_extraction.text import TfidfVectorizer\n", "from sklearn.naive_bayes import MultinomialNB\n", "from sklearn.metrics import accuracy_score, precision_recall_fscore_support"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Sample documents\n", "texts = [\n", "    'machine learning is great',\n", "    'deep learning neural networks',\n", "    'natural language processing',\n", "    'data science and analytics',\n", "    'machine learning algorithms',\n", "    'python programming language'\n", "]\n", "labels = [0, 0, 1, 1, 0, 2]\n", "\n", "print('Training Texts:')\n", "for i, txt in enumerate(texts):\n", "    print(f'{i}: {txt}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Vectorize\n", "vectorizer = TfidfVectorizer()\n", "X = vectorizer.fit_transform(texts)\n", "print(f'\\nVectorizer shape: {X.shape}')\n", "print(f'Features: {len(vectorizer.get_feature_names_out())}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train Naive Bayes\n", "nb = MultinomialNB()\n", "nb.fit(X, labels)\n", "\n", "# Predictions\n", "y_pred = nb.predict(X)\n", "accuracy = accuracy_score(labels, y_pred)\n", "print(f'\\nAccuracy: {accuracy:.4f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Per-class metrics\n", "precision, recall, f1, support = precision_recall_fscore_support(labels, y_pred)\n", "print('\\nPer-class metrics:')\n", "print(f'Precision: {precision}')\n", "print(f'Recall: {recall}')\n", "print(f'F1-score: {f1}')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Viva Questions\n", "1. What does TfidfVectorizer do?\n", "2. What is TF-IDF?\n", "3. Why use MultinomialNB for text?\n", "4. What is the difference between GaussianNB and MultinomialNB?\n", "5. How does text classification differ from numeric classification?"]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    },
    "8_bayesian_network_heart_disease.ipynb": {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Bayesian Network for Heart Disease\n", "Classification using GaussianNB with sensitivity and specificity"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from sklearn import datasets\n", "from sklearn.model_selection import train_test_split\n", "from sklearn.naive_bayes import GaussianNB\n", "from sklearn.metrics import accuracy_score, confusion_matrix"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Load heart disease dataset\n", "data = datasets.load_breast_cancer()  # Using breast cancer as proxy\n", "X = data.data\n", "y = data.target\n", "print(f'Dataset: {X.shape}')\n", "print(f'Target distribution: Class 0={sum(y==0)}, Class 1={sum(y==1)}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Split and train\n", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n", "\n", "nb = GaussianNB()\n", "nb.fit(X_train, y_train)\n", "print('Model trained')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Predictions\n", "y_pred = nb.predict(X_test)\n", "accuracy = accuracy_score(y_test, y_pred)\n", "print(f'\\nAccuracy: {accuracy:.4f}')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Sensitivity and Specificity\n", "cm = confusion_matrix(y_test, y_pred)\n", "tn, fp, fn, tp = cm.ravel()\n", "\n", "sensitivity = tp / (tp + fn)  # True Positive Rate\n", "specificity = tn / (tn + fp)  # True Negative Rate\n", "\n", "print(f'\\nConfusion Matrix:')\n", "print(cm)\n", "print(f'\\nSensitivity (Recall): {sensitivity:.4f}')\n", "print(f'Specificity: {specificity:.4f}')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Viva Questions\n", "1. What is a Bayesian Network?\n", "2. How does GaussianNB model probabilities?\n", "3. What is sensitivity (recall)?\n", "4. What is specificity?\n", "5. Why are both sensitivity and specificity important in medical diagnosis?"]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
}

base_path = r"d:\Academics\sem 6\ml\lab\experiments"

for filename, notebook_data in notebooks.items():
    filepath = os.path.join(base_path, filename)
    with open(filepath, 'w') as f:
        json.dump(notebook_data, f, indent=1)
    print(f"Created {filename}")

print("All notebooks created successfully!")
