# 🩺 Diabetes Detector — Neural Network from Scratch

A neural network built using **pure NumPy** (no TensorFlow, no PyTorch) that predicts diabetes from patient data using matrix-based forward pass and backpropagation.

---

## 📌 Overview

This project implements a **3 → 3 → 1 neural network** from scratch to classify patients as diabetic or non-diabetic using the Pima Indians Diabetes Dataset from Kaggle. The entire training loop — including forward pass, gradient computation, and weight updates — is written manually using matrix operations.

---

## 🗂️ Project Structure

```
nerualnetwork_diabeties/
├── DiabetesDetector.ipynb   ← Main notebook
├── dataset/
│   └── diabetes.csv         ← Downloaded from Kaggle (see setup)
└── README.md
```

---

## 🧠 Network Architecture

```
Input Layer     Hidden Layer    Output Layer
   (3)      →      (3)      →      (1)
 Glucose        [H] [H] [H]      0 or 1
 BMI             ↑ Sigmoid        ↑ Sigmoid
 Age            W1 (3×3)         W2 (3×1)
```

| Layer | Nodes | Activation |
|-------|-------|------------|
| Input | 3 | — |
| Hidden | 3 | Sigmoid |
| Output | 1 | Sigmoid |

---

## 📦 Dataset

**Pima Indians Diabetes Database**  
Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Property | Value |
|----------|-------|
| Rows | 768 patients |
| Features | 8 (3 used) |
| Target | `Outcome` (0 = no diabetes, 1 = diabetic) |
| Missing values | None |

**Top 3 features selected by Pearson correlation:**

| Feature | Correlation | 
|---------|-------------|
| Glucose | 0.466 |
| BMI | 0.293 |
| Age | 0.238 |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Shreejal172/Neural-Network-Phishing-Detector
cd Neural-Network-Phishing-Detector
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Download `diabetes.csv`
- Place it inside a `dataset/` folder next to the notebook

### 4. Run the notebook
```bash
jupyter notebook DiabetesDetector.ipynb
```

---

## 🔬 How It Works

### Feature Preparation
The top 3 correlated features are extracted and normalized using **Min-Max scaling** so all values fall between 0 and 1 — preventing larger-scale features from dominating weight updates.

```python
X = dataset[['Glucose', 'BMI', 'Age']].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = dataset['Outcome'].values.reshape(-1, 1)
```

### Forward Pass
```python
hidden_layer_input  = np.dot(X, W1)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input  = np.dot(hidden_layer_output, W2)
predicted_output    = sigmoid(output_layer_input)
```

### Backpropagation
```python
error              = y - predicted_output
d_predicted_output = error * sigmoid_derivative(predicted_output)
error_hidden_layer = d_predicted_output.dot(W2.T)
d_hidden_layer     = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
```

### Weight Update
```python
W2 += hidden_layer_output.T.dot(d_predicted_output) * (learning_rate / m)
W1 += X.T.dot(d_hidden_layer) * (learning_rate / m)
```

---

## 📊 Results

| Metric | Before Training | After 2 Epochs |
|--------|----------------|----------------|
| MAE | 0.49340 | 0.46766 ↓ |
| Accuracy | 64.84% | 65.10% ↑ |

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations, weight initialization |
| `pandas` | Dataset loading and feature selection |
| `matplotlib` | Visualization |
| `seaborn` | Correlation bar chart |

---

## 📚 Reference

- Dataset: [UCI ML Repository — Pima Indians Diabetes](https://archive.ics.uci.edu/ml/datasets/diabetes)
- Course: TECH 405 — Artificial Neural Network & Deep Learning
- Author: Sandip Rai — Presidential Graduate School
