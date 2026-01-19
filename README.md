# Garbage-Classification-Deep-Learning 

This repository contains the official implementation and research findings from the paper: **"A Comparative Benchmark of Machine Learning Algorithms for Garbage Classification using Deep Feature Extraction"** by Vishal Agarwal, et al..

The project focuses on building a high-performance waste classification system by benchmarking 10 classical machine learning algorithms against features extracted from deep pre-trained networks.

**Live Demo:** [Click Here](https://garbage-classification-deep-learning.streamlit.app/)

---

## Methodology
This study utilizes a **Deep Feature Extraction** pipeline rather than training simple models on raw pixels.

1.  **Dataset:** The "Garbage Classification V2" dataset (Kaggle), featuring **19,762 images** across 10 classes: *Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, and Trash*.
2.  **Feature Extraction:** Utilized a pre-trained **MobileNetV2** model (initialized with ImageNet weights) as a bottleneck feature extractor.
3.  **Preprocessing:** Images were resized to 128x128 pixels and transformed into 1,280-dimensional feature vectors.
4.  **Classification:** Benchmarked 10 algorithms: Logistic Regression, KNN, SVM, Naive Bayes, Bayesian Network, Decision Tree, Random Forest, XGBoost, AdaBoost, and MLP.

## Benchmarking Results
The Support Vector Machine (SVM) emerged as the top-performing model, while the Multi-Layer Perceptron (MLP) offered the best balance for real-world deployment.

| Model | Accuracy (%) | Macro F1 (%) | ROC-AUC (%) |
| :--- | :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | **93.65%** | **92.40%** | **99.63%** |
| **Multi-Layer Perceptron (MLP)** | **92.56%** | **90.94%** | **99.56%** |
| **K-Nearest Neighbors (KNN)** | 92.26% | 90.88% | 98.11% |
| **XGBoost (XGB)** | 91.30% | 89.38% | 99.41% |
| **Logistic Regression (LR)** | 90.92% | 89.13% | 99.31% |

*Key finding: Kernel-based methods (SVM) were found particularly effective for the high-dimensional features extracted by MobileNetV2.*

## Key Features
- **Hybrid Architecture:** Combines deep learning feature extraction with efficient classical ML classifiers.
- **Handling Class Imbalance:** Employs class-weighted loss optimization and stratified sampling to manage the natural imbalance in waste streams.
- **Real-time Interface:** Built with Streamlit, providing instant predictions and confidence scores for user-uploaded images.

## Tech Stack
- **Deep Learning:** TensorFlow, Keras (Feature Extraction) 
- **Machine Learning:** Scikit-learn, XGBoost 
- **Web App:** Streamlit 
- **Probabilistic Modeling:** pgmpy (for Bayesian Network implementation)
  
### 1. Clone the repository
```bash
git clone https://github.com/your-username/Garbage_Classification-Deep-Learning.git
cd Garbage_Classification-Deep-Learning
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run locally
```bash
streamlit run Home.py
```


## 📄 Research Paper
For a full deep dive into the experimental setup, hyperparameter tuning, and detailed per-class performance analysis, refer to the full paper:

**"A Comparative Benchmark of Machine Learning Algorithms for Garbage Classification using Deep Feature Extraction"** 

For detailed explanations of the project’s design, methodology, and results, please refer to the full paper:  
[Download Research Paper](docs/ResearchPaper)
