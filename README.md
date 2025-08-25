# Human-Activity-Recognition

## Overview
This project demonstrates **Human Activity Recognition (HAR)** using the **UCI HAR Dataset**. The dataset contains accelerometer and gyroscope data collected from 30 volunteers performing six activities:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

We build and evaluate multiple models: **Logistic Regression, Linear SVM, RBF SVM, Decision Tree, and LSTM**.

---

## Dataset
The dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).  

- Training data: 70% of subjects  
- Test data: 30% of subjects  
- Each instance is a fixed-width window (2.56 seconds, 128 readings).  
- Features: 561 engineered features + raw inertial signals (for LSTM).  
