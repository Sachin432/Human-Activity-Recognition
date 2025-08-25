"""
Human Activity Recognition (HAR) using UCI HAR Dataset
-----------------------------------------------------
This script trains multiple models (Logistic Regression, Linear SVM, RBF SVM,
Decision Tree, and LSTM) to classify human activities based on smartphone sensor data.
"""

# ==============================
# Import Libraries
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# Load Dataset
# ==============================
DATASET_PATH = "UCI_HAR_dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Load feature names
features = pd.read_csv(os.path.join(DATASET_PATH, "features.txt"),
                       sep="\s+", header=None, names=["index", "feature"])
feature_names = features["feature"].values

# Load train and test data
X_train = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.txt"), sep="\s+", header=None, names=feature_names)
y_train = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.txt"), sep="\s+", header=None, names=["Activity"])

X_test = pd.read_csv(os.path.join(TEST_PATH, "X_test.txt"), sep="\s+", header=None, names=feature_names)
y_test = pd.read_csv(os.path.join(TEST_PATH, "y_test.txt"), sep="\s+", header=None, names=["Activity"])

# Activity labels
activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
                   'SITTING', 'STANDING', 'LAYING']

# ==============================
# Preprocessing
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train.values.ravel())
y_test_enc = encoder.transform(y_test.values.ravel())

# ==============================
# Helper Functions
# ==============================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=activity_labels))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# ==============================
# Logistic Regression
# ==============================
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, solver="liblinear")
grid_lr = GridSearchCV(lr, {"C": [0.01, 0.1, 1, 10, 30]}, cv=3, n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train_enc)
best_lr = grid_lr.best_estimator_
evaluate_model(best_lr, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, "Logistic Regression")

# ==============================
# Linear SVM
# ==============================
print("\n--- Linear SVM ---")
svc_linear = SVC(kernel="linear")
grid_lsvm = GridSearchCV(svc_linear, {"C": [0.125, 0.5, 1, 2, 8, 16]}, cv=3, n_jobs=-1)
grid_lsvm.fit(X_train_scaled, y_train_enc)
best_lsvm = grid_lsvm.best_estimator_
evaluate_model(best_lsvm, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, "Linear SVM")

# ==============================
# RBF Kernel SVM
# ==============================
print("\n--- RBF Kernel SVM ---")
svc_rbf = SVC(kernel="rbf")
grid_rbf = GridSearchCV(svc_rbf, {"C": [1, 10], "gamma": [0.01, 0.001]}, cv=3, n_jobs=-1)
grid_rbf.fit(X_train_scaled, y_train_enc)
best_rbf = grid_rbf.best_estimator_
evaluate_model(best_rbf, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, "RBF SVM")

# ==============================
# Decision Tree
# ==============================
print("\n--- Decision Tree ---")
tree = DecisionTreeClassifier(random_state=42)
grid_tree = GridSearchCV(tree, {"max_depth": [5, 10, 20, None]}, cv=3, n_jobs=-1)
grid_tree.fit(X_train_scaled, y_train_enc)
best_tree = grid_tree.best_estimator_
evaluate_model(best_tree, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, "Decision Tree")

# ==============================
# LSTM Model
# ==============================
print("\n--- LSTM ---")

# Load inertial signals (9 channels)
def load_signals(subset):
    signals = []
    signals_list = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]
    for signal in signals_list:
        filename = os.path.join(DATASET_PATH, subset, "Inertial Signals", signal + subset + ".txt")
        signals.append(pd.read_csv(filename, delim_whitespace=True, header=None).values)
    return np.transpose(np.array(signals), (1, 2, 0))

X_train_lstm = load_signals("train")
X_test_lstm = load_signals("test")

y_train_lstm = to_categorical(y_train_enc, num_classes=len(activity_labels))
y_test_lstm = to_categorical(y_test_enc, num_classes=len(activity_labels))

timesteps, features = X_train_lstm.shape[1], X_train_lstm.shape[2]

lstm_model = Sequential([
    LSTM(64, input_shape=(timesteps, features)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(len(activity_labels), activation="softmax")
])

lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

y_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
y_true_lstm = np.argmax(y_test_lstm, axis=1)

print("\nLSTM Accuracy:", accuracy_score(y_true_lstm, y_pred_lstm))
print(classification_report(y_true_lstm, y_pred_lstm, target_names=activity_labels))

cm_lstm = confusion_matrix(y_true_lstm, y_pred_lstm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Blues",
            xticklabels=activity_labels, yticklabels=activity_labels)
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("lstm_confusion_matrix.png")
plt.close()
