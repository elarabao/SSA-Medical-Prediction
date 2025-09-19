import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import seaborn as sns
import os
import json
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Create directory for saving results
timestamp = time.strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

def save_hyperparameters(params, filename):
    """Save hyperparameters to JSON file"""
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(params, f, indent=4)

def save_confusion_matrix(y_true, y_pred, model_name, title, dpi=400):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.jpg'), dpi=dpi)
    plt.close()

def save_roc_curve(y_true, y_pred_proba, model_name, title, dpi=400):
    """Generate and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f'roc_curve_{model_name}.jpg'), dpi=dpi)
    plt.close()

def save_convergence_curve(fitness_history, model_name, title, dpi=400):
    """Plot and save convergence curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.title(f'Convergence Curve - {title}')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Error Rate)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'convergence_{model_name}.jpg'), dpi=dpi)
    plt.close()

def evaluate_model(model, X, y, model_name, title):
    """Evaluate model and save results"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    # Print metrics
    print(f"\n{title} - {model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=4))
    
    # Save confusion matrix and ROC curve
    save_confusion_matrix(y, y_pred, f"{model_name}_{title.lower()}", title)
    save_roc_curve(y, y_pred_proba, f"{model_name}_{title.lower()}", title)
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'roc_auc': roc_auc
    }

# SSA (Salp Swarm Algorithm) Optimizer Implementation
class SalpSwarmOptimizer:
    def __init__(self, n_salps, dim, bounds, max_iter, leader_init=None):
        """
        Initialize SSA optimizer
        n_salps: Number of salps
        dim: Problem dimension
        bounds: Parameter bounds
        max_iter: Maximum iterations
        leader_init: Initial leader position
        """
        self.n_salps = n_salps
        self.dim = dim
        self.bounds = np.array(bounds)
        self.max_iter = max_iter

        # Initialize salp positions
        self.salps = np.zeros((n_salps, dim))
        for i in range(dim):
            min_bound, max_bound = self.bounds[i]
            self.salps[:, i] = np.random.uniform(min_bound, max_bound, n_salps)
        
        # Set leader initial position
        if leader_init is not None:
            self.salps[0, :] = leader_init

        # Store best position and fitness
        self.best_position = None
        self.best_fitness = float('inf')
        self.fitness_history = []  # Record best fitness for each iteration
        self.initial_fitness = None  # Record initial fitness

    def optimize(self, fitness_func):
        """Execute optimization process"""
        # Evaluate initial population
        fitness = np.zeros(self.n_salps)
        for i in range(self.n_salps):
            fitness[i] = fitness_func(self.salps[i, :])
            if fitness[i] < self.best_fitness:
                self.best_fitness = fitness[i]
                self.best_position = self.salps[i, :].copy()
        
        # Record initial fitness
        self.initial_fitness = fitness[0]
        self.fitness_history.append(self.best_fitness)
        print(f"Initial fitness: {self.initial_fitness:.4f}")

        # Start optimization iterations
        for t in range(self.max_iter):
            # Update convergence factor c1
            c1 = 2 * np.exp(-(4 * t / self.max_iter) ** 2)

            # Update leader position
            for i in range(self.dim):
                c2 = np.random.random()
                c3 = np.random.random()

                if c3 < 0.5:
                    self.salps[0, i] = self.best_position[i] + c1 * (
                            (self.bounds[i, 1] - self.bounds[i, 0]) * c2 + self.bounds[i, 0])
                else:
                    self.salps[0, i] = self.best_position[i] - c1 * (
                            (self.bounds[i, 1] - self.bounds[i, 0]) * c2 + self.bounds[i, 0])

                # Ensure position is within bounds
                self.salps[0, i] = np.clip(
                    self.salps[0, i], self.bounds[i, 0], self.bounds[i, 1]
                )

            # Update follower positions
            for i in range(1, self.n_salps):
                for j in range(self.dim):
                    self.salps[i, j] = 0.5 * (self.salps[i, j] + self.salps[i - 1, j])

            # Evaluate new positions and update best solution
            for i in range(self.n_salps):
                fitness[i] = fitness_func(self.salps[i, :])
                if fitness[i] < self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_position = self.salps[i, :].copy()

            self.fitness_history.append(self.best_fitness)

            # Print progress
            if (t + 1) % 5 == 0:
                print(f"Iteration {t + 1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")

        return self.best_position, self.best_fitness

# =============================================================
# Data Loading and Preprocessing
# =============================================================
# Load your dataset here
# Please replace this with your own dataset path
print("Please load your dataset here. Replace the following lines with your data loading code:")
print("df = pd.read_csv('your_dataset.csv')")
print("Make sure your dataset has a target column named 'output' for binary classification")

# Example dataset loading (replace with your own)
# df = pd.read_csv('your_dataset.csv', encoding='utf-8_sig')

# For demonstration purposes, create a sample dataset structure
# Remove this section and use your actual dataset
np.random.seed(42)
n_samples = 1000
n_features = 20

# Create sample data
X_sample = np.random.randn(n_samples, n_features)
y_sample = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # Imbalanced dataset

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X_sample, columns=feature_names)
df['output'] = y_sample

print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {Counter(df['output'])}")

# Split data into train, validation, and test sets (64%/16%/20%)
X = df.drop('output', axis=1)
y = df['output']

# First split: train+val vs test (80% vs 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: train vs val (64% vs 16% of total data)
# Since we have 80% left, we need 64/80 = 0.8 for train and 16/80 = 0.2 for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Use SMOTE for oversampling on training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print('Class distribution before oversampling:', Counter(y_train))
print('Class distribution after oversampling:', Counter(y_train_resampled))

# =============================================================
# 1. SSA-Optimized Random Forest Model
# =============================================================
# Define Random Forest fitness function using new loss function
def rf_fitness(position):
    """Calculate Random Forest loss function with given parameters"""
    n_estimators = int(position[0])
    max_depth = int(position[1]) if position[1] >= 1 else None
    min_samples_split = int(position[2])
    min_samples_leaf = int(position[3])

    # Create Random Forest model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # Use 5-fold cross-validation to calculate accuracy and recall
    acc_scores = cross_val_score(rf, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    recall_scores = cross_val_score(rf, X_train_resampled, y_train_resampled, cv=5, scoring='recall')
    
    # Calculate mean accuracy and recall
    mean_accuracy = np.mean(acc_scores)
    mean_recall = np.mean(recall_scores)
    
    # Calculate new loss function (weighted combination)
    loss = (1 - mean_recall) * 0.8 + (1 - mean_accuracy) * 0.2
    return loss

# Set Random Forest leader initial parameters
rf_leader_init = [100, 10, 4, 1]  # n_estimators, max_depth, min_samples_split, min_samples_leaf

# Save initial parameters
initial_rf_params = {
    'n_estimators': rf_leader_init[0],
    'max_depth': rf_leader_init[1],
    'min_samples_split': rf_leader_init[2],
    'min_samples_leaf': rf_leader_init[3]
}
save_hyperparameters(initial_rf_params, 'initial_rf_params.json')

# SSA parameter settings
print("\nStarting Random Forest parameter optimization...")
rf_ssa = SalpSwarmOptimizer(
    n_salps=10,
    dim=4,
    bounds=[(50, 200), (3, 20), (2, 10), (1, 5)],  # n_estimators, max_depth, min_samples_split, min_samples_leaf
    max_iter=20,
    leader_init=rf_leader_init
)

# Train initial model
initial_rf = RandomForestClassifier(
    n_estimators=rf_leader_init[0],
    max_depth=rf_leader_init[1],
    min_samples_split=rf_leader_init[2],
    min_samples_leaf=rf_leader_init[3],
    random_state=42,
    n_jobs=-1
)
initial_rf.fit(X_train_resampled, y_train_resampled)

# Evaluate initial model on validation set
print("\nInitial Random Forest model performance on validation set:")
initial_rf_val_metrics = evaluate_model(initial_rf, X_val_scaled, y_val, 'rf', 'Initial RF Validation')

# Evaluate initial model on test set
print("\nInitial Random Forest model performance on test set:")
initial_rf_test_metrics = evaluate_model(initial_rf, X_test_scaled, y_test, 'rf', 'Initial RF Test')

# Execute optimization
start_time = time.time()
best_rf_params, best_rf_fitness = rf_ssa.optimize(rf_fitness)
elapsed_time = time.time() - start_time

# Parse best parameters
n_estimators = int(best_rf_params[0])
max_depth = int(best_rf_params[1]) if best_rf_params[1] >= 1 else None
min_samples_split = int(best_rf_params[2])
min_samples_leaf = int(best_rf_params[3])

# Save optimized parameters
optimized_rf_params = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'fitness': best_rf_fitness
}
save_hyperparameters(optimized_rf_params, 'optimized_rf_params.json')

print("\nRandom Forest optimization completed!")
print(f"Optimization time: {elapsed_time:.2f} seconds")
print(f"Best parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
      f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
print(f"Best fitness value: {best_rf_fitness:.4f}")

# Train final model with best parameters
optimized_rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    n_jobs=-1
)
optimized_rf.fit(X_train_resampled, y_train_resampled)

# Evaluate optimized model on validation set
print("\nOptimized Random Forest model performance on validation set:")
optimized_rf_val_metrics = evaluate_model(optimized_rf, X_val_scaled, y_val, 'rf', 'Optimized RF Validation')

# Evaluate optimized model on test set
print("\nOptimized Random Forest model performance on test set:")
optimized_rf_test_metrics = evaluate_model(optimized_rf, X_test_scaled, y_test, 'rf', 'Optimized RF Test')

# Save convergence curve
save_convergence_curve(rf_ssa.fitness_history, 'rf', 'Random Forest Optimization')

# =============================================================
# 2. SSA-Optimized XGBoost Model
# =============================================================
# Calculate scale_pos_weight parameter
counter = Counter(y_train_resampled)
scale_pos_weight = counter[0] / counter[1] if 1 in counter and 0 in counter else 1

# Define XGBoost fitness function using new loss function
def xgb_fitness(position):
    """Calculate XGBoost loss function with given parameters"""
    n_estimators = int(position[0])
    max_depth = int(position[1])
    learning_rate = position[2]
    gamma = position[3]
    min_child_weight = position[4]

    # Create XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Use 5-fold cross-validation to calculate accuracy and recall
    acc_scores = cross_val_score(xgb_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    recall_scores = cross_val_score(xgb_model, X_train_resampled, y_train_resampled, cv=5, scoring='recall')
    
    # Calculate mean accuracy and recall
    mean_accuracy = np.mean(acc_scores)
    mean_recall = np.mean(recall_scores)
    
    # Calculate new loss function
    loss = (1 - mean_recall) * 0.8 + (1 - mean_accuracy) * 0.2
    return loss

# SSA parameter settings for XGBoost
print("\nStarting XGBoost parameter optimization...")
xgb_ssa = SalpSwarmOptimizer(
    n_salps=10,
    dim=5,
    bounds=[(50, 200), (3, 10), (0.01, 0.3), (0, 1), (1, 10)],  # n_estimators, max_depth, learning_rate, gamma, min_child_weight
    max_iter=20
)

# Train initial model with default parameters
initial_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
initial_xgb.fit(X_train_resampled, y_train_resampled)

# Save initial parameters
initial_xgb_params = initial_xgb.get_params()
save_hyperparameters(initial_xgb_params, 'initial_xgb_params.json')

# Evaluate initial model on validation set
print("\nInitial XGBoost model performance on validation set:")
initial_xgb_val_metrics = evaluate_model(initial_xgb, X_val_scaled, y_val, 'xgb', 'Initial XGBoost Validation')

# Evaluate initial model on test set
print("\nInitial XGBoost model performance on test set:")
initial_xgb_test_metrics = evaluate_model(initial_xgb, X_test_scaled, y_test, 'xgb', 'Initial XGBoost Test')

# Execute optimization
start_time = time.time()
best_xgb_params, best_xgb_fitness = xgb_ssa.optimize(xgb_fitness)
elapsed_time = time.time() - start_time

# Parse best parameters
n_estimators_xgb = int(best_xgb_params[0])
max_depth_xgb = int(best_xgb_params[1])
learning_rate = best_xgb_params[2]
gamma = best_xgb_params[3]
min_child_weight = best_xgb_params[4]

# Save optimized parameters
optimized_xgb_params = {
    'n_estimators': n_estimators_xgb,
    'max_depth': max_depth_xgb,
    'learning_rate': learning_rate,
    'gamma': gamma,
    'min_child_weight': min_child_weight,
    'fitness': best_xgb_fitness
}
save_hyperparameters(optimized_xgb_params, 'optimized_xgb_params.json')

print("\nXGBoost optimization completed!")
print(f"Optimization time: {elapsed_time:.2f} seconds")
print(f"Best parameters: n_estimators={n_estimators_xgb}, max_depth={max_depth_xgb}, "
      f"learning_rate={learning_rate:.4f}, gamma={gamma:.4f}, min_child_weight={min_child_weight:.1f}")
print(f"Best fitness value: {best_xgb_fitness:.4f}")

# Train final model with best parameters
optimized_xgb = xgb.XGBClassifier(
    n_estimators=n_estimators_xgb,
    max_depth=max_depth_xgb,
    learning_rate=learning_rate,
    gamma=gamma,
    min_child_weight=min_child_weight,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
optimized_xgb.fit(X_train_resampled, y_train_resampled)

# Evaluate optimized model on validation set
print("\nOptimized XGBoost model performance on validation set:")
optimized_xgb_val_metrics = evaluate_model(optimized_xgb, X_val_scaled, y_val, 'xgb', 'Optimized XGBoost Validation')

# Evaluate optimized model on test set
print("\nOptimized XGBoost model performance on test set:")
optimized_xgb_test_metrics = evaluate_model(optimized_xgb, X_test_scaled, y_test, 'xgb', 'Optimized XGBoost Test')

# Save convergence curve
save_convergence_curve(xgb_ssa.fitness_history, 'xgb', 'XGBoost Optimization')

# Save all evaluation results
all_metrics = {
    'initial_rf_validation': initial_rf_val_metrics,
    'initial_rf_test': initial_rf_test_metrics,
    'optimized_rf_validation': optimized_rf_val_metrics,
    'optimized_rf_test': optimized_rf_test_metrics,
    'initial_xgb_validation': initial_xgb_val_metrics,
    'initial_xgb_test': initial_xgb_test_metrics,
    'optimized_xgb_validation': optimized_xgb_val_metrics,
    'optimized_xgb_test': optimized_xgb_test_metrics
}

with open(os.path.join(results_dir, 'all_metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=4)

print(f"\nAll results saved to directory: {results_dir}")
print("\nSummary:")
print("="*60)
print("Random Forest Results:")
print(f"Initial model (Test): Accuracy={initial_rf_test_metrics['accuracy']:.4f}, Recall={initial_rf_test_metrics['recall']:.4f}")
print(f"Optimized model (Test): Accuracy={optimized_rf_test_metrics['accuracy']:.4f}, Recall={optimized_rf_test_metrics['recall']:.4f}")
print("\nXGBoost Results:")
print(f"Initial model (Test): Accuracy={initial_xgb_test_metrics['accuracy']:.4f}, Recall={initial_xgb_test_metrics['recall']:.4f}")
print(f"Optimized model (Test): Accuracy={optimized_xgb_test_metrics['accuracy']:.4f}, Recall={optimized_xgb_test_metrics['recall']:.4f}")
print("="*60)