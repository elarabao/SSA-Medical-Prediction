# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             classification_report, precision_score, recall_score, f1_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Ensure results directory exists
if not os.path.exists('../results'):
    os.makedirs('../results')


# =============================================================
# Helper functions: save parameters, evaluation metrics and confusion matrix
# =============================================================
def save_parameters(filename, params, model_type):
    """Save parameters to file"""
    with open(filename, 'w') as f:
        f.write(f"{model_type} Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def save_evaluation_metrics(filename, metrics, prefix=""):
    """Save evaluation metrics to file"""
    with open(filename, 'a') as f:
        f.write(f"\n{prefix} Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")


def plot_and_save_confusion_matrix(y_true, y_pred, filename, title="Confusion Matrix", dpi=400):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    classes = ["0", "1"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, format='jpg')
    plt.close()


def evaluate_model_final(pipeline, X_train, X_test, y_train, y_test, model_name):
    """Evaluate final model performance on held-out test set"""
    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\n{model_name} Final Test Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return metrics, y_pred, pipeline


# =============================================================
# Data loading and preprocessing
# =============================================================

def load_and_split_data():
    """Load data and perform initial train/test split ONLY"""
    # Data loading - Please replace with your own dataset path
    # Note: Replace the following paths with your actual dataset paths
    print("Loading dataset...")
    print("Please replace 'your_dataset.csv' with your actual dataset path")

    # For demonstration - replace with your actual data loading
    # df = pd.read_csv('your_dataset.csv')  # Please use your dataset file

    # Creating a sample dataset for demonstration
    # Remove this section and use your actual data loading code
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X_demo, columns=feature_names)
    df['target'] = y_demo
    # End of demonstration data - replace with your actual data

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    # Split features and target
    X = df.drop('target', axis=1)  # Modify according to your target column name
    y = df['target']  # Modify according to your target column name

    # ONLY split into train and test - validation will be handled by CV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Training set target distribution: {Counter(y_train)}")
    print(f"Test set target distribution: {Counter(y_test)}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# =============================================================
# Pipeline Creation with Proper Data Leakage Prevention
# =============================================================

def create_rf_pipeline():
    """Create Random Forest pipeline with all preprocessing steps"""
    pipeline = ImbPipeline([
        ('feature_selection', SelectKBest(f_classif)),  # Feature selection within CV
        ('scaler', StandardScaler()),  # Scaling within CV
        ('smote', SMOTE(random_state=42)),  # Resampling within CV
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return pipeline


def create_xgb_pipeline():
    """Create XGBoost pipeline with all preprocessing steps"""
    pipeline = ImbPipeline([
        ('feature_selection', SelectKBest(f_classif)),  # Feature selection within CV
        ('scaler', StandardScaler()),  # Scaling within CV
        ('smote', SMOTE(random_state=42)),  # Resampling within CV
        ('classifier', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])
    return pipeline


# =============================================================
# Nested Cross-Validation with Grid Search
# =============================================================

def perform_nested_cv_gridsearch(X_train, y_train, feature_names):
    """Perform nested cross-validation with grid search to prevent data leakage"""

    print("\n" + "=" * 60)
    print("NESTED CROSS-VALIDATION WITH GRID SEARCH")
    print("=" * 60)
    print("Outer CV: Model evaluation (5-fold)")
    print("Inner CV: Hyperparameter optimization (3-fold)")
    print("All preprocessing (scaling, SMOTE, feature selection) done within CV folds")

    # Define cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Define models and parameter grids
    models_config = {
        'Random Forest': {
            'pipeline': create_rf_pipeline(),
            'param_grid': {
                'feature_selection__k': [10, 15, 20],  # Number of features to select
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__max_features': ['sqrt', 'log2']
            }
        },
        'XGBoost': {
            'pipeline': create_xgb_pipeline(),
            'param_grid': {
                'feature_selection__k': [10, 15, 20],  # Number of features to select
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        }
    }

    # Store results for each model
    nested_cv_results = {}
    best_models = {}

    for model_name, config in models_config.items():
        print(f"\n{'-' * 50}")
        print(f"Processing {model_name}")
        print(f"{'-' * 50}")

        pipeline = config['pipeline']
        param_grid = config['param_grid']

        # Nested CV scores storage
        nested_scores = []
        fold_best_params = []

        # Outer CV loop
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
            print(f"Outer CV Fold {fold + 1}/5")

            # Split data for this fold
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            # Inner CV: Grid search for hyperparameter optimization
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            # Fit grid search on training fold
            grid_search.fit(X_train_fold, y_train_fold)

            # Evaluate best model on validation fold
            best_model_fold = grid_search.best_estimator_
            val_score = roc_auc_score(y_val_fold, best_model_fold.predict_proba(X_val_fold)[:, 1])

            nested_scores.append(val_score)
            fold_best_params.append(grid_search.best_params_)

            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Validation AUC: {val_score:.4f}")

        # Calculate nested CV performance
        mean_score = np.mean(nested_scores)
        std_score = np.std(nested_scores)

        print(f"\n{model_name} Nested CV Results:")
        print(f"Mean AUC: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in nested_scores]}")

        # Store results
        nested_cv_results[model_name] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': nested_scores,
            'fold_params': fold_best_params
        }

        # Train final model on entire training set with best average parameters
        # Find most common parameter choices across folds
        final_grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        final_grid_search.fit(X_train, y_train)
        best_models[model_name] = final_grid_search.best_estimator_

        # Save parameters
        save_parameters(f'results/best_{model_name.lower().replace(" ", "_")}_parameters.txt',
                        final_grid_search.best_params_, model_name)

    return nested_cv_results, best_models


# =============================================================
# Final Model Evaluation on Test Set
# =============================================================

def evaluate_final_models(best_models, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate final models on held-out test set"""

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    final_results = {}

    for model_name, pipeline in best_models.items():
        print(f"\nEvaluating {model_name}...")

        # Evaluate on test set
        test_metrics, test_pred, fitted_pipeline = evaluate_model_final(
            pipeline, X_train, X_test, y_train, y_test, model_name
        )

        final_results[model_name] = {
            'metrics': test_metrics,
            'predictions': test_pred,
            'pipeline': fitted_pipeline
        }

        # Save metrics
        save_evaluation_metrics(f'results/{model_name.lower().replace(" ", "_")}_final_metrics.txt',
                                test_metrics, "Final Test Set")

        # Save confusion matrix
        plot_and_save_confusion_matrix(
            y_test, test_pred,
            f'results/{model_name.lower().replace(" ", "_")}_confusion_matrix.jpg',
            f'{model_name} - Test Set Confusion Matrix'
        )

        # Save model
        model_filename = f'results/{model_name.lower().replace(" ", "_")}_final_model.pkl'
        joblib.dump(fitted_pipeline, model_filename)
        print(f"Model saved to {model_filename}")

        # Feature importance (for tree-based models)
        if hasattr(fitted_pipeline.named_steps['classifier'], 'feature_importances_'):
            # Get selected features
            selected_features_mask = fitted_pipeline.named_steps['feature_selection'].get_support()
            selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]

            importance_df = pd.DataFrame({
                'feature': selected_feature_names,
                'importance': fitted_pipeline.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)

            # Save feature importance
            importance_filename = f'results/{model_name.lower().replace(" ", "_")}_feature_importance.csv'
            importance_df.to_csv(importance_filename, index=False)

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(min(20, len(importance_df)))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Feature Importance (Selected Features Only)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'results/{model_name.lower().replace(" ", "_")}_feature_importance.jpg', dpi=400)
            plt.close()

    return final_results


# =============================================================
# Results Summary and Comparison
# =============================================================

def summarize_results(nested_cv_results, final_results):
    """Summarize and compare all results"""

    print("\n" + "=" * 60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 60)

    # Create comparison dataframes
    cv_comparison = pd.DataFrame({
        model: {
            'CV_Mean_AUC': results['mean_score'],
            'CV_Std_AUC': results['std_score']
        }
        for model, results in nested_cv_results.items()
    }).T

    test_comparison = pd.DataFrame({
        model: results['metrics']
        for model, results in final_results.items()
    }).T

    # Combine results
    full_comparison = pd.concat([cv_comparison, test_comparison], axis=1)

    print("\nNested Cross-Validation vs Test Set Performance:")
    print(full_comparison.round(4))

    # Save comparison
    full_comparison.to_csv('results/comprehensive_model_comparison.csv')

    # Determine best model
    best_model_cv = cv_comparison['CV_Mean_AUC'].idxmax()
    best_model_test = test_comparison['auc'].idxmax()

    print(f"\nBest model by nested CV: {best_model_cv} (AUC: {cv_comparison.loc[best_model_cv, 'CV_Mean_AUC']:.4f})")
    print(f"Best model by test set: {best_model_test} (AUC: {test_comparison.loc[best_model_test, 'auc']:.4f})")

    # Check for overfitting
    print("\nOverfitting Analysis:")
    for model in nested_cv_results.keys():
        cv_score = cv_comparison.loc[model, 'CV_Mean_AUC']
        test_score = test_comparison.loc[model, 'auc']
        difference = cv_score - test_score
        print(f"{model}: CV AUC = {cv_score:.4f}, Test AUC = {test_score:.4f}, Difference = {difference:.4f}")
        if abs(difference) > 0.05:
            print(f"  WARNING: Possible overfitting detected for {model}")

    return full_comparison


# =============================================================
# Main Execution Pipeline
# =============================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("MACHINE LEARNING PIPELINE WITH PROPER DATA LEAKAGE PREVENTION")
    print("=" * 60)
    print("This pipeline uses:")
    print("1. Proper train/test split")
    print("2. Nested cross-validation")
    print("3. All preprocessing within CV folds")
    print("4. Final evaluation on held-out test set")

    # Step 1: Load and split data
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()

    # Step 2: Perform nested cross-validation with grid search
    start_time = time.time()
    nested_cv_results, best_models = perform_nested_cv_gridsearch(X_train, y_train, feature_names)
    cv_time = time.time() - start_time
    print(f"\nNested CV completed in {cv_time:.2f} seconds")

    # Step 3: Final evaluation on test set
    final_results = evaluate_final_models(best_models, X_train, X_test, y_train, y_test, feature_names)

    # Step 4: Comprehensive results summary
    comparison_df = summarize_results(nested_cv_results, final_results)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ No data leakage: All preprocessing done within CV folds")
    print("✅ Unbiased estimates: Nested cross-validation used")
    print("✅ Proper validation: Final test on completely held-out data")
    print("✅ Results saved to 'results/' folder")
    print("\nCheck the 'results' folder for detailed outputs including:")
    print("- Model parameters and performance metrics")
    print("- Confusion matrices and feature importance plots")
    print("- Trained model files (.pkl)")
    print("- Comprehensive comparison results")


if __name__ == "__main__":
    main()