import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os
import pickle
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def prepare_data(dataset_path, task_type='classification'):
    """
    Prepare dataset for neural network training and evaluation
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file (CSV format expected)
    task_type : str
        'classification' or 'regression'
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split the data into training, validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # For classification tasks, convert labels to categorical format if needed
    if task_type == 'classification':
        # Check if we need to convert to categorical (for multi-class)
        num_classes = len(np.unique(y))
        if num_classes > 2:
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_val = keras.utils.to_categorical(y_val, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def create_traditional_ann(input_shape, task_type='classification', num_classes=None):
    """
    Create a traditional Artificial Neural Network
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input features (n_features,)
    task_type : str
        'classification' or 'regression'
    num_classes : int
        Number of classes for classification (None for binary classification or regression)
        
    Returns:
    --------
    Keras model
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=input_shape))
    
    # Hidden layers
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(32, activation='relu'))
    
    # Output layer
    if task_type == 'classification':
        if num_classes is None or num_classes == 2:  # Binary classification
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:  # Multi-class classification
            model.add(keras.layers.Dense(num_classes, activation='softmax'))
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    else:  # Regression
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
    
    return model

def create_neural_fabric(input_shape, task_type='classification', num_classes=None):
    """
    Create a NeuralFabric model
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input features (n_features,)
    task_type : str
        'classification' or 'regression'
    num_classes : int
        Number of classes for classification (None for binary classification or regression)
        
    Returns:
    --------
    NeuralFabric model
    """
    # Replace this implementation with your actual NeuralFabric code
    
    # This is just a placeholder structure similar to traditional ANN
    # but with some architectural differences to simulate NeuralFabric
    
    inputs = keras.layers.Input(shape=input_shape)
    
    # Example of a more complex architecture that might represent NeuralFabric
    # Replace with your actual NeuralFabric implementation
    
    # Creating multiple paths (fabric-like structure)
    path1 = keras.layers.Dense(64, activation='relu')(inputs)
    path2 = keras.layers.Dense(64, activation='relu')(inputs)
    
    # Adding some complexity to path 1
    path1 = keras.layers.Dense(32, activation='relu')(path1)
    path1 = keras.layers.Dropout(0.3)(path1)
    
    # Adding some complexity to path 2
    path2 = keras.layers.Dense(32, activation='tanh')(path2)
    path2 = keras.layers.Dropout(0.3)(path2)
    
    # Merging paths
    merged = keras.layers.Concatenate()([path1, path2])
    merged = keras.layers.Dense(48, activation='relu')(merged)
    merged = keras.layers.Dropout(0.2)(merged)
    
    # Output layer
    if task_type == 'classification':
        if num_classes is None or num_classes == 2:  # Binary classification
            outputs = keras.layers.Dense(1, activation='sigmoid')(merged)
            loss = 'binary_crossentropy'
        else:  # Multi-class classification
            outputs = keras.layers.Dense(num_classes, activation='softmax')(merged)
            loss = 'categorical_crossentropy'
    else:  # Regression
        outputs = keras.layers.Dense(1, activation='linear')(merged)
        loss = 'mean_squared_error'
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    if task_type == 'classification':
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['mae']
        )
    
    return model

def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, task_type='classification', batch_size=32, epochs=50):
    """
    Train and evaluate a neural network model
    
    Parameters:
    -----------
    model : Keras model
        The model to train and evaluate
    model_name : str
        Name of the model for reporting
    X_train, y_train, X_val, y_val, X_test, y_test : arrays
        Training, validation, and test data
    task_type : str
        'classification' or 'regression'
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs for training
        
    Returns:
    --------
    Dictionary containing training history, trained model, and evaluation metrics
    """
    results = {}
    
    # Track training time
    start_time = time.time()
    
    # Implement early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    print(f"\nTraining {model_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    # Store training history and time
    results['history'] = history.history
    results['training_time'] = training_time
    results['model'] = model
    
    # Measure inference time
    inference_start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - inference_start_time
    results['inference_time'] = inference_time
    results['inference_time_per_sample'] = inference_time / len(X_test)
    
    # Evaluate based on task type
    if task_type == 'classification':
        # Process predictions for classification
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-class
            y_pred = np.argmax(predictions, axis=1)
            if len(y_test.shape) > 1:  # If y_test is one-hot encoded
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
        else:  # Binary
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = y_test
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve and AUC for binary classification
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, predictions.flatten())
            results['roc_auc'] = auc(fpr, tpr)
            results['fpr'] = fpr
            results['tpr'] = tpr
        
        print(f"\n{model_name} Evaluation Metrics:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        if 'roc_auc' in results:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    else:  # Regression
        y_pred = predictions.flatten()
        
        # Calculate metrics
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation Metrics:")
        print(f"Mean Absolute Error: {results['mae']:.4f}")
        print(f"Mean Squared Error: {results['mse']:.4f}")
        print(f"Root Mean Squared Error: {results['rmse']:.4f}")
        print(f"R² Score: {results['r2']:.4f}")
    
    # Calculate model complexity
    results['model_params'] = model.count_params()
    results['model_layers'] = len(model.layers)
    
    print(f"Model Parameters: {results['model_params']}")
    print(f"Inference Time: {results['inference_time']:.4f} seconds")
    print(f"Inference Time per Sample: {results['inference_time_per_sample']*1000:.4f} ms")
    
    return results

def run_comparison(dataset_path, task_type='classification', batch_size=32, epochs=50):
    """
    Run a comprehensive comparison between Traditional ANN and NeuralFabric
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    task_type : str
        'classification' or 'regression'
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs for training
        
    Returns:
    --------
    Dictionary containing results for both models
    """
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(dataset_path, task_type)
    
    # Get input shape
    input_shape = (X_train.shape[1],)
    
    # Get number of classes for classification
    if task_type == 'classification':
        if len(y_train.shape) > 1:  # One-hot encoded
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))
            # If binary classification, set to None
            if num_classes == 2:
                num_classes = None
    else:
        num_classes = None
    
    # Create models
    traditional_ann = create_traditional_ann(input_shape, task_type, num_classes)
    neural_fabric = create_neural_fabric(input_shape, task_type, num_classes)
    
    # Train and evaluate Traditional ANN
    traditional_results = train_and_evaluate_model(
        traditional_ann, "Traditional ANN", 
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs
    )
    
    # Train and evaluate NeuralFabric
    neural_fabric_results = train_and_evaluate_model(
        neural_fabric, "NeuralFabric", 
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs
    )
    
    return {
        'Traditional ANN': traditional_results,
        'NeuralFabric': neural_fabric_results,
        'task_type': task_type
    }

def visualize_training_history(results):
    """
    Visualize training history for both models
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    """
    task_type = results['task_type']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot training loss
    for ax, metric in zip(axes, ['loss', 'val_loss']):
        for model_name, model_results in results.items():
            if model_name not in ['task_type']:
                history = model_results['history']
                ax.plot(history[metric], label=f"{model_name} - {metric}")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"{'Training' if 'val' not in metric else 'Validation'} Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training/validation metrics based on task type
    if task_type == 'classification':
        metric_name = 'accuracy'
    else:  # Regression
        metric_name = 'mae'
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, metric in zip(axes, [metric_name, f'val_{metric_name}']):
        for model_name, model_results in results.items():
            if model_name not in ['task_type']:
                history = model_results['history']
                ax.plot(history[metric], label=f"{model_name} - {metric}")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.title())
        ax.set_title(f"{'Training' if 'val' not in metric else 'Validation'} {metric_name.title()}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_performance_metrics(results):
    """
    Visualize performance metrics comparison
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    """
    task_type = results['task_type']
    
    if task_type == 'classification':
        # Collect metrics
        metrics = {
            'Accuracy': [results[model]['accuracy'] for model in results if model != 'task_type'],
            'Precision': [results[model]['precision'] for model in results if model != 'task_type'],
            'Recall': [results[model]['recall'] for model in results if model != 'task_type'],
            'F1 Score': [results[model]['f1'] for model in results if model != 'task_type']
        }
        
        # If binary classification, add ROC AUC
        if 'roc_auc' in results['Traditional ANN']:
            metrics['ROC AUC'] = [results[model]['roc_auc'] for model in results if model != 'task_type']
        
    else:  # Regression
        metrics = {
            'MAE': [results[model]['mae'] for model in results if model != 'task_type'],
            'MSE': [results[model]['mse'] for model in results if model != 'task_type'],
            'RMSE': [results[model]['rmse'] for model in results if model != 'task_type'],
            'R² Score': [results[model]['r2'] for model in results if model != 'task_type']
        }
    
    # Plot metrics comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Get model names
    model_names = [model for model in results if model != 'task_type']
    
    # Plot bars
    rects1 = ax.bar(x - width/2, [metrics[m][0] for m in metrics], width, label=model_names[0])
    rects2 = ax.bar(x + width/2, [metrics[m][1] for m in metrics], width, label=model_names[1])
    
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Metrics Comparison for {task_type.title()} Task')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # For binary classification, plot ROC curves
    if task_type == 'classification' and 'roc_auc' in results['Traditional ANN']:
        plt.figure(figsize=(10, 8))
        
        for model_name, model_results in results.items():
            if model_name not in ['task_type']:
                plt.plot(
                    model_results['fpr'], 
                    model_results['tpr'], 
                    label=f"{model_name} (AUC = {model_results['roc_auc']:.4f})"
                )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_efficiency_metrics(results):
    """
    Visualize efficiency metrics comparison
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    """
    # Collect metrics
    metrics = {
        'Training Time (s)': [results[model]['training_time'] for model in results if model != 'task_type'],
        'Inference Time (ms/sample)': [results[model]['inference_time_per_sample']*1000 for model in results if model != 'task_type'],
        'Model Parameters (log10)': [np.log10(results[model]['model_params']) for model in results if model != 'task_type'],
        'Layers': [results[model]['model_layers'] for model in results if model != 'task_type']
    }
    
    # Plot metrics comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Get model names
    model_names = [model for model in results if model != 'task_type']
    
    # Plot bars
    rects1 = ax.bar(x - width/2, [metrics[m][0] for m in metrics], width, label=model_names[0])
    rects2 = ax.bar(x + width/2, [metrics[m][1] for m in metrics], width, label=model_names[1])
    
    ax.set_ylabel('Value')
    ax.set_title('Efficiency Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if 'Parameters' in ax.get_xticklabels()[rects.index(rect)].get_text():
                # Convert log back to actual value for display
                label = f'{10**height:.0f}'
            else:
                label = f'{height:.2f}'
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_confusion_matrices(results):
    """
    Visualize confusion matrices for classification tasks
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    """
    if results['task_type'] != 'classification':
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for i, (model_name, model_results) in enumerate([item for item in results.items() if item[0] != 'task_type']):
        cm = model_results['confusion_matrix']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=axes[i]
        )
        
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_regression_predictions(results, X_test, y_test):
    """
    Visualize regression predictions
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    X_test : array
        Test features
    y_test : array
        True test values
    """
    if results['task_type'] != 'regression':
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for i, (model_name, model_results) in enumerate([item for item in results.items() if item[0] != 'task_type']):
        model = model_results['model']
        y_pred = model.predict(X_test).flatten()
        
        axes[i].scatter(y_test, y_pred, alpha=0.5)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[i].set_title(f'{model_name} Predictions')
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predictions')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add metrics text
        metrics_text = (
            f"MAE: {model_results['mae']:.4f}\n"
            f"MSE: {model_results['mse']:.4f}\n"
            f"RMSE: {model_results['rmse']:.4f}\n"
            f"R²: {model_results['r2']:.4f}"
        )
        axes[i].text(
            0.05, 0.95, metrics_text,
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig('regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comparison_report(results):
    """
    Generate a detailed comparison report
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_comparison
    """
    task_type = results['task_type']
    model_names = [model for model in results if model != 'task_type']
    
    print("\n" + "="*80)
    print(f"COMPARISON REPORT: {model_names[0]} vs {model_names[1]}")
    print("="*80)
    
    # Model architecture comparison
    print("\nMODEL ARCHITECTURE COMPARISON:")
    print("-"*40)
    for model_name in model_names:
        print(f"\n{model_name}:")
        print(f"  - Total Parameters: {results[model_name]['model_params']:,}")
        print(f"  - Total Layers: {results[model_name]['model_layers']}")
    
    # Training efficiency comparison
    print("\nTRAINING EFFICIENCY COMPARISON:")
    print("-"*40)
    for model_name in model_names:
        print(f"\n{model_name}:")
        print(f"  - Training Time: {results[model_name]['training_time']:.2f} seconds")
        
        # Find epoch where validation loss was minimum
        val_loss = results[model_name]['history']['val_loss']
        best_epoch = np.argmin(val_loss) + 1
        print(f"  - Best Validation Loss at Epoch: {best_epoch}")
        print(f"  - Best Validation Loss: {min(val_loss):.6f}")
    
    # Inference efficiency comparison
    print("\nINFERENCE EFFICIENCY COMPARISON:")
    print("-"*40)
    for model_name in model_names:
        print(f"\n{model_name}:")
        print(f"  - Total Inference Time: {results[model_name]['inference_time']:.4f} seconds")
        print(f"  - Inference Time per Sample: {results[model_name]['inference_time_per_sample']*1000:.4f} ms")
    
    # Performance metrics comparison
    print("\nPERFORMANCE METRICS COMPARISON:")
    print("-"*40)
    
    if task_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Add ROC AUC if available
        if 'roc_auc' in results[model_names[0]]:
            metrics.append('roc_auc')
            metric_names.append('ROC AUC')
    else:  # Regression
        metrics = ['mae', 'mse', 'rmse', 'r2']
        metric_names = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R² Score']
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name}:")
        for model_name in model_names:
            if metric in results[model_name]:
                print(f"  - {model_name}: {results[model_name][metric]:.6f}")
        
        # Calculate percentage improvement
        if metric in results[model_names[0]] and metric in results[model_names[1]]:
            # Determine which model is better for this metric
            if metric in ['mae', 'mse', 'rmse']:  # Lower is better
                if results[model_names[0]][metric] < results[model_names[1]][metric]:
                    better_model = model_names[0]
                    worse_model = model_names[1]
                    improvement = ((results[worse_model][metric] - results[better_model][metric]) / 
                                  results[worse_model][metric] * 100)
                else:
                    better_model = model_names[1]
                    worse_model = model_names[0]
                    improvement = ((results[worse_model][metric] - results[better_model][metric]) / 
                                  results[worse_model][metric] * 100)
            else:  # Higher is better
                if results[model_names[0]][metric] > results[model_names[1]][metric]:
                    better_model = model_names[0]
                    worse_model = model_names[1]
                    improvement = ((results[better_model][metric] - results[worse_model][metric]) / 
                                  results[worse_model][metric] * 100)
                else:
                    better_model = model_names[1]
                    worse_model = model_names[0]
                    improvement = ((results[better_model][metric] - results[worse_model][metric]) / 
                                  results[worse_model][metric] * 100)
            
            print(f"  - {better_model} outperforms {worse_model} by {improvement:.2f}%")
    
    # Overall summary
    print("\nOVERALL SUMMARY:")
    print("-"*40)
    
    # Determine winner based on primary metrics
    if task_type == 'classification':
        primary_metric = 'f1'  # F1 score as primary metric for classification
        primary_metric_name = 'F1 Score'
    else:
        primary_metric = 'rmse'  # RMSE as primary metric for regression
        primary_metric_name = 'Root Mean Squared Error'
    
    if ((primary_metric in ['mae', 'mse', 'rmse'] and 
         results[model_names[0]][primary_metric] < results[model_names[1]][primary_metric]) or
        (primary_metric not in ['mae', 'mse', 'rmse'] and 
         results[model_names[0]][primary_metric] > results[model_names[1]][primary_metric])):
        winner = model_names[0]
        loser = model_names[1]
    else:
        winner = model_names[1]
        loser = model_names[0]
    
    # Calculate percentage difference
    if primary_metric in ['mae', 'mse', 'rmse']:  # Lower is better
        improvement = ((results[loser][primary_metric] - results[winner][primary_metric]) / 
                      results[loser][primary_metric] * 100)
    else:  # Higher is better
        improvement = ((results[winner][primary_metric] - results[loser][primary_metric]) / 
                      results[loser][primary_metric] * 100)
    
    print(f"\nBased on {primary_metric_name}, {winner} outperforms {loser} by {improvement:.2f}%")
    
    # Consider other factors
    winner_params = results[winner]['model_params']
    loser_params = results[loser]['model_params']
    
    if winner_params < loser_params:
        param_ratio = loser_params / winner_params
        print(f"{winner} is more efficient with {param_ratio:.2f}x fewer parameters")
    else:
        param_ratio = winner_params / loser_params
        print(f"{winner} uses {param_ratio:.2f}x more parameters than {loser}")
    
    winner_time = results[winner]['inference_time_per_sample']
    loser_time = results[loser]['inference_time_per_sample']
    
    if winner_time < loser_time:
        time_ratio = loser_time / winner_time
        print(f"{winner} is {time_ratio:.2f}x faster in inference time")
    else:
        time_ratio = winner_time / loser_time
        print(f"{winner} is {time_ratio:.2f}x slower in inference time than {loser}")
    
    print("\nFINAL RECOMMENDATION:")
    
    performance_winner = winner
    efficiency_winner = model_names[0] if (results[model_names[0]]['model_params'] < results[model_names[1]]['model_params'] and 
                                         results[model_names[0]]['inference_time_per_sample'] < results[model_names[1]]['inference_time_per_sample']) else model_names[1]
    
    if performance_winner == efficiency_winner:
        print(f"- {performance_winner} is recommended as it offers both better performance and efficiency")
    else:
        print(f"- For optimal performance: Use {performance_winner}")
        print(f"- For optimal efficiency: Use {efficiency_winner}")
        
        # Make a final recommendation
        if task_type == 'classification':
            # For classification, F1 score improvement > 5% might be worth efficiency trade-off
            if improvement > 5:
                print(f"- Final recommendation: {performance_winner} (performance gain outweighs efficiency loss)")
            else:
                print(f"- Final recommendation: {efficiency_winner} (efficiency gain outweighs small performance loss)")
        else:
            # For regression, RMSE improvement > 10% might be worth efficiency trade-off
            if improvement > 10:
                print(f"- Final recommendation: {performance_winner} (performance gain outweighs efficiency loss)")
            else:
                print(f"- Final recommendation: {efficiency_winner} (efficiency gain outweighs small performance loss)")

def main(dataset_path, task_type='classification', batch_size=32, epochs=50):
    """
    Main function to run the entire comparison pipeline
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    task_type : str
        'classification' or 'regression'
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs for training
    """
    print("\n" + "="*50)
    print(f"Starting comparison: Traditional ANN vs NeuralFabric")
    print("="*50)
    
    # Run comparison
    comparison_results = run_comparison(dataset_path, task_type, batch_size, epochs)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_results.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)
    
    # Load test data for regression visualization
    if task_type == 'regression':
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(dataset_path, task_type)
    else:
        X_test = None
        y_test = None
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_training_history(comparison_results)
    visualize_performance_metrics(comparison_results)
    visualize_efficiency_metrics(comparison_results)
    
    if task_type == 'classification':
        visualize_confusion_matrices(comparison_results)
    else:
        visualize_regression_predictions(comparison_results, X_test, y_test)
    
    # Generate report
    generate_comparison_report(comparison_results)
    
    print("\n" + "="*50)
    print("Comparison completed successfully!")
    print("="*50)
    print("\nResults are saved in the 'results' directory.")
    print("Visualizations are saved as PNG files.")

# Example usage
if __name__ == "__main__":
    # For a classification task
    # main("path/to/your/classification_dataset.csv", task_type='classification')
    
    # For a regression task
    # main("path/to/your/regression_dataset.csv", task_type='regression')
    
    # For a binary classification example using a well-known dataset
    from sklearn.datasets import load_breast_cancer
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                     columns=np.append(cancer['feature_names'], ['target']))
    df.to_csv("breast_cancer_dataset.csv", index=False)
    
    # Run comparison
    main("breast_cancer_dataset.csv", task_type='classification', batch_size=32, epochs=30)
