import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

# Import machine learning models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class DiabetesModel:
    """
    A class for training and evaluating machine learning models for diabetes prediction.
    
    This class implements XGBoost, Random Forest, and Gradient Boosting models and provides
    methods for training, evaluation, and visualization of results.
    """
    
    def __init__(self, models_dir='models', random_state=42):
        """
        Initialize the DiabetesModel.
        
        Parameters:
        -----------
        models_dir : str, default='models'
            Directory to save trained models.
        random_state : int, default=42
            Random state for reproducibility.
        """
        self.models_dir = models_dir
        self.random_state = random_state
        self.trained_models = {}
        self.evaluation_results = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def train_xgboost(self, X_train, y_train, params=None):
        """
        Train an XGBoost model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features.
        y_train : pandas.Series or numpy.ndarray
            Training target.
        params : dict, optional
            XGBoost parameters. If None, default parameters will be used.
            
        Returns:
        --------
        xgboost.Booster
            Trained XGBoost model.
        """
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        start_time = time()
        
        # Default parameters
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'alpha': 0,
                'lambda': 1,
                'tree_method': 'auto',
                'seed': self.random_state
            }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train the model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            verbose_eval=100
        )
        
        # Calculate training time
        training_time = time() - start_time
        print(f"XGBoost training completed in {training_time:.2f} seconds")
        
        # Store the trained model
        self.trained_models['xgboost'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train, params=None):
        """
        Train a Random Forest model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features.
        y_train : pandas.Series or numpy.ndarray
            Training target.
        params : dict, optional
            Random Forest parameters. If None, default parameters will be used.
            
        Returns:
        --------
        sklearn.ensemble.RandomForestClassifier
            Trained Random Forest model.
        """
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        start_time = time()
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        # Create and train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time() - start_time
        print(f"Random Forest training completed in {training_time:.2f} seconds")
        
        # Store the trained model
        self.trained_models['random_forest'] = model
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, params=None):
        """
        Train a Gradient Boosting model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features.
        y_train : pandas.Series or numpy.ndarray
            Training target.
        params : dict, optional
            Gradient Boosting parameters. If None, default parameters will be used.
            
        Returns:
        --------
        sklearn.ensemble.GradientBoostingClassifier
            Trained Gradient Boosting model.
        """
        print("\n" + "="*50)
        print("Training Gradient Boosting Model")
        print("="*50)
        
        start_time = time()
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8,
                'random_state': self.random_state
            }
        
        # Create and train the model
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time() - start_time
        print(f"Gradient Boosting training completed in {training_time:.2f} seconds")
        
        # Store the trained model
        self.trained_models['gradient_boosting'] = model
        
        return model
    
    def train_logistic_regression(self, X_train, y_train, params=None):
        """
        Train a Logistic Regression model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features.
        y_train : pandas.Series or numpy.ndarray
            Training target.
        params : dict, optional
            Logistic Regression parameters. If None, default parameters will be used.
            
        Returns:
        --------
        sklearn.linear_model.LogisticRegression
            Trained Logistic Regression model.
        """
        print("\n" + "="*50)
        print("Training Logistic Regression Model")
        print("="*50)
        
        start_time = time()
        
        # Default parameters
        if params is None:
            params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        # Create and train the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time() - start_time
        print(f"Logistic Regression training completed in {training_time:.2f} seconds")
        
        # Store the trained model
        self.trained_models['logistic_regression'] = model
        
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate.
        X_test : pandas.DataFrame or numpy.ndarray
            Test features.
        y_test : pandas.Series or numpy.ndarray
            Test target.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")
        
        model = self.trained_models[model_name]
        
        # Get predictions
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model.predict(dtest)
        else:  # scikit-learn models
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Binary classification threshold
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store and print results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        self.evaluation_results[model_name] = results
        
        print("\n" + "-"*50)
        print(f"Evaluation Results for {model_name.upper()}")
        print("-"*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return results
    
    def plot_confusion_matrix(self, model_name):
        """
        Plot confusion matrix for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The confusion matrix plot.
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results.")
        
        results = self.evaluation_results[model_name]
        y_test = results.get('y_test')
        y_pred = results.get('y_pred')
        
        if y_test is None or y_pred is None:
            raise ValueError("Test data or predictions not found in evaluation results.")
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the confusion matrix plot
        plt.savefig(os.path.join(self.models_dir, f'{model_name}_confusion_matrix.png'))
        
        return plt
    
    def plot_roc_curve(self, model_names=None):
        """
        Plot ROC curve for one or more models.
        
        Parameters:
        -----------
        model_names : list or None
            List of model names to include in the plot. If None, all evaluated models are included.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The ROC curve plot.
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"Warning: Model '{model_name}' not found in evaluation results.")
                continue
            
            results = self.evaluation_results[model_name]
            y_test = results.get('y_test')
            y_pred_proba = results.get('y_pred_proba')
            
            if y_test is None or y_pred_proba is None:
                print(f"Warning: Test data or probability predictions not found for '{model_name}'.")
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results.get('roc_auc', auc(fpr, tpr))
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save the ROC curve plot
        plt.savefig(os.path.join(self.models_dir, 'roc_curves.png'))
        
        return plt
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance from a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model.
        feature_names : list, optional
            List of feature names.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importances.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")
        
        model = self.trained_models[model_name]
        
        if model_name == 'xgboost':
            # XGBoost feature importance
            importance_type = 'gain'
            importance = model.get_score(importance_type=importance_type)
            
            if feature_names is None:
                features = list(importance.keys())
            else:
                # Map feature indices to names
                features = []
                for key in importance.keys():
                    try:
                        # Handle feature names with format 'f123'
                        index = int(key[1:]) if key.startswith('f') else int(key)
                        if index < len(feature_names):
                            features.append(feature_names[index])
                        else:
                            features.append(key)
                    except (ValueError, IndexError):
                        features.append(key)
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': list(importance.values())
            })
            
        elif model_name in ['random_forest', 'gradient_boosting']:
            # Scikit-learn tree-based models
            importance = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importance)],
                'Importance': importance
            })
            
        elif model_name == 'logistic_regression':
            # Logistic Regression coefficients
            coef = np.abs(model.coef_[0])
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(coef))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(coef)],
                'Importance': coef
            })
            
        else:
            raise ValueError(f"Feature importance not supported for model type '{model_name}'.")
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def plot_feature_importance(self, model_name, feature_names=None, top_n=10):
        """
        Plot feature importance for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model.
        feature_names : list, optional
            List of feature names.
        top_n : int, default=10
            Number of top features to display.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The feature importance plot.
        """
        importance_df = self.get_feature_importance(model_name, feature_names)
        
        # Select top N features
        if len(importance_df) > top_n:
            plot_df = importance_df.head(top_n)
        else:
            plot_df = importance_df
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=plot_df)
        plt.title(f'Top {len(plot_df)} Feature Importances - {model_name.upper()}')
        plt.tight_layout()
        
        # Save the feature importance plot
        plt.savefig(os.path.join(self.models_dir, f'{model_name}_feature_importance.png'))
        
        return plt
    
    def save_model(self, model_name):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")
        
        model = self.trained_models[model_name]
        
        # Save model using joblib (works for both XGBoost and scikit-learn models)
        model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        
        print(f"Model '{model_name}' saved to {model_path}")
    
    def load_model(self, model_name):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load.
            
        Returns:
        --------
        object
            The loaded model.
        """
        model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
        model = joblib.load(model_path)
        
        self.trained_models[model_name] = model
        print(f"Model '{model_name}' loaded from {model_path}")
        
        return model
    
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Train and evaluate all models.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.ndarray
            Training features.
        y_train : pandas.Series or numpy.ndarray
            Training target.
        X_test : pandas.DataFrame or numpy.ndarray
            Test features.
        y_test : pandas.Series or numpy.ndarray
            Test target.
        feature_names : list, optional
            List of feature names.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation results for all models.
        """
        models_to_train = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression']
        
        # Store test data for evaluation
        for model_name in models_to_train:
            if model_name not in self.evaluation_results:
                self.evaluation_results[model_name] = {}
            
            self.evaluation_results[model_name]['y_test'] = y_test
        
        # Train and evaluate XGBoost
        self.train_xgboost(X_train, y_train)
        self.evaluate_model('xgboost', X_test, y_test)
        self.save_model('xgboost')
        self.plot_confusion_matrix('xgboost')
        
        if feature_names is not None:
            xgb_importance = self.get_feature_importance('xgboost', feature_names)
            print("\nXGBoost Feature Importance:")
            print(xgb_importance.head(10))
            self.plot_feature_importance('xgboost', feature_names)
        
        # Train and evaluate Random Forest
        self.train_random_forest(X_train, y_train)
        self.evaluate_model('random_forest', X_test, y_test)
        self.save_model('random_forest')
        self.plot_confusion_matrix('random_forest')
        
        if feature_names is not None:
            rf_importance = self.get_feature_importance('random_forest', feature_names)
            print("\nRandom Forest Feature Importance:")
            print(rf_importance.head(10))
            self.plot_feature_importance('random_forest', feature_names)
        
        # Train and evaluate Gradient Boosting
        self.train_gradient_boosting(X_train, y_train)
        self.evaluate_model('gradient_boosting', X_test, y_test)
        self.save_model('gradient_boosting')
        self.plot_confusion_matrix('gradient_boosting')
        
        if feature_names is not None:
            gb_importance = self.get_feature_importance('gradient_boosting', feature_names)
            print("\nGradient Boosting Feature Importance:")
            print(gb_importance.head(10))
            self.plot_feature_importance('gradient_boosting', feature_names)
        
        # Train and evaluate Logistic Regression
        self.train_logistic_regression(X_train, y_train)
        self.evaluate_model('logistic_regression', X_test, y_test)
        self.save_model('logistic_regression')
        self.plot_confusion_matrix('logistic_regression')
        
        if feature_names is not None:
            lr_importance = self.get_feature_importance('logistic_regression', feature_names)
            print("\nLogistic Regression Feature Importance:")
            print(lr_importance.head(10))
            self.plot_feature_importance('logistic_regression', feature_names)
        
        # Plot ROC curves for all models
        self.plot_roc_curve()
        
        # Compare model performances
        comparison = {}
        for model_name, results in self.evaluation_results.items():
            comparison[model_name] = {
                'accuracy': results.get('accuracy'),
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'f1_score': results.get('f1_score'),
                'roc_auc': results.get('roc_auc')
            }
        
        comparison_df = pd.DataFrame(comparison).T
        print("\n" + "="*50)
        print("Model Performance Comparison")
        print("="*50)
        print(comparison_df)
        
        # Determine best model based on ROC AUC
        best_model = comparison_df['roc_auc'].idxmax()
        print(f"\nBest model based on ROC AUC: {best_model.upper()} (AUC = {comparison_df.loc[best_model, 'roc_auc']:.4f})")
        
        return self.evaluation_results

# Example usage
if __name__ == "__main__":
    # Import necessary modules
    from sklearn.model_selection import train_test_split
    
    # Import the DataProcessor to load and preprocess the data
    from DataPreprocessing import DataProcessor
    
    # Initialize the processor
    processor = DataProcessor()
    
    # Path to the dataset (adjust as needed)
    dataset_path = "D:/Project/Diabetes Prediction/Dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    
    # Load the data
    data = processor.load_data(dataset_path)
    
    if data is not None:
        # Preprocess the data
        X_train, X_test, y_train, y_test = processor.preprocess_data(
            data, 
            target_col='Diabetes_binary',
            test_size=0.2,
            use_smote=False
        )
        
        # Initialize the model trainer
        model_trainer = DiabetesModel(models_dir='models')
        
        # Train and evaluate all models
        feature_names = processor.feature_names
        results = model_trainer.train_and_evaluate_all(X_train, y_train, X_test, y_test, feature_names)