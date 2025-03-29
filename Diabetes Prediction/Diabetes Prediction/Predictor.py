import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class DiabetesPredictor:
    """
    A class for loading trained models and making predictions for diabetes risk.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the DiabetesPredictor.
        
        Parameters:
        -----------
        models_dir : str, default='models'
            Directory containing trained models.
        """
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
    def load_models(self):
        """
        Load all available models from the models directory.
        
        Returns:
        --------
        dict
            Dictionary of loaded models.
        """
        # Define model file names
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'logistic_regression': 'logistic_regression_model.pkl'
        }
        
        # Load each model if available
        for model_name, file_name in model_files.items():
            model_path = os.path.join(self.models_dir, file_name)
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name} model: {str(e)}")
        
        # Try to load the scaler if available
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler: {str(e)}")
        
        return self.models
    
    def get_available_models(self):
        """
        Get a list of available model names.
        
        Returns:
        --------
        list
            List of available model names.
        """
        return list(self.models.keys())
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        input_data : dict or pandas.DataFrame
            Input data to preprocess.
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed input data.
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            input_df = pd.DataFrame(
                self.scaler.transform(input_df), 
                columns=input_df.columns
            )
        
        return input_df
    
    def predict(self, input_data, model_name=None):
        """
        Make a prediction using a specified model or all models.
        
        Parameters:
        -----------
        input_data : dict or pandas.DataFrame
            Input data for prediction.
        model_name : str, optional
            Name of the model to use. If None, all models are used.
            
        Returns:
        --------
        dict
            Dictionary containing prediction results.
        """
        # Check if models are loaded
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        # Preprocess input data
        input_df = self.preprocess_input(input_data)
        
        # Make predictions with one or all models
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        # Get predictions from each model
        predictions = {}
        
        for name, model in models_to_use.items():
            # Get probability prediction
            if name == 'xgboost':
                dmatrix = xgb.DMatrix(input_df)
                prob = float(model.predict(dmatrix)[0])
            else:  # scikit-learn models
                prob = float(model.predict_proba(input_df)[0, 1])
            
            # Make class prediction (0 or 1)
            pred_class = 1 if prob >= 0.5 else 0
            
            predictions[name] = {
                'probability': prob,
                'prediction': pred_class,
                'risk_level': 'High' if prob >= 0.7 else 'Medium' if prob >= 0.3 else 'Low'
            }
        
        # Calculate ensemble prediction if using multiple models
        if len(models_to_use) > 1:
            ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
            ensemble_class = 1 if ensemble_prob >= 0.5 else 0
            
            predictions['ensemble'] = {
                'probability': float(ensemble_prob),
                'prediction': ensemble_class,
                'risk_level': 'High' if ensemble_prob >= 0.7 else 'Medium' if ensemble_prob >= 0.3 else 'Low'
            }
        
        return predictions
    
    def get_feature_importance(self, model_name, top_n=10):
        """
        Get feature importances for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model.
        top_n : int, optional
            Number of top features to return.
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing feature importances, or None if not available.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Try to load feature importance file if available
        importance_path = os.path.join(self.models_dir, f'{model_name}_feature_importance.csv')
        if os.path.exists(importance_path):
            try:
                importance_df = pd.read_csv(importance_path)
                return importance_df.head(top_n)
            except Exception as e:
                print(f"Error loading feature importance from file: {str(e)}")
        
        # If file not available, try to compute from model
        try:
            if model_name == 'xgboost':
                importance = model.get_score(importance_type='gain')
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                })
            elif model_name in ['random_forest', 'gradient_boosting']:
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': [f'Feature_{i}' for i in range(len(importance))],
                    'Importance': importance
                })
            elif model_name == 'logistic_regression':
                coef = np.abs(model.coef_[0])
                importance_df = pd.DataFrame({
                    'Feature': [f'Feature_{i}' for i in range(len(coef))],
                    'Importance': coef
                })
            else:
                return None
            
            # Sort and return top N
            return importance_df.sort_values('Importance', ascending=False).head(top_n)
            
        except Exception as e:
            print(f"Error computing feature importance: {str(e)}")
            return None
    
    def generate_feature_importance_plot(self, model_name, top_n=10):
        """
        Generate a feature importance plot for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model.
        top_n : int, optional
            Number of top features to include.
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Feature importance plot, or None if not available.
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df is None or len(importance_df) == 0:
            return None
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {len(importance_df)} Feature Importances - {model_name.upper()}')
        plt.tight_layout()
        
        return plt

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor(models_dir='models')
    
    # Load models
    predictor.load_models()
    
    # Display available models
    print("\nAvailable models:")
    for model in predictor.get_available_models():
        print(f"- {model}")
    
    # Example input data
    example_input = {
        'HighBP': 1,           # High blood pressure (1=yes, 0=no)
        'HighChol': 1,         # High cholesterol (1=yes, 0=no)
        'CholCheck': 1,        # Cholesterol check in last 5 years (1=yes, 0=no)
        'BMI': 30,             # Body Mass Index
        'Smoker': 0,           # Smoker (1=yes, 0=no)
        'Stroke': 0,           # Had a stroke (1=yes, 0=no)
        'HeartDiseaseorAttack': 0,  # Heart disease or attack (1=yes, 0=no)
        'PhysActivity': 0,     # Physical activity in past 30 days (1=yes, 0=no)
        'Fruits': 0,           # Consume fruit 1+ times per day (1=yes, 0=no)
        'Veggies': 0,          # Consume vegetables 1+ times per day (1=yes, 0=no)
        'HvyAlcoholConsump': 0,  # Heavy alcohol consumption (1=yes, 0=no)
        'AnyHealthcare': 1,    # Any healthcare coverage (1=yes, 0=no)
        'NoDocbcCost': 0,      # Could not see doctor due to cost (1=yes, 0=no)
        'GenHlth': 4,          # General health (1=excellent, 5=poor)
        'MentHlth': 0,         # Days of poor mental health (0-30)
        'PhysHlth': 0,         # Days of poor physical health (0-30)
        'DiffWalk': 0,         # Difficulty walking/climbing stairs (1=yes, 0=no)
        'Sex': 1,              # Sex (1=male, 0=female)
        'Age': 8,              # Age category (1-13)
        'Education': 5,        # Education level (1-6)
        'Income': 7            # Income category (1-8)
    }
    
    # Make predictions using all models
    predictions = predictor.predict(example_input)
    
    # Display predictions
    print("\nPredictions:")
    for model_name, result in predictions.items():
        print(f"{model_name.upper()}:")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Prediction: {'Diabetic' if result['prediction'] == 1 else 'Non-Diabetic'}")
        print(f"  Risk Level: {result['risk_level']}")