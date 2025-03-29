import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import os

class DataProcessor:
    """
    A class for handling all data processing tasks for the diabetes prediction project.
    
    This class handles loading the BRFSS dataset, exploratory data analysis,
    preprocessing, and preparation of data for model training.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the DataProcessor.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def load_data(self, file_path):
        """
        Load the dataset from the specified file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset CSV file.
            
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load the dataset
            data = pd.read_csv(file_path)
            print(f"Successfully loaded dataset from {file_path}")
            print(f"Dataset shape: {data.shape}")
            
            return data
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self, data):
        """
        Perform exploratory data analysis on the dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to explore.
            
        Returns:
        --------
        dict
            A dictionary containing various stats and insights about the data.
        """
        # Dictionary to store all insights
        insights = {}
        
        # Basic information
        insights['shape'] = data.shape
        insights['columns'] = data.columns.tolist()
        insights['dtypes'] = data.dtypes
        insights['missing_values'] = data.isnull().sum()
        
        # Summary statistics
        insights['summary'] = data.describe()
        
        # Target distribution (assuming 'Diabetes_binary' or 'Diabetes_012' is the target column)
        if 'Diabetes_binary' in data.columns:
            target_col = 'Diabetes_binary'
            insights['target_distribution'] = data[target_col].value_counts()
            insights['target_distribution_percent'] = data[target_col].value_counts(normalize=True) * 100
        elif 'Diabetes_012' in data.columns:
            target_col = 'Diabetes_012'
            insights['target_distribution'] = data[target_col].value_counts()
            insights['target_distribution_percent'] = data[target_col].value_counts(normalize=True) * 100
        
        # Identify feature types
        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_col in self.numerical_features:
            self.numerical_features.remove(target_col)
            
        self.categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        insights['numerical_features'] = self.numerical_features
        insights['categorical_features'] = self.categorical_features
        
        # Print key insights
        print(f"\nDataset has {data.shape[0]} rows and {data.shape[1]} columns")
        
        if 'target_distribution' in insights:
            print(f"\nTarget distribution ({target_col}):")
            for label, count in insights['target_distribution'].items():
                percent = insights['target_distribution_percent'][label]
                print(f"  Class {label}: {count} samples ({percent:.2f}%)")
        
        print(f"\nMissing values:")
        missing = insights['missing_values']
        if missing.sum() == 0:
            print("  No missing values found")
        else:
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} missing values")
        
        return insights
    
    def plot_distributions(self, data, target_col=None):
        """
        Plot distributions of features in the dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to visualize.
        target_col : str, optional
            Target column name for visualizing class distributions.
        """
        # Set up the figure
        plt.figure(figsize=(20, 15))
        
        # Determine number of features to plot
        if target_col:
            features = [col for col in data.columns if col != target_col]
        else:
            features = data.columns.tolist()
        
        # Calculate grid dimensions
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Plot histograms for each feature
        for i, feature in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # For categorical features or binary features
            if data[feature].nunique() < 10:
                if target_col:
                    sns.countplot(x=feature, hue=target_col, data=data)
                    plt.title(f'Distribution of {feature} by {target_col}')
                else:
                    sns.countplot(x=feature, data=data)
                    plt.title(f'Distribution of {feature}')
            # For continuous features
            else:
                if target_col and data[target_col].nunique() < 5:
                    for value in data[target_col].unique():
                        sns.histplot(data[data[target_col] == value][feature], 
                                     label=f'{target_col}={value}',
                                     kde=True, alpha=0.5)
                    plt.legend()
                else:
                    sns.histplot(data[feature], kde=True)
                plt.title(f'Distribution of {feature}')
                
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        return plt
    
    def plot_correlation_matrix(self, data, target_col=None):
        """
        Plot correlation matrix for the dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to visualize.
        target_col : str, optional
            Target column name to highlight correlations with target.
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Set up the figure
        plt.figure(figsize=(16, 12))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # If target column is provided, print its correlations
        if target_col and target_col in data.columns:
            target_corr = corr_matrix[target_col].sort_values(ascending=False)
            print(f"\nFeature correlations with {target_col}:")
            for feature, corr in target_corr.items():
                if feature != target_col:
                    print(f"  {feature}: {corr:.4f}")
        
        return plt
    
    def preprocess_data(self, data, target_col, test_size=0.2, use_smote=False):
        """
        Preprocess the data for model training.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to preprocess.
        target_col : str
            The target column name.
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split.
        use_smote : bool, default=False
            Whether to apply SMOTE for addressing class imbalance.
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        try:
            # Make a copy of the data
            df = data.copy()
            
            # Handle missing values (if any)
            print("Handling missing values...")
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            
            # Separate features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Store feature names for later use
            self.feature_names = X.columns.tolist()
            
            # Split the data
            print(f"Splitting data with test_size={test_size}...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            # Apply feature scaling
            print("Applying feature scaling...")
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train), 
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test), 
                columns=X_test.columns
            )
            
            # Apply SMOTE for imbalanced data if requested
            if use_smote:
                print("Applying SMOTE to balance the training dataset...")
                smote = SMOTE(random_state=self.random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                
                print(f"Class distribution before SMOTE:")
                print(y_train.value_counts(normalize=True))
                print(f"Class distribution after SMOTE:")
                print(pd.Series(y_train_resampled).value_counts(normalize=True))
                
                return X_train_resampled, X_test_scaled, y_train_resampled, y_test
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        except Exception as e:
            print(f"Error during data preprocessing: {str(e)}")
            raise
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """
        Save preprocessed data to disk.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : pandas.DataFrame or numpy.ndarray
            Preprocessed data splits.
        output_dir : str
            Directory to save the data.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy arrays to dataframes if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=self.feature_names)
        
        # Save the data
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        
        # Convert series to dataframes for saving
        pd.DataFrame(y_train, columns=[y_train.name if hasattr(y_train, 'name') else 'target']).to_csv(
            os.path.join(output_dir, 'y_train.csv'), index=False
        )
        pd.DataFrame(y_test, columns=[y_test.name if hasattr(y_test, 'name') else 'target']).to_csv(
            os.path.join(output_dir, 'y_test.csv'), index=False
        )
        
        # Save the scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        print(f"Preprocessed data saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = DataProcessor()
    
    # Path to the dataset (adjust as needed)
    # Using the balanced dataset for demonstration
    dataset_path = "D:/Project/Diabetes Prediction/Dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    
    # Load the data
    data = processor.load_data(dataset_path)
    
    if data is not None:
        # Explore the data
        insights = processor.explore_data(data)
        
        # Plot distributions
        processor.plot_distributions(data, target_col='Diabetes_binary')
        
        # Plot correlation matrix
        processor.plot_correlation_matrix(data, target_col='Diabetes_binary')
        
        # Preprocess the data
        X_train, X_test, y_train, y_test = processor.preprocess_data(
            data, 
            target_col='Diabetes_binary',
            test_size=0.2,
            use_smote=False  # Set to True if you want to balance the classes
        )
        
        # Print shapes of the preprocessed data
        print("\nPreprocessed data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Save preprocessed data (optional)
        # processor.save_preprocessed_data(X_train, X_test, y_train, y_test, "processed_data")