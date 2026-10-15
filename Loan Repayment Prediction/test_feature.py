import pytest
import pandas as pd
import numpy as np

# Import the variables directly from your original script file.
# Note: This will execute the machine learning script upon import.
try:
    from Loan_Repayment_Prediction import models_performance, df_models
except ImportError:
    # Fallback placeholders if the file can't be found
    models_performance = {}
    df_models = pd.DataFrame()

def test_model_performance_metrics_structure():
    """Test that the model performance dictionary from the script contains expected models and valid scores."""
    expected_models = [
        'Decision Tree', 'Bagging (DT)', 'AdaBoost (DT)', 
        'Random Forest', 'AdaBoost (RF)', 'Gradient Boosting'
    ]
    
    # Verify all models are present in the dictionary
    for model in expected_models:
        assert model in models_performance, f"Missing model: {model}"
        
    # Verify scores are valid float values between 0.0 and 1.0
    for model, score in models_performance.items():
        assert isinstance(score, (float, np.floating)), f"Score for {model} must be a float."
        assert 0.0 <= score <= 1.0, f"Score for {model} is out of valid bounds: {score}"

def test_model_comparison_dataframe_creation():
    """Test that the DataFrame created in the script has the correct structure and values."""
    assert not df_models.empty, "The df_models DataFrame is empty."
    assert list(df_models.columns) == ['Model', 'Accuracy'], "Columns do not match ['Model', 'Accuracy']."
    
    # Verify shape matches the number of models evaluated
    assert df_models.shape[0] == len(models_performance)

def test_best_model_selection():
    """Test the logic used to find the top-performing model from the script's metrics."""
    assert len(models_performance) > 0, "No model performance metrics found."
    
    # Extract the key with the highest score
    best_model = max(models_performance, key=models_performance.get)
    
    # Assert that a valid model name is returned
    assert best_model in models_performance
    print(f"Verified Best Model: {best_model} with score {models_performance[best_model]}")