from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, Pipeline as ColPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def build_preprocessor(X_train):
    """
    Build the preprocessing transformer.
    
    Args:
        X_train: Training data (DataFrame)
        
    Returns:
        ColumnTransformer with numeric and categorical pipelines
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    num_pipe = ColPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = ColPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    
    return preprocess, num_cols, cat_cols


def build_pipeline(preprocessor, classifier):
    """
    Build full training pipeline.
    
    Args:
        preprocessor: ColumnTransformer
        classifier: sklearn classifier instance
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    Returns:
        Pipeline(preprocessor -> classifier)
    """
    return Pipeline([
        ("prep", preprocessor),
        ("clf", classifier)
    ])