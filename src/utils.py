import pandas as pd
import numpy as np

import os
import dill

import sys

from src.logger import logging
from src.exception import CustomException
#from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pickle
import os

def save_object(file_path: str, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        logging.exception("Failed to save object")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test, y_test, models):
    try:

     report = {}
    
     for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        report[name] = score
    
     return report

    except Exception as e:
        logging.exception("Failed to evaluate models")
        raise CustomException(e, sys)
        