import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

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


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
        try:
            report = {}

            for model_name, model in models.items():
                model_params = params.get(model_name, {})  # get params for the model or empty dict

                if model_params:
                    # Use RandomizedSearchCV if parameters exist for the model
                    randomized_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=model_params,  # ✅ Use model_params directly
                        n_iter=5,
                        cv=3,
                        verbose=1,
                        n_jobs=-1
                    )

                    randomized_search.fit(X_train, y_train)
                    best_model = randomized_search.best_estimator_
                else:
                    # No params: fit model directly
                    model.fit(X_train, y_train)
                    best_model = model

                predictions = best_model.predict(X_test)
                score = r2_score(y_test, predictions)
                report[model_name] = score

            return report


        except Exception as e:
            logging.exception("Failed to evaluate models")
            raise CustomException(e, sys)
        