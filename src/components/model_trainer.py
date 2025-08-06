import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 
from src.utils import evaluate_models



@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_trainer(self, train_array, test_array,):
        try:
            logging.info("spliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                    "Linear Regression": LinearRegression(),
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    "RandomForestRegressor": RandomForestRegressor(),
                    "GradientBoostingRegressor": GradientBoostingRegressor(),
                    "AdaBoostRegressor": AdaBoostRegressor(),
                    "CatBoostRegressor": CatBoostRegressor(verbose=0),
                    "XGBRegressor": XGBRegressor(),
                    "KNeighborsRegressor": KNeighborsRegressor()
                }

            params = {
                    "DecisionTreeRegressor": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                    "RandomForestRegressor": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "GradientBoostingRegressor": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Linear Regression": {},
                    "XGBRegressor": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "CatBoostRegressor": {
                        'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoostRegressor": {
                        'learning_rate': [0.1, 0.01, 0.5, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "KNeighborsRegressor": {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']
                    }
                }

            logging.info("Starting model evaluation...")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,
                                                    y_test=y_test , models=models, params=params)
                
            best_model_score = max(model_report.values())
                
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6", sys)

            best_model = models[best_model_name]  # Now get the actual model object

            logging.info("Best model found both on training as well as on test data")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            model_r2_score = r2_score(y_test, predicted)

            print(f"Best Model: {best_model_name} with R2 Score: {model_r2_score}")

            logging.info("model evaluation completed...")
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            print(model_report)
            return best_model_name, model_r2_score

            
        
        except Exception as e:
            print("Original Error:", e)
            raise CustomException(e,sys) 