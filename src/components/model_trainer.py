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
from src.utils import save_object, evaluate_models


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
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "XGBoost": XGBRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,
                                                  y_test=y_test , models=models)
            
            best_model_score = max(model_report.values())
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6", sys)

            best_model = models[best_model_name]  # Now get the actual model object

            logging.info("Best model found both on training as well as on test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            model_r2_score = r2_score(y_test, predicted)
            return model_r2_score
        
        except Exception as e:
            print("Original Error:", e)
            raise CustomException(e,sys) 