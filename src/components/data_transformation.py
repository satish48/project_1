import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging  # ensure logging is configured somewhere globally


import logging as pylog
pylog.basicConfig(level=pylog.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl") #this is saving the file


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_cols = ["Gender", "Location", "Phone_Usage_Purpose","School_Grade"]
            numerical_cols = [
                "Age",
                "Daily_Usage_Hours",
                "Sleep_Hours",
                "Academic_Performance",
                "Social_Interactions",
                "Exercise_Hours",
                "Anxiety_Level",
                "Depression_Level",
                "Self_Esteem",
                "Parental_Control",
                "Screen_Time_Before_Bed",
                "Phone_Checks_Per_Day",
                "Apps_Used_Daily",
                "Time_on_Social_Media",
                "Time_on_Gaming",
                "Time_on_Education",
                "Family_Communication",
                "Weekend_Usage_Hours",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            logging.info(f"categorical columns: {categorical_cols}")
            logging.info(f"categorical columns: {numerical_cols}")


            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read and clean
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")


            # Strip whitespace from column names if any
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            

            preprocessing_object = self.get_data_transformer_object()

            target_column = "Addiction_Level"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Applying preprocessing to train and test sets.")
            X_train_array = preprocessing_object.fit_transform(X_train)
            X_test_array = preprocessing_object.transform(X_test)

            train_array = np.c_[X_train_array, np.array(y_train)]
            test_array = np.c_[X_test_array, np.array(y_test)]

            # Save the preprocessor object
            logging.info(f"Saving preprocessor to {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    train_arr, test_arr, prep_path = transformer.initiate_data_transformation(
        "artifacts/train.csv", "artifacts/test.csv"
    )
    logging.info(f"Transformation done. Preprocessor saved at: {prep_path}")
