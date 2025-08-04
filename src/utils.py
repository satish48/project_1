import pandas as pd
import numpy as np

import os
import dill

import sys

from src.logger import logging
from src.exception import CustomException


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
