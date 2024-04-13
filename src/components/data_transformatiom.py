import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """"
        This function is resposible for data transformation
        """
        try:
            numerical_columns = ['wrinting_score','reading_score']
            categorical_columns = [
                "gender",
                "race_ethinicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
                )              
            cat_pipeline= Pipeline(

                steps=[
                    ("imputer",StandardScaler(strategy="most_frequent")),
                    ("one_hot_coder",OneHotEncoder()),
                    ("scaler", StandardScaler())

                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

                
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns=["writing_score","Reading_score"]

        except:
            pass

