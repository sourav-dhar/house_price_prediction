import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Premium', 'Very Good', 'Ideal', 'Good', 'Fair']
            color_categories = ['F', 'J', 'G', 'E', 'D', 'H', 'I']
            clarity_categories = ['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1']

            logging.info('pipeline initiated')

            #Numerical pipeline
            num_pipeline = Pipeline(
                steps = [
                        ('imputer', SimpleImputer(strategy = 'median')),
                        ('scaler', StandardScaler())
                        ])

            #Categorical pipeline

            cat_pipeline = Pipeline(
                steps = [
                        ('imputer', SimpleImputer(strategy = 'most_frequent')),
                        ('encoder', OrdinalEncoder(categories = [cut_categories,color_categories,clarity_categories])),
                        ('scaler', StandardScaler()) ]
                                    )
            preprocessor = ColumnTransformer([
                        ('num_pipeline',num_pipeline, numerical_cols),
                        ('cat_pipeline', cat_pipeline, categorical_cols)])
            
            return preprocessor
        
            logging.info('pipeline completed')


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')
            logging.info(f'train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'test dataframe head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            #transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets.')


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('preprocessor pickle file saved')

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logging.info('Exception occured in the data transformation stage')
            raise CustomException(e,sys)