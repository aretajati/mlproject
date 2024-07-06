import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation  import DataTransformation
from src.components.data_ingestion import DataIngestion

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts1","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1], ## selects all rows (:), and all columns except the last one (:-1).
                train_array[:,-1], ## selects all rows (:), and selects only the last column (-1).
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            

            # Hyperparameter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)
            
            # ## To get best model score from dict
            # best_model_score = max(sorted(model_report.values())[2]) # only sort and take the highest R2 score

            # ## To get best model name from dict
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]

            ## To get best model score and model name from dict
            best_model_key = max(model_report, key=lambda x: model_report[x]['R2_score'])
            best_model_score = model_report[best_model_key]['R2_score']

            best_model = models[best_model_key]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))

            # return best_model, r2_square, mae, rmse
            return best_model


        except Exception as e:
            raise CustomException(e,sys)
        
## to get the path of train data and test data from data_ingestion file
train_data,test_data = DataIngestion().initiate_data_ingestion()

## to get the path of train array, test array, and preprocessing pkl file from data_transformation file
train_arr,test_arr,_ = DataTransformation().initiate_data_transformation(train_data,test_data)

if __name__=="__main__":
    data_trainer=ModelTrainer()
    output_data_trainer = data_trainer.initiate_model_trainer(train_arr,test_arr,_)
    # print(f"Best model: {output_data_trainer[0]} (R2 Score: {output_data_trainer[1]}, MAE: {output_data_trainer[2]}, RMSE: MAE: {output_data_trainer[3]})")
    print(output_data_trainer)


### WRITE CODE BELOW TO RUN THIS DATA IN THE TERMINAL
### python -m src.components.model_trainer