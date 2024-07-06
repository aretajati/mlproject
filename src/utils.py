import os
import sys

import numpy as np
import pandas as pd
import dill    ## lib to help make pickle file
import pickle

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_techniques(y_train, y_train_pred):
    mae = mean_absolute_error(y_train, y_train_pred)
    # mse = mean_squared_error(y_train, predicted)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_square = r2_score(y_train, y_train_pred)
    return mae, rmse, r2_square

def evaluate_models(X_train, y_train,X_test,y_test,models,param):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate Train and Test dataset
            model_train_mae, model_train_rmse, model_train_r2 = evaluate_techniques(y_train, y_train_pred)
            
            model_test_mae, model_test_rmse, model_test_r2 = evaluate_techniques(y_test, y_test_pred)


            report[list(models.keys())[i]] = {"MAE": model_test_mae, "RMSE": model_test_rmse, "R2_score":model_test_r2}

        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:  # rb is read byte mode
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
