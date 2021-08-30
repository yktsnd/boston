import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow

from models import Model_RFR
from utils import Trainer

def main():
    mlflow.start_run()

    df = pd.read_csv(args.input)
    #print(df.head())
    
    X = df.drop('y', axis=1).values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    params_dict = {#'objective': args.objective,
                   }
    model = Model_RFR(params=params_dict)
    trainer = Trainer(exp_name=args.exp_name, model=model, model_name=args.model_name, cv=args.cv)
    train_pred = trainer.predict_cv(X_train, y_train)

    train_score = mean_squared_error(y_train, train_pred)
    print(train_score)
    mlflow.log_metric('Validation Mean Squared Error', train_score)

    mlflow.end_run()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--exp_name')
    parser.add_argument('--model_name', default='RFR')
    parser.add_argument('--cv', type=int, default=5)
    #parser.add_argument('--objective', default='regression')
    #parser.add_argument('--metric', default='mean_squared_error')

    args = parser.parse_args()
    
    main()