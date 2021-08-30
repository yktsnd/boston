import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gc
from tqdm import tqdm
from time import time
now = time()

class Trainer:
    def __init__(self, exp_name, model, model_name, cv):
        self.exp_name = exp_name
        self.model = model
        self.model_name = model_name
        self.cv = cv

        os.makedirs('{}/../../models/{}_{}'.format(os.path.dirname(__file__), self.model_name, self.exp_name), exist_ok=True)
        #os.makedirs('{}/../../models/{}/fold'.format(os.path.dirname(__file__), self.model_name), exist_ok=True)
    
    def predict_cv(self, train_x, train_y):
        np.random.seed = 42
        seeds = np.random.randint(0,10000,1)
        train_x = np.array(train_x)
        train_y = np.array(train_y)

        preds_train = np.zeros(len(train_x))
        for s in tqdm(seeds):
            print(s)
            self.model.seed=s

            pred_val = []
            va_idxes = []
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=s)
            # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
            for i, (tr_idx, va_idx) in enumerate(kf.split(train_x, train_y)):
                self.model.fold = i
                print(i)

                tr_x, va_x = train_x[tr_idx], train_x[va_idx]
                tr_y, va_y = train_y[tr_idx], train_y[va_idx]

                self.model.fit(tr_x, tr_y, va_x, va_y)

                # predict for validation
                pred = self.model.predict(va_x)
                pred_val.append(pred)
                va_idxes.append(va_idx)

                #self.save('{}/../../models/{}/fold/seed_{}-fold_{}'.format(os.path.dirname(__file__), self.model_name, s, i))

            # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
            va_idxes = np.concatenate(va_idxes)
            pred_val = np.concatenate(pred_val, axis=0)
            order = np.argsort(va_idxes)
            pred_train = pred_val[order]

            # seed averaging
            preds_train += pred_train/len(seeds)

        gc.collect()
        return preds_train
    
    def train(self, X_train, X_test, y_train, y_test):
        #全データで学習
        self.model.seed=42
        self.model.fit(X_train, y_train, X_test, y_test)
        self.save('{}/../../models/{}_{}.pickle'.format(os.path.dirname(__file__), self.model_name, self.exp_name))

    def save(self, path:str):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
