import pickle
import numpy as np

#import catboost
import lightgbm as lgb
import tensorflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#from tensorflow.keras import regularizers
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
#                                     Dropout)
#from tensorflow.keras.models import Sequential, load_model
#import tensorflow.keras.backend as K

class Model_lgb:
    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold

    def fit(self, tr_x, tr_y, va_x, va_y):
        if self.params['objective'] == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        elif self.params['objective'] == ('binary' or 'multiclass'):
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            print('objective parameter is needed. (regression/binary, multiclass)')
        self.model.fit(tr_x, tr_y, eval_set = (va_x, va_y), early_stopping_rounds=50)
        #print(self.model.score(va_x,va_y))
        
    def predict_proba(self, x):
        proba = self.model.predict(x)
        return proba
    
    def predict(self, x):
        pred = np.rint(self.model.predict(x))
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))

    def get_params(self):
        return self.model.get_params()

"""
class Model_catboost:

    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model = catboost.CatBoostClassifier(**self.params)
        self.model.fit(tr_x,tr_y,eval_set=(va_x,va_y),early_stopping_rounds=50)
        
    def predict_proba(self, x):
        proba = self.model.predict_proba(x)[:,1]
        return proba
    
    def predict(self, x):
        pred = np.rint(self.model.predict_proba(x)[:,1])
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))

    def get_params(self):
        return self.model.get_params()


class Model_NN:
    def __init__(self, fold=None, seed=None, trained=False):
        self.model = None
        self.seed = seed
        self.fold = fold
        self.scaler = StandardScaler()
        self.trained = trained
        tensorflow.random.set_seed(self.seed)
 
    def fit(self, tr_x, tr_y, va_x, va_y):
        tr_x, va_x, tr_y, va_y = tr_x.values, va_x.values, tr_y.values, va_y.values
 
        self.scaler.fit(tr_x)
 
        batch_size = 100
        epochs = 10000
 
        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
 
        early_stopping =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=20
        )
 
        model = Sequential()
        
        # layers
        num_layer=5
        for i in range(num_layer):
            model.add(Dense(256, input_shape=(tr_x.shape[1],), kernel_regularizer=regularizers.l2(9e-5)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.6))
        
        #final layer
        model.add(Dense(1, kernel_regularizer=regularizers.l2(9e-5)))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='NAdam',
                      metrics=['accuracy'])
        
        model.summary()
        
        checkpoint_path = f'Model_NN/best_checkpoint_seed-{self.seed}_fold-{self.fold}.ckpt'
        checkpoint_dir = Path(checkpoint_path)
        cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, monitor='val_accuracy', verbose=1)
 
        if not self.trained:
            history = model.fit(tr_x, tr_y,
                                batch_size=batch_size, epochs=epochs,
                                verbose=1,
                                validation_data=(va_x, va_y),
                                callbacks=[early_stopping, cp_callback])
            #keras_plot(history)
            del history

        model.load_weights(str(checkpoint_dir))
        
        self.model = model
 
    def predict_proba(self, x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x).reshape(-1)
        return proba
    
    def predict(self, x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x).reshape(-1)
        pred = np.rint(proba)
        return pred
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)

    def get_summary(self):
        return self.model.summary()
"""

class Model_SVC:

    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold
        self.scaler = StandardScaler()

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = SVC(**self.params)
        self.model.fit(tr_x,tr_y)

    def predict_proba(self,x):
        x = self.scaler.transform(x)
        proba = np.array(self.model.decision_function(x))
        return proba
    
    def predict(self,x):
        x = self.scaler.transform(x)
        proba = np.array(self.model.decision_function(x))
        pred = np.where(proba<0,0,1)
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))
    
    def get_params(self):
        return self.model.get_params()


class Model_Logistic:
    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold
        self.scaler = StandardScaler()

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(**self.params)
        self.model.fit(tr_x,tr_y)

    def predict_proba(self,x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[:,1]
        return proba
    
    def predict(self,x):
        x = self.scaler.transform(x)
        pred = np.rint(self.model.predict_proba(x)[:,1])
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))

    def get_params(self):
        return self.model.get_params()


class Model_RFC:
    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold
        self.scaler = StandardScaler()

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(tr_x,tr_y)

    def predict_proba(self,x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[:,1]
        return proba
    
    def predict(self, x):
        x = self.scaler.transform(x)
        pred = np.rint(self.model.predict_proba(x)[:,1])
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))
    
    def get_params(self):
        return self.model.get_params()

class Model_RFR:
    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.params['random_state'] = s
        self.fold = fold
        #self.scaler = StandardScaler()

    def fit(self, tr_x, tr_y, va_x, va_y):
        #self.scaler.fit(tr_x)
        #tr_x = self.scaler.transform(tr_x)
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(tr_x,tr_y)
  
    def predict(self, x):
        #x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))
    
    def get_params(self):
        return self.model.get_params()


class Model_KNN:
    def __init__(self, params, s=42, fold=None):
        self.model = None
        self.params = params
        self.seed = s
        self.fold = fold
        self.scaler = StandardScaler()

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = KNeighborsClassifier(**self.params)

        self.model.fit(tr_x,tr_y)

    def predict_proba(self,x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[:,1]
        return proba
    
    def predict(self, x):
        x = self.scaler.transform(x)
        pred = np.rint(self.model.predict_proba(x)[:,1])
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))

    def get_params(self):
        return self.model.get_params()


class Model_NB:
    def __init__(self, s=42, fold=None):
        self.model = None
        self.seed = s
        self.fold = fold
        self.scaler = StandardScaler()
    
    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = BernoulliNB()
        self.model.fit(tr_x,tr_y)

    def predict_proba(self,x):
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[:,1]
        return proba
    
    def predict(self, x):
        x = self.scaler.transform(x)
        pred = np.rint(self.model.predict_proba(x)[:,1])
        return pred
    
    def save(self, path):
        pickle.dump(self.model, open(path + '.pickle', 'wb'))
    
    def load(self, path):
        self.model = pickle.load(open(path + '.pickle', 'rb'))

    def get_params(self):
        return self.model.get_params()

