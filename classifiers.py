import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# Fit and transform ytrain and ytest to sequential labels
def train_xgboost(Xtrain, ytrain, n_estimators, max_depth):
   num_classes = len(np.unique(ytrain))
   model = xgb.XGBClassifier(objective='multi:softprob',num_class=num_classes,booster='gbtree',eval_metric= 'mlogloss', random_state=12,n_estimators= n_estimators, max_depth = max_depth)
   model.fit(Xtrain, ytrain)
   return model

def train_random_forest(x_train, y_train, n_estimators=50):
   clf = RandomForestClassifier(n_estimators=n_estimators)
   clf.fit(x_train, y_train)
   return clf

def train_perceptron(x_train, y_train):
   perceptron = Perceptron(max_iter=100, random_state=42)
   perceptron.fit(x_train, y_train)
   return perceptron

def train_svm(x_train, y_train, kernel='poly'):
   svm_model = svm.SVC(kernel=kernel)
   svm_model.fit(x_train, y_train)
   return svm_model

def train_knn(x_train, y_train, n_neighbors=3):
   knn = KNeighborsClassifier(n_neighbors=n_neighbors)
   knn.fit(x_train, y_train)
   return knn