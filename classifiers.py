import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def train_xgboost(X_train, y_train, n_estimators, max_depth):
   """
   Create and Train the xgboost model.

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      n_estimators (int): number of trees the model will have
      max_depth (int): limits the depth of each tree witihin the model
   
   Returns:
      model: XGBClassifier model
   """
   num_classes = len(np.unique(y_train))
   model = xgb.XGBClassifier(objective='multi:softprob',num_class=num_classes,booster='gbtree',eval_metric= 'mlogloss', random_state=12,n_estimators= n_estimators, max_depth = max_depth)
   model.fit(X_train, y_train)
   return model

def train_random_forest(X_train, y_train, n_estimators=50):
   """
   Create and Train RF model

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      n_estimators (int): number of decision trees the model will use
   
   Returns:
      clf: RandomForest Classfier model
   """
   clf = RandomForestClassifier(n_estimators=n_estimators)
   clf.fit(X_train, y_train)
   return clf

def train_perceptron(X_train, y_train, max_iter=100):
   """
   Create and Train perceptron model

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      max_iter (int): max number of iterations over training data
   Returns:
      perceptron: perceptron model
   """
   perceptron = Perceptron(max_iter=max_iter, random_state=42)
   perceptron.fit(X_train, y_train)
   return perceptron

def train_svm(X_train, y_train, kernel='poly'):
   """
   Create and Train svm model

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      kernal (String): classification type
   Returns:
      perceptron: perceptron model
   """
   svm_model = svm.SVC(kernel=kernel)
   svm_model.fit(X_train, y_train)
   return svm_model

def train_knn(X_train, y_train, n_neighbors=94, weight="distance"):
   """
   Create and Train knn model

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      n_neighbors (int): number of classes
      weight (string): different weight labels
   Returns:
      knn: knn model
   """
   knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
   knn.fit(X_train, y_train)
   return knn