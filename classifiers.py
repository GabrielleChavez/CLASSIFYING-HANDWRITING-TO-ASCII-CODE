import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Fit and transform ytrain and ytest to sequential labels
def xgboost_model(Xtrain, ytrain, n_estimators, max_depth):
   num_classes = len(np.unique(ytrain))
   model = xgb.XGBClassifier(objective='multi:softprob',num_class=num_classes,booster='gbtree',eval_metric= 'mlogloss', random_state=12,n_estimators= n_estimators, max_depth = max_depth)
   model.fit(Xtrain, ytrain)
   return model

def xgboost_test(model, Xtest, ytest):
   y_pred = model.predict(Xtest)
   f1 = f1_score(ytest, y_pred, labels = range(0,94), average='maco')
   acc = accuracy_score(ytest, y_pred)
   cm = confusion_matrix(ytest, y_pred)
   prec = precision_score(ytest, y_pred, labels = range(0,94), average = 'macro')
   recall = recall_score(ytest, y_pred, labels = range(0,94), average = 'macro')
   return acc, prec, recall, f1, cm

def train_random_forest(x_train, y_train, n_estimators=50):
   clf = RandomForestClassifier(n_estimators=n_estimators)
   clf.fit(x_train, y_train)
   return clf

def test_random_forest(clf, x_test, y_test):
   y_pred = clf.predict(x_test)
   accuracy = clf.score(x_test, y_test)
   precision = precision_score(y_test, y_pred, average='macro')
   recall = recall_score(y_test, y_pred, average='macro')
   f1 = f1_score(y_test, y_pred, average='macro')
   conf_matrix = confusion_matrix(y_test, y_pred)
   return accuracy, precision, recall, f1, conf_matrix