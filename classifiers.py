import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


# Fit and transform ytrain and ytest to sequential labels
def xgboost_model(Xtrain, ytrain, n_estimators, max_depth):
   label_encoder = LabelEncoder()
   ytrain_encoded = label_encoder.fit_transform(ytrain)
   num_classes = len(np.unique(ytrain_encoded))
   model = xgb.XGBClassifier(objective='multi:softprob',num_class=num_classes,booster='gbtree',eval_metric= 'mlogloss', random_state=12,n_estimators= n_estimators, max_depth = max_depth)
   model.fit(Xtrain, ytrain_encoded)
   return model

def xgboost_test(model, Xtest, ytest):
   label_encoder = LabelEncoder()
   y_pred_encoded = model.predict(Xtest)
   y_pred = label_encoder.inverse_transform(y_pred_encoded)
   f1 = f1_score(ytest, y_pred, labels = range(0,94), average='maco')
   acc = accuracy_score(ytest, y_pred)
   cm = confusion_matrix(ytest, y_pred)
   prec = precision_score(ytest, y_pred, labels = range(0,94), average = 'macro')
   recall = recall_score(ytest, y_pred, labels = range(0,94), average = 'macro')
   accuracy = accuracy_score(ytest, y_pred)
   return f1, acc, cm, prec, recall, accuracy