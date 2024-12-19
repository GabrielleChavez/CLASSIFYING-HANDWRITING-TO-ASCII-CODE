import numpy as np
import xgboost as xgb
import numpy as np
import matplotlib as plt
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

def train_knn(X_train, y_train, n_neighbors=94):
   """
   Create and Train knn model

   Parameters:
      X_train (Numpy Array): training data
      y_train (Numpy Array): training labels
      n_neighbors (int): number of datapoints
   Returns:
      knn: knn model
   """
   knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
   knn.fit(X_train, y_train)
   return knn

def XGBoost_Estimators_Plots(estimators, f1_list_estimators, acc_list_estimators, prec_list_estimators, recall_list_estimators):
   #XGBoost Estimator Variant Plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 4))  # 1 row, 5 columns

   # Plot data on each subplot
   axes[0, 0].plot(estimators, f1_list_estimators)
   axes[0, 0].set_title("F1 Score Progression")

   axes[0, 1].plot(estimators, acc_list_estimators)
   axes[0, 1].set_title("Accuracy Progression")

   axes[1, 0].plot(estimators, prec_list_estimators)
   axes[1, 0].set_title("Precision Score Progression")

   axes[1, 1].plot(estimators, recall_list_estimators)
   axes[1, 1].set_title("Recall Score Progression")

   # Adjust layout
   plt.tight_layout()

   # Show the figure
   plt.show()

def XGBoost_Depth_Plots(depth, f1_list_depth, acc_list_depth, prec_list_depth, recall_list_depth):
   #XGBoost Estimator Variant Plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 4))  # 1 row, 5 columns

   # Plot data on each subplot
   axes[0, 0].plot(depth, f1_list_depth)
   axes[0, 0].set_title("F1 Score Progression- XGBoost")

   axes[0, 1].plot(depth, acc_list_depth)
   axes[0, 1].set_title("Accuracy Progression- XGBoost")

   axes[1, 0].plot(depth, prec_list_depth)
   axes[1, 0].set_title("Precision Score Progression- XGBoost")

   axes[1, 1].plot(depth, recall_list_depth)
   axes[1, 1].set_title("Recall Score Progression- XGBoost")

   # Adjust layout
   plt.tight_layout()

   # Show the figure
   plt.show()

def RandomForest_plots(depth, f1_list_estimators, acc_list_estimators, prec_list_estimators, recall_list_estimators):
   #Random Forest Plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 4))  # 1 row, 5 columns

   # Plot data on each subplot
   axes[0, 0].plot(depth, f1_list_estimators)
   axes[0, 0].set_title("F1 Score Progression- Random Forest")

   axes[0, 1].plot(depth, acc_list_estimators)
   axes[0, 1].set_title("Accuracy Progression- Random Forest")

   axes[1, 0].plot(depth, prec_list_estimators)
   axes[1, 0].set_title("Precision Score Progression- Random Forest")

   axes[1, 1].plot(depth, recall_list_estimators)
   axes[1, 1].set_title("Recall Score Progression- Random Forest")

   plt.tight_layout()

   plt.show()

def KNN_Plots(depth, f1_list_neighbors, acc_list_neighbors, prec_list_neighbors, recall_list_neighbors):
   #KNN Plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 4))  # 1 row, 5 columns

   axes[0, 0].plot(depth, f1_list_neighbors)
   axes[0, 0].set_title("F1 Score Progression- KNN")

   axes[0, 1].plot(depth, acc_list_neighbors)
   axes[0, 1].set_title("Accuracy Progression- KNN")

   axes[1, 0].plot(depth, prec_list_neighbors)
   axes[1, 0].set_title("Precision Score Progression- KNN")

   axes[1, 1].plot(depth, recall_list_neighbors)
   axes[1, 1].set_title("Recall Score Progression- KNN")

   plt.tight_layout()

   plt.show()

def SVM_plot(kernels, f1_list_kernel, acc_list_kernel, prec_list_kernel, recall_list_kernel):
   #SVM Plot
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   axes[0, 0].bar(kernels, f1_list_kernel, color='skyblue')
   axes[0, 0].set_title("F1 Scores by Kernel")

   axes[0, 1].bar(kernels, acc_list_kernel, color='skyblue')
   axes[0, 1].set_title("Accuracy by Kernel")

   axes[1, 0].bar(kernels, prec_list_kernel, color='skyblue')
   axes[1, 0].set_title("Precision by Kernel")

   axes[1, 1].bar(kernels, recall_list_kernel, color='skyblue')
   axes[1, 1].set_title("Recall by Kernel")


   plt.tight_layout()

   plt.show()
