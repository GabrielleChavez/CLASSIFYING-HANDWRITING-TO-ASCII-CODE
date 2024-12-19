# **CLASSIFYING HANDWRITING TO ASCII CODE**

Project Summary:
----------------
Our goal is to convert handwritten samples into ascii. We want to compare the efficieny of different machine learning and deep learning models including:
* Random Forest
* XGBoost
* K-Nearest Neighbors
* FeedFoward NN
* Convolutional NM
* Transformer w/CNN Features


Dataset:
-----------
We are using the **AlphaNum** dataset that consist of handwritten letters that are 28x28 pixel images in grayscale. 
The dataset that can be found at: https://www.kaggle.com/datasets/lopalp/alphanum/data

Logistics:
------------------
The neural network models can be found under nn.py. The other machine learning models can be found under classifiers.py. The preprocessing functions can be found under preprocess.py. The testing functions are in testing_models.py file. The rest of the .py and .ipynb files might not be used, they are there in case you want to check our progress.

For the graders: We have provided a Jupyter Notebook called **run_me.py** where we step-by-step explain what we did and provide the code for you to run. You will be able to reproduce the results just by following this code. 

The write-up can be found in **report.pdf** and the slides can be found in **presentation.pptx**.

The dataset is split into folders train, test and validation.
