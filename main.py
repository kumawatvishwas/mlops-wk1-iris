from google.cloud import aiplatform
import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


import pickle
import joblib

PROJECT_ID = "focused-catfish-459803-m9"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
BUCKET_URI = f"gs://mlops-21f1001848"  # @param {type:"string"}


aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"  # @param {type:"string"}
REPOSITORY = "iris-classifier-repo"  # @param {type:"string"}
IMAGE = "iris-classifier-img"  # @param {type:"string"}
MODEL_DISPLAY_NAME = "iris-classifier"  # @param {type:"string"}

# Set the defaults if no names were specified
if MODEL_ARTIFACT_DIR == "[your-artifact-directory]":
    MODEL_ARTIFACT_DIR = "custom-container-prediction-model"

if REPOSITORY == "[your-repository-name]":
    REPOSITORY = "custom-container-prediction"

if IMAGE == "[your-image-name]":
    IMAGE = "sklearn-fastapi-server"

if MODEL_DISPLAY_NAME == "[your-model-display-name]":
    MODEL_DISPLAY_NAME = "sklearn-custom-container"



data = pd.read_csv('data/iris.csv')
data.head(5)

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))



joblib.dump(mod_dt, "artifacts/model.joblib")