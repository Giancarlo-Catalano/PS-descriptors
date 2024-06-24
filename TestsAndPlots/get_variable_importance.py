import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
import shap

import sklearn.ensemble
import lime
import lime.lime_tabular


def new_method():
    # Load the data
    data = pd.read_csv(r'/Experimentation/BT/Final/pRefsolutions.csv')

    # subsample the data, at least for now
    data = data.sample(n=1000)

    # Separate features and target variable
    X = data.drop('Fitness', axis=1)
    y = data['Fitness']

    # get the best entry
    best_index = min(range(len(y)), key=lambda i:y.iloc[i])
    print(f"The best is at index {best_index}, with value {y.iloc[best_index]}")


    # Split the data into training and test sets
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, train_size=0.80)

    # Define the network model
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100)

    # Train the model
    rf.fit(train, labels_train)

    # print the errors
    print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))
    print('MSError when predicting the mean', np.mean((labels_train.mean() - labels_test) ** 2))


    # find which features are categorical
    categorical_features = np.arange(X.shape[1])

    # construct explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data = train.to_numpy(),
                                                       mode='regression',
                                                       feature_names=list(X.columns),
                                                       categorical_features=categorical_features,
                                                       class_names=['Fitness'],
                                                       verbose=True)


    # Explain a specific instance
    to_explain = X.iloc[best_index]
    print(f"I will explain {np.array(to_explain.to_numpy(), dtype=int)}")
    exp = explainer.explain_instance(to_explain, rf.predict, num_features=2)
    exp.show_in_notebook(show_table=True)


new_method()