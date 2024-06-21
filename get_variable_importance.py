import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
import shap

def old_method():
    # Load the data
    data = pd.read_csv(r'C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\Final\pRefsolutions.csv')

    # Separate features and target variable
    X = data.drop('Fitness', axis=1)
    y = data['Fitness']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Use SHAP's DeepExplainer to explain the model's predictions
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Compute the mean absolute SHAP values for each feature
    shap_importance = np.mean(np.abs(shap_values), axis=0).flatten()

    # Print the variable importance
    for var_name, importance in zip(X.columns, shap_importance):
        print(f"{var_name}: {importance}")

import sklearn.ensemble
import lime
import lime.lime_tabular


def new_method():
    # Load the data
    data = pd.read_csv(r'C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\Final\pRefsolutions.csv')

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