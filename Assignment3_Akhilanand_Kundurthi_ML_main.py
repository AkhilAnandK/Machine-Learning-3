print("Beginning of assignment 3\n")
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.keras import models,layers

#Loading Hill Valley dataset
dataset_1=fetch_openml(data_id=1479)

#Loading steel-plates-fault dataset
dataset_2=fetch_openml(data_id=1504)

# one-hot encoding for Hill Valley target
enc = OneHotEncoder(sparse=False)
tmp_1 = [[x] for x in dataset_1.target]
ohe_target_1= enc.fit_transform(tmp_1)

#one-hot encoding for steel-plates-fault target
tmp_2 = [[x] for x in dataset_2.target]
ohe_target_2= enc.fit_transform(tmp_2)

#Cross Validation
kfolds = KFold(n_splits=10, shuffle=True, random_state=0)

def allModels():
    print("No Hidden Layer model\n")
    #Hill Valley Dataset
    test_fold_accuracy_1=[]
    for train, test in kfolds.split(dataset_1.data, ohe_target_1) :
        # build neural network for Hill Valley Dataset
        nn1= models.Sequential()
        nn1.add(layers.Dense(2, activation="softmax"))
        nn1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 100
        nn1.fit(dataset_1.data.iloc[train], ohe_target_1[train], epochs=n_epochs,verbose=0)
        # testing
        s1 = nn1.evaluate(dataset_1.data.iloc[test], ohe_target_1[test])
        test_fold_accuracy_1.append(s1[1])
    print("\nAverage testing accuracy for no hidden layer model for Hill Valley dataset is :",np.mean(test_fold_accuracy_1))

    #steel-plates-fault dataset
    test_fold_accuracy_2=[]
    for train, test in kfolds.split(dataset_2.data, ohe_target_2) :
        # build neural network for steel-plates-fault dataset
        nn2 = models.Sequential()
        nn2.add(layers.Dense(2, activation="softmax"))
        nn2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn2.fit(dataset_2.data.iloc[train], ohe_target_2[train], epochs=n_epochs,verbose=0)
        # testing
        s2 = nn2.evaluate(dataset_2.data.iloc[test], ohe_target_2[test])
        test_fold_accuracy_2.append(s2[1])
    print("\nAverage testing accuracy for no hidden layer model for steel-plates-fault dataset is :",np.mean(test_fold_accuracy_2))
    
    print("\nOne hidden layer with very few nodes\n")
    test_fold_accuracy_3=[]
    for train, test in kfolds.split(dataset_1.data, ohe_target_1) :
        # build neural network for Hill Valley Dataset
        nn3 = models.Sequential()
        nn3.add(layers.Dense(20, activation="relu", input_dim=100))
        nn3.add(layers.Dense(2, activation="softmax"))
        nn3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn3.fit(dataset_1.data.iloc[train], ohe_target_1[train], epochs=n_epochs,verbose=0)
        # testing
        s3 = nn3.evaluate(dataset_1.data.iloc[test], ohe_target_1[test])
        test_fold_accuracy_3.append(s3[1])
    print("\nAverage testing accuracy for one hidden layer with few nodes model for Hill Valley dataset is :",np.mean(test_fold_accuracy_3))

    #steel-plates-fault dataset
    test_fold_accuracy_4=[]
    for train, test in kfolds.split(dataset_2.data, ohe_target_2) :
        # build neural network for steel-plates-fault dataset
        nn4 = models.Sequential()
        nn4.add(layers.Dense(20, activation="relu", input_dim=33))
        nn4.add(layers.Dense(2, activation="softmax"))
        nn4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn4.fit(dataset_2.data.iloc[train], ohe_target_2[train], epochs=n_epochs,verbose=0)
        # testing
        s4 = nn4.evaluate(dataset_2.data.iloc[test], ohe_target_2[test])
        test_fold_accuracy_4.append(s4[1])
    print("\nAverage testing accuracy for one hidden layer model with few nodes for steel-plates-fault dataset is :",np.mean(test_fold_accuracy_4))
    
    print("\nOne hidden layer with more nodes")
    test_fold_accuracy_5=[]
    for train, test in kfolds.split(dataset_1.data, ohe_target_1) :
        # build neural network for Hill Valley Dataset
        nn5 = models.Sequential()
        nn5.add(layers.Dense(100, activation="relu", input_dim=100))
        nn5.add(layers.Dense(2, activation="softmax"))
        nn5.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn5.fit(dataset_1.data.iloc[train], ohe_target_1[train], epochs=n_epochs,verbose=0)
        # testing
        s5 = nn5.evaluate(dataset_1.data.iloc[test], ohe_target_1[test])
        test_fold_accuracy_5.append(s5[1])
    print("\nAverage testing accuracy for one hidden layer with more nodes model for Hill Valley dataset is :",np.mean(test_fold_accuracy_5))

    #steel-plates-fault dataset
    test_fold_accuracy_6=[]
    for train, test in kfolds.split(dataset_2.data, ohe_target_2) :
        # build neural network for steel-plates-fault dataset
        nn6 = models.Sequential()
        nn6.add(layers.Dense(100, activation="relu", input_dim=33))
        nn6.add(layers.Dense(2, activation="softmax"))
        nn6.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn6.fit(dataset_2.data.iloc[train], ohe_target_2[train], epochs=n_epochs,verbose=0)
        # testing
        s6 = nn6.evaluate(dataset_2.data.iloc[test], ohe_target_2[test])
        test_fold_accuracy_6.append(s6[1])
    print("\nAverage testing accuracy for one hidden layer model with more nodes for steel-plates-fault dataset is :",np.mean(test_fold_accuracy_6))
    
    print("\nTwo hidden layers model")
    test_fold_accuracy_7=[]
    for train, test in kfolds.split(dataset_1.data, ohe_target_1) :
        # build neural network for Hill Valley Dataset
        nn7 = models.Sequential()
        nn7.add(layers.Dense(100, activation="relu", input_dim=100))
        nn7.add(layers.Dense(60, activation='relu'))
        nn7.add(layers.Dense(2, activation="softmax"))
        nn7.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn7.fit(dataset_1.data.iloc[train], ohe_target_1[train], epochs=n_epochs,verbose=0)
        # testing
        s7 = nn7.evaluate(dataset_1.data.iloc[test], ohe_target_1[test])
        test_fold_accuracy_7.append(s7[1])
    print("\nAverage testing accuracy for Two hidden layers model for Hill Valley dataset is :",np.mean(test_fold_accuracy_7))

    #steel-plates-fault dataset
    test_fold_accuracy_8=[]
    for train, test in kfolds.split(dataset_2.data, ohe_target_2) :
        # build neural network for steel-plates-fault dataset
        nn8 = models.Sequential()
        nn8.add(layers.Dense(100, activation="relu", input_dim=33))
        nn8.add(layers.Dense(60, activation='relu'))
        nn8.add(layers.Dense(2, activation="softmax"))
        nn8.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # training
        n_epochs = 200
        nn8.fit(dataset_2.data.iloc[train], ohe_target_2[train], epochs=n_epochs,verbose=0)
        # testing
        s8 = nn8.evaluate(dataset_2.data.iloc[test], ohe_target_2[test])
        test_fold_accuracy_8.append(s8[1])
    print("\nAverage testing accuracy for Two Hidden Layers model for steel-plates-fault dataset is :",np.mean(test_fold_accuracy_8))
    import scipy
    print("\nStatistical Significance")
    print("\nHill Valley Dataset")
    print("\nStatistical Significance between one hidden layer with more nodes and no hidden layer is",scipy.stats.ttest_rel(test_fold_accuracy_5,test_fold_accuracy_1))
    print("\nStatistical Significance between one hidden layer with more nodes and one hidden layer with few nodes is",scipy.stats.ttest_rel(test_fold_accuracy_5,test_fold_accuracy_3))
    print("\nStatistical Significance between one hidden layer with more nodes and two hidden layers is",scipy.stats.ttest_rel(test_fold_accuracy_5,test_fold_accuracy_7))
    print("\n steel-plates-fault dataset\n")
    print("\nStatistical Significance between Two Hidden layer and no hidden layer is",scipy.stats.ttest_rel(test_fold_accuracy_8,test_fold_accuracy_2))
    print("\nStatistical Significance between Two Hidden layer and one hidden layer with few nodes is",scipy.stats.ttest_rel(test_fold_accuracy_8,test_fold_accuracy_4))
    print("\nStatistical Significance between Two Hidden layer and one hidden layer with more nodes is",scipy.stats.ttest_rel(test_fold_accuracy_8,test_fold_accuracy_6))

allModels()
print("\nEnd of Assignment 3")

