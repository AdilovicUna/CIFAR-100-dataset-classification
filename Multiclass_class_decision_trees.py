import lzma
import pickle
import sys
import os

import numpy as np

from LoadDataset  import CIFAR100
from Results import show_results
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cross_validation(data, target):
    depth = 1
    best_mean = -1
    for d in range (3,11):
        mean = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=d), data, target, cv=7))
        if mean > best_mean:
            best_mean = mean    
            depth = d
    
    return depth

def main(test):
    dataset = CIFAR100(['bottle', 'bowl', 'can', 'cup', 'plate'], coarse=False)
    filename = 'multi_class_decision_trees'

    # Train the model
    if not test:
        train_data, train_target = dataset.train_data, dataset.train_target

        # normalize the data and preform the dimensionality reduction
        sc = StandardScaler()
        pca = PCA(n_components=0.95)

        train_data = sc.fit_transform(train_data)
        train_data = pca.fit_transform(train_data)

        with lzma.open('pca/' + filename, "wb") as transform:
            pickle.dump(pca, transform)

        # find the best suitable k using cross_validation
        depth = cross_validation(train_data, train_target)

        # train the model using DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=depth)

        parameter_space = { 
            'criterion': ['gini','entropy'],
        }

        dt_grid = GridSearchCV(dt, parameter_space, n_jobs=-1, cv=7)
        model = dt_grid.fit(train_data, train_target)

        # serialize the model
        with lzma.open('models/' + filename + '.model', "wb") as model_file:
            pickle.dump(model, model_file)

    # Test
    else:
        test_data, test_target = dataset.test_data, dataset.test_target

        # normalize the data and preform the dimensionality reduction (using the PCA that was fitted during training)
        sc = StandardScaler()
        test_data = sc.fit_transform(test_data)

        with lzma.open('pca/' + filename, "rb") as transform:
            pca = pickle.load(transform)
        
        test_data = pca.transform(test_data)

        with lzma.open('models/' + filename + '.model', "rb") as model_file:
            model = pickle.load(model_file)
            
        prediction = model.predict(test_data)

        show_results(test_data, test_target, prediction, model, filename)


if __name__ == "__main__":
    # create a directory for the models
    path = 'models'
    if not os.path.exists(path):
        os.makedirs(path)

    # create a directory for the pca
    path = 'pca'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # create a directory for the plots
    path = 'plots'
    if not os.path.exists(path):
        os.makedirs(path)
        
    # determine if we have to train the model or test it
    test = False
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test = True

    main(test)