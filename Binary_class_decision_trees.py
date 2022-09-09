import lzma
import pickle
import sys
import os

import numpy as np

from LoadDataset  import CIFAR100
from Results import show_results
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def cross_validation(data, target):
    depth = 1
    best_mean = -1
    for d in range (3,21):
        mean = np.mean(cross_val_score(DecisionTreeClassifier(max_depth=d), data, target, cv=7))
        if mean > best_mean:
            best_mean = mean
            depth = d
    
    return depth

def main(test):
    dataset = CIFAR100(['aquatic_mammals', 'non-insect_invertebrates'])
    filename = 'bin_class_decision_trees'

    # Train the model
    if not test:
        train_data, train_target = dataset.train_data, dataset.train_target

        # find the best suitable k using cross_validation
        d = cross_validation(train_data, train_target)

        # train the model using DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=d)
        model = dt.fit(train_data, train_target)

        # serialize the model
        with lzma.open('models/' + filename + '.model', "wb") as model_file:
            pickle.dump(model, model_file)

    # Test
    else:
        test_data, test_target = dataset.test_data, dataset.test_target

        with lzma.open('models/' + filename + '.model', "rb") as model_file:
            model = pickle.load(model_file)

        prediction = model.predict(test_data)
        
        show_results(test_data, test_target, prediction, model, filename)


if __name__ == "__main__":

    # create a directory for the models
    path = 'models'
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