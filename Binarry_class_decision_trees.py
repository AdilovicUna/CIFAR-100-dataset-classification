import lzma
import pickle
import sys
import os

import numpy as np

from LoadDataset  import CIFAR100
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def cross_validation(data, target):
    best_k = 1
    best_mean = -1
    for k in range (1,11):
        mean = np.mean(cross_val_score(KNeighborsClassifier(k), data, target, cv=5))
        if mean > best_mean:
            best_mean = mean
            best_k = k
    
    return best_k

def main(test):
    dataset = CIFAR100(['aquatic_mammals', 'non-insect_invertebrates'])

    # Train the model
    if not test:
        train_data, train_target = dataset.train_data, dataset.train_target

        # find the best suitable k using cross_validation
        k = cross_validation(train_data, train_target)

        # train the model using KNeighborsClassifier
        knn = KNeighborsClassifier(k)
        model = knn.fit(train_data, train_target)

        # serialize the model
        with lzma.open('models/bin_class_knn.model', "wb") as model_file:
            pickle.dump(model, model_file)

    # Test
    else:
        test_data, test_target = dataset.test_data, dataset.test_target

        with lzma.open('models/buin_class_knn.model', "rb") as model_file:
            model = pickle.load(model_file)

        prediction = model.predict(test_data)

if __name__ == "__main__":

    # create a directory for the models
    path = 'models'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # determine if we have to train the model or test it
    test = False
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test = True

    main(test)