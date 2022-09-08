import lzma
import pickle
import sys
import os

import numpy as np

from LoadDataset  import CIFAR100
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

def cross_validation(data, target):
    best_k = 1
    best_mean = -1
    for k in range (1,11):
        mean = np.mean(cross_val_score(KNeighborsClassifier(k), data, target, cv=5))
        if mean > best_mean:
            best_mean = mean
            best_k = k
    
    return best_k


def evaluate(predictions, pred_probabilities, test_target, classes):
    # Compute accuracy
    print("Accuracy Score: ", accuracy_score(test_target, predictions))

    # Compute the confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_target, predictions)
    print("Confusion Matrix: ")
    print(confusion_matrix)

    # Compute macro-averaged precision and recall values
    p, r, _, _ = precision_recall_fscore_support(test_target, predictions, average='macro')
    print("Precision: ", p)
    print("Recall", r)

    # Plotting the results into a precision-recall curve space
    precision, recall, _ = precision_recall_curve(test_target, pred_probabilities[:, 4], pos_label=classes[4])
    
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()



def get_best_k(train_data, train_target):
    """
        This method find the best value for K in KNN classfier
        using the cross_val_score and returns the best parameter
    """
    num_folds = 10
    # Prepare the K paramters
    k_range = list(range(1, 31))
    # Cross validation to choose k from 1 to 31.
    k_scores = []
    for i in k_range:
        model = KNeighborsClassifier(n_neighbors=i, weights="distance")
        cv_scores = cross_val_score(model, train_data, train_target, cv=num_folds, scoring="accuracy")
        k_scores.append(np.mean(cv_scores))

    # Choose hyperparameter with lowest mean cross validation error
    return np.argmax(k_scores)



def main(test):
    dataset = CIFAR100(['bottle', 'bowl', 'can', 'cup', 'plate'], coarse=False)

    # Train the model
    if not test:
        train_data, train_target = dataset.train_data, dataset.train_target

        # find the best suitable k using cross_validation
        k = cross_validation(train_data, train_target)

        # train the model using KNeighborsClassifier
        knn = KNeighborsClassifier(k)
        model = knn.fit(train_data, train_target)

        # serialize the model
        with lzma.open('models/multi_class_knn.model', "wb") as model_file:
            pickle.dump(model, model_file)

    # Test
    else:
        test_data, test_target = dataset.test_data, dataset.test_target

        with lzma.open('models/multi_class_knn.model', "rb") as model_file:
            model = pickle.load(model_file)

        prediction = model.predict(test_data)

        pred_probabilities = model.predict_proba(test_data)

        # Evaluating the classification model
        evaluate(prediction, pred_probabilities, test_target, model.classes_)


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