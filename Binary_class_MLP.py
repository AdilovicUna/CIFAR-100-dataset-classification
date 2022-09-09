import lzma
import pickle
import sys
import os

from LoadDataset  import CIFAR100
from Results import show_results
from sklearn.neural_network import MLPClassifier


def main(test):
    dataset = CIFAR100(['aquatic_mammals', 'non-insect_invertebrates'])
    filename = 'bin_class_mlp'

    # Train the model
    if not test:
        train_data, train_target = dataset.train_data, dataset.train_target

        # train the model using MLPClassifier
        mlp = MLPClassifier(max_iter=100)

        model = mlp.fit(train_data, train_target)

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