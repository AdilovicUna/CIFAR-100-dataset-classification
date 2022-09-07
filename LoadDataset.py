import pickle
from tkinter import S
import pandas as pd
import numpy as np

class CIFAR100:
    def __init__(self, 
            classes, # names of the classes we want to extract
            coarse = True # True if we want to work with the super classes, False if we need fine classes
            ):

        # get train and test data
        train = self.unpickle('train')
        test = self.unpickle('test')

        # map class names
        meta = self.unpickle('meta')
        s = 'coarse_label_names' if coarse else 'fine_label_names'
        name_dict =  {k: v for v, k in enumerate(meta[s])}

        # extract train and test data for required classes
        labels = 'coarse_labels' if coarse else 'fine_labels'

        self.train_data, self.train_target = self.get_classes(train['data'], np.array(train[labels]), classes, name_dict)
        self.test_data, self.test_target = self.get_classes(test['data'], np.array(test[labels]), classes, name_dict)


    def get_classes(self, data, target, classes, name_dict):
        all_classes_indexes = []
        for name in classes:
            class_indexes = np.argwhere(target == name_dict[name])
            class_indexes = class_indexes.reshape(len(class_indexes),)
            all_classes_indexes.append(class_indexes)
        
        all_classes_indexes = np.concatenate(all_classes_indexes)

        return (data[all_classes_indexes], target[all_classes_indexes])


    def unpickle(self, file):
        with open(r'cifar-100-python/' + file, 'rb') as fo:
            return pickle.load(fo, encoding='latin1')