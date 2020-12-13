
import random

from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor, HungaBungaZeroKnowledge, HungaBungaRandomClassifier, HungaBungaRandomRegressor
import pandas as pd
import sys
import os


def main(project_name, ind=None):
    training = pd.read_csv(os.path.normpath(os.path.realpath(r"dataset\{0}\classes\training.csv".format(project_name))), sep=';')
    testing = pd.read_csv(os.path.normpath(os.path.realpath(r"dataset\{0}\classes\testing.csv".format(project_name))), sep=';')
    training_y = training['Bugged'].apply(lambda x: 1 if x else 0)
    training_X = training.drop('Bugged', axis=1)
    testing_y = testing['Bugged'].apply(lambda x: 1 if x else 0)
    testing_X = testing.drop('Bugged', axis=1)
    clf = HungaBungaClassifier(brain=True, ind=ind)
    clf.fit(training_X, training_y)


if __name__ == "__main__":
    ind = None
    if len(sys.argv) >= 3:
        ind = sys.argv[2]
    main(sys.argv[1], ind)