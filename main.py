
import random

from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor, HungaBungaZeroKnowledge, HungaBungaRandomClassifier, HungaBungaRandomRegressor
import pandas as pd
import sys


def main(project_name):
    training = pd.read_csv(r"dataset\{0}\classes\training.csv".format(project_name), sep=';')
    testing = pd.read_csv(r"dataset\{0}\classes\testing.csv".format(project_name), sep=';')
    training_y = training['Bugged']
    training_X = training.drop('Bugged', axis=1)
    testing_y = testing['Bugged']
    testing_X = testing.drop('Bugged', axis=1)
    clf = HungaBungaClassifier(brain=True)
    clf.fit(training_X, training_y)
    print(clf.predict(testing_X))


if __name__ == "__main__":
    main(sys.argv[1])