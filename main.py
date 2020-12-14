
import random

from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor, HungaBungaZeroKnowledge, HungaBungaRandomClassifier, HungaBungaRandomRegressor
import pandas as pd
import sys
import os
# from tpot import TPOTClassifier
import json
import sklearn.metrics as metrics


def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.to_list(), y_score)
    return metrics.auc(recall, precision)


def eval(model, classes, X, y):
    predictions_proba = list(zip(*model.predict_proba(X)))
    y_prob = dict(zip(classes, predictions_proba))[1]
    return pr_auc_score(y, y_prob)


def main(project_name, ind=None):
    training = pd.read_csv(os.path.normpath(os.path.realpath(r"dataset\{0}\classes\training.csv".format(project_name))), sep=';')
    testing = pd.read_csv(os.path.normpath(os.path.realpath(r"dataset\{0}\classes\testing.csv".format(project_name))), sep=';')
    training_y = training['Bugged'].apply(lambda x: 1 if x else 0)
    training_X = training.drop('Bugged', axis=1)
    testing_y = testing['Bugged'].apply(lambda x: 1 if x else 0)
    testing_X = testing.drop('Bugged', axis=1)
    clf = HungaBungaClassifier(brain=True, ind=ind, scoring=metrics.make_scorer(pr_auc_score, needs_proba=True))
    clf.fit(training_X, training_y)
    model = clf.model
    print(json.dumps({'winner': model.__class__.__name__, 'score': '%0.3f' % eval(model, model.classes_, testing_X, testing_y)}))

    # tpot = TPOTClassifier(max_time_mins=1, scoring=metrics.make_scorer(pr_auc_score, needs_proba=True))
    # tpot.fit(training_X, training_y)
    # model = tpot.fitted_pipeline_._final_estimator
    # classes = None
    # for s in tpot.fitted_pipeline_.steps:
    #     if hasattr(s[1], 'classes_'):
    #         classes = s[1].classes_
    # print(json.dumps({'winner': model.__class__.__name__, 'score': '%0.3f' % eval(tpot, classes, testing_X, testing_y)}))


if __name__ == "__main__":
    ind = None
    if len(sys.argv) >= 3:
        ind = sys.argv[2]
    main(sys.argv[1], ind)