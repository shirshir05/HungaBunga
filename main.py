
import random

from sklearn.model_selection import train_test_split

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

def f1_score(y_true, y_score):
    return metrics.f1_score(y_true, y_score)


def eval(model, classes, X, y):
    predictions_proba = list(zip(*model.predict_proba(X)))
    y_prob = dict(zip(classes, predictions_proba))[0]
    return pr_auc_score(y, y_prob)


def preprocessing(path):
    df = pd.read_csv(path)
    df.rename(columns={'blame commit': 'commit insert bug?'}, inplace=True)
    # Remove redundant columns
    df = df.iloc[:, 4:]
    del df["bugfix_commit"]
    del df["file"]
    del df['added_lines+removed_lines']
    del df['added_lines-removed_lines']

    # Remove commit that doesn't change lines
    df = df[df['current_changed_used_lines'] != 0]
    print(f"Remove commit that doesn't change lines")
    print(df.shape)

    # Remove parent and current features
    features_to_drop = ['parent', 'current']
    df = df.drop(columns=list(filter(lambda c: any(map(lambda f: f in c, features_to_drop)), df.columns)), axis=1)
    print(f"Remove parent and current features")
    print(df.shape)

    # Remove test file
    def filter_fn(row):
        if "test" in row['file_name'].lower():
            return False
        else:
            return True

    df = df[df.apply(filter_fn, axis=1)]
    del df["file_name"]
    print(f"Remove test file")
    print(df.shape)
    # Remove col that contain more than 95% zeros
    df.replace(0, pd.np.nan, inplace=True)
    df.dropna(axis=1, how='any', thresh=0.05 * df.shape[1], inplace=True)
    df.replace(pd.np.nan, 0, inplace=True)
    print(f"Remove col that contain more than 95% zeros")
    print(df.shape)

    print(f"Number bug")
    print(df['commit insert bug?'].sum())
    print(f"percentage bug")
    print(df['commit insert bug?'].sum() / df.shape[0])
    return df


def main(project_name, ind=None):
    # df = preprocessing(r"dataset\{0}\all_data.csv".format(project_name))
    df = pd.read_csv(r"dataset\{0}\preprocessing.csv".format(project_name))

    print(df.shape)

    df.to_csv(r"dataset\{0}\preprocessing.csv".format(project_name))
    y = df.pop('commit insert bug?')
    X = df


    training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size=.2, random_state=12, stratify=y)
    # testing = pd.read_csv(os.path.normpath(os.path.realpath(r"dataset\{0}\classes\testing.csv".format(project_name))), sep=';')
    # training_y = training['Bugged'].apply(lambda x: 1 if x else 0)
    # training_X = training.drop('Bugged', axis=1)
    # testing_y = testing['Bugged'].apply(lambda x: 1 if x else 0)
    # testing_X = testing.drop('Bugged', axis=1)
    clf = HungaBungaClassifier(brain=True, ind=ind, scoring=metrics.make_scorer(f1_score, needs_proba=True))
    clf.fit(training_X, training_y)
    model = clf.model
    print(json.dumps({'model': model.__class__.__name__, 'score': '%0.3f' % eval(model, model.classes_, testing_X, testing_y)}))

    # # tpot = TPOTClassifier(max_time_mins=1, scoring=metrics.make_scorer(pr_auc_score, needs_proba=True))
    # # tpot.fit(training_X, training_y)
    # # model = tpot.fitted_pipeline_._final_estimator
    # # classes = None
    # # for s in tpot.fitted_pipeline_.steps:
    # #     if hasattr(s[1], 'classes_'):
    # #         classes = s[1].classes_
    # # print(json.dumps({'winner': model.__class__.__name__, 'score': '%0.3f' % eval(tpot, classes, testing_X, testing_y)}))
    #

if __name__ == "__main__":
    ind = None
    if len(sys.argv) >= 3:
        ind = sys.argv[2]
    main(sys.argv[1], ind)