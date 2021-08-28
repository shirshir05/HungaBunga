import glob
import pickle
import random

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import name_features
from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor, HungaBungaZeroKnowledge, HungaBungaRandomClassifier, \
    HungaBungaRandomRegressor
import pandas as pd
import sys
import os
# from tpot import TPOTClassifier
import json
import sklearn.metrics as metrics
import numpy as np


def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.to_list(), y_score)
    return metrics.auc(recall, precision)


def f1_score(y_true, y_score):
    return metrics.f1_score(y_true, y_score)


def evaluate_dense(y_true, y_pred, classes, predicitons_proba, tensor=False):
    scores = {}
    if tensor:
        y_prob_true = predicitons_proba
    else:
        y_prob_true = dict(zip(classes, predicitons_proba))['1.0']
    scores['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
    scores['precision_score'] = metrics.precision_score(y_true, y_pred)
    scores['recall_score'] = metrics.recall_score(y_true, y_pred)
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)
    scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_prob_true)
    scores['tn'], scores['fp'], scores['fn'], scores['tp'] = [int(i) for i in
                                                              list(confusion_matrix(y_true, y_pred).ravel())]
    return scores


def eval(model, classes, X, y):
    scores = {}
    predictions_proba = list(zip(*model.predict_proba(X)))
    y_prob_true = dict(zip(classes, predictions_proba))[1]
    predictions = model.predict(X)

    scores['accuracy_score'] = metrics.accuracy_score(y, predictions)
    scores['precision_score'] = metrics.precision_score(y, predictions)
    scores['recall_score'] = metrics.recall_score(y, predictions)
    scores['f1_score'] = metrics.f1_score(y, predictions)
    scores['roc_auc_score'] = metrics.roc_auc_score(y, y_prob_true)
    scores['pr_auc_score'] = pr_auc_score(y, y_prob_true)
    scores['tn'], scores['fp'], scores['fn'], scores['tp'] = [int(i) for i in
                                                              list(confusion_matrix(y, predictions).ravel())]

    return scores


def dense_model(model, name, testing_X, testing_y):
    import tensorflow
    from tensorflow.keras import activations, regularizers
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras.models import Sequential
    kernel_initializer_0 = tensorflow.keras.initializers.constant(model.coefs_[0])
    bias_initializer_0 = tensorflow.keras.initializers.constant(model.intercepts_[0])

    kernel_initializer_1 = tensorflow.keras.initializers.constant(model.coefs_[1])
    bias_initializer_1 = tensorflow.keras.initializers.constant(model.intercepts_[1])

    kernel_initializer_2 = tensorflow.keras.initializers.constant(model.coefs_[2])
    bias_initializer_2 = tensorflow.keras.initializers.constant(model.intercepts_[2])

    model = Sequential()
    model.add(Dense(512, activation=activations.relu,
                    kernel_regularizer=regularizers.l2(0.0001), kernel_initializer=kernel_initializer_0,
                    bias_initializer=bias_initializer_0))
    model.add(Dense(1024, activation=activations.relu, kernel_regularizer=regularizers.l2(0.0001),
                    kernel_initializer=kernel_initializer_1,
                    bias_initializer=bias_initializer_1))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer_2,
                    bias_initializer=bias_initializer_2))

    predictions_proba = model.predict(testing_X)[:, 0]
    predictions = np.where(predictions_proba < 0.5, 0, 1)
    score_dense = evaluate_dense(testing_y, predictions, None, predictions_proba, tensor=True)
    with open(r"./results/bic_scores_dense_" + name + ".json", 'w') as f:
        json.dump({**score_dense}, f)


def main(ind=0, rf=False):
    name_project = "cayenne"

    df = pd.read_csv(os.path.join("dataset", name_project, "train.csv"))
    df = df.iloc[:, 1:]

    y_train = df.pop('commit insert bug?')
    X_train = df
    features_check_before_pre_process = name_features.JAVADIFF_FEATURES_DIFF + name_features.JAVADIFF_FEATURES_STATEMENT + \
                                        name_features.JAVADIFF_FEATURES_AST + name_features.PMD_FEATURES + name_features.STATIC_FEATURES
    features_check = [col for col in X_train.columns if col in features_check_before_pre_process]

    X_train = X_train[features_check]

    df_test = pd.read_csv(os.path.join("dataset", name_project, "test.csv"))
    df_test = df_test.iloc[:, 1:]
    y_test = df_test.pop('commit insert bug?')

    X_test = df_test
    X_test = X_test[features_check]

    # scaler_min_max = MinMaxScaler((0, 1))
    # scaler_min_max.fit(X_train)
    # X_train = pd.DataFrame(scaler_min_max.transform(X_train))
    # X_test = pd.DataFrame(scaler_min_max.transform(X_test))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    # X_test = X_test.apply(np.log, axis=0)
    # X_train = X_train.apply(np.log, axis=0)

    testing_X = X_test
    testing_y = y_test
    training_X = X_train
    training_y = y_train

    clf = HungaBungaClassifier(brain=True, ind=int(ind), scoring=metrics.make_scorer(f1_score, needs_proba=True))
    clf.fit(training_X, training_y, RF=rf)
    model = clf.model

    score = eval(model, model.classes_, testing_X, testing_y)
    # print(json.dumps({'model': clf.combination, 'score': '%0.3f' % score}))
    print(json.dumps({**clf.combination, **score}))
    with open(r"./results/bic_scores_" + str(ind) + ".json", 'w') as f:
        json.dump({**clf.combination, **score}, f)

    if not rf:
        import matplotlib.pyplot as plt
        plt.plot(model.loss_curve_)

        if not os.path.exists(os.path.join(".", "results", "loss")):
            os.mkdir(os.path.join(".", "results", "loss"))
        if not os.path.exists(os.path.join(".", "results", "save_model")):
            os.mkdir(os.path.join(".", "results", "save_model"))

        plt.savefig(r"./results/loss/loss_" + str(ind) + "_" + str(len(model.loss_curve_)) + ".png")

        # # TODO: Save model
        filename_pkl = r"./results/save_model/save_model_test" + str(ind) + ".pkl"
        pickle.dump(model, open(filename_pkl, 'wb'))

        # check save model
        # loaded_model = pickle.load(open(filename_pkl, 'rb'))
        # score_pkl = eval(loaded_model, model.classes_, testing_X, testing_y)
        # with open(r"./results/bic_scores_score_pkl" + str(ind) + ".json", 'w') as f:
        #     json.dump({**clf.combination, **score_pkl}, f)


if __name__ == "__main__":
    ind = None
    ind = sys.argv[1]
    import ast

    rf = ast.literal_eval(sys.argv[2])
    main(ind, rf)
    # for ind in range(0, 49):
    #     main(ind, False)
