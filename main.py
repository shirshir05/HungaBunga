import pickle
import random

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def preprocessing_bugzilla(path):
    df = pd.read_csv(path)
    if 'bugzilla' in path:
        del df["transactionid"]
        del df["commitdate"]
        df = df.rename(columns={'bug': 'commit insert bug?'})
    print(df.shape)
    # Remove col that contain more than 95% zeros
    df.replace(0, np.nan, inplace=True)
    df.dropna(axis=1, how='any', thresh=0.05 * df.shape[1], inplace=True)
    df.replace(np.nan, 0, inplace=True)
    print(f"Remove col that contain more than 95% zeros")
    print(df.shape)
    df = df.drop_duplicates()
    return df

def preprocessing(path):
    df = pd.read_csv(path)
    df.rename(columns={'blame commit': 'commit insert bug?'}, inplace=True)
    # Remove redundant columns
    df = df.iloc[:, 4:]
    del df["bugfix_commit"]
    del df["file"]
    del df['added_lines+removed_lines']
    del df['added_lines-removed_lines']
    del df['commit_sha']
    del df['modification_type']
    del df['is_java']
    del df['is_test']
    del df['added_lines']
    del df['deleted_lines']
    print(df.shape)
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


def main(project_name, ind=0, rf=False):
    # df = preprocessing(r"dataset\{0}\java_diff_without_modification_file_new.csv".format(project_name)).to_csv(r"dataset\{0}\preprocessing_modification_file_new.csv".format(project_name))

    # # TODO: preprocessing_modification_file
    # df = pd.read_csv(r"dataset\{0}\preprocessing_modification_file.csv".format(project_name))
    # df = df.iloc[:, 1:]
    # # print(df.shape)
    # y = df.pop('commit insert bug?')
    # X = df
    # training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size=.1, random_state=12, stratify=y)
    name_project = "commons-math"

    df = pd.read_csv(os.path.join("dataset", name_project, "train.csv"))
    df = df.iloc[:, 1:]

    PMD_FEATURES_AFTER_PRE = [col for col in df.columns if 'PMD' in col]
    for i in ['file_system_sum_WD', 'author_delta_sum_WD', 'system_WD']:
        if i in df.columns:
            PMD_FEATURES_AFTER_PRE.append(i)
    first_inx_pmd = df.columns.get_loc(PMD_FEATURES_AFTER_PRE[0])
    last_inx_pmd = df.columns.get_loc(PMD_FEATURES_AFTER_PRE[-1])
    STATIC_FEATURES_AFTER_PRE = list(df.columns[0:first_inx_pmd])
    JAVADIFF_FEATURES_AFTER_PRE = list(df.columns[last_inx_pmd + 1:-1])
    y_train = df.pop('commit insert bug?')
    X_train = df

    X_train = X_train[PMD_FEATURES_AFTER_PRE + STATIC_FEATURES_AFTER_PRE + JAVADIFF_FEATURES_AFTER_PRE]

    df_test = pd.read_csv(os.path.join("dataset", name_project, "test.csv"))
    df_test = df_test.iloc[:, 1:]
    y_test = df_test.pop('commit insert bug?')
    X_test = df_test
    X_test = X_test[PMD_FEATURES_AFTER_PRE + STATIC_FEATURES_AFTER_PRE+JAVADIFF_FEATURES_AFTER_PRE]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

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
        plt.savefig(r"./results/loss_" + str(ind) + "_" + str(len(model.loss_curve_)) + ".png")

        # # TODO: Save model
        # filename_pkl = r"./results/save_model_test" + str(ind) + ".pkl"
        # pickle.dump(model, open(filename_pkl, 'wb'))
        # loaded_model = pickle.load(open(filename_pkl, 'rb'))
        # score_pkl = eval(loaded_model, model.classes_, testing_X, testing_y)
        # with open(r"./results/bic_scores_score_pkl" + str(ind) + ".json", 'w') as f:
        #     json.dump({**clf.combination, **score_pkl}, f)
        #
        # import joblib
        # filename = r"./results/save_model_test" + str(ind) + ".sav"
        # joblib.dump(model, filename)
        # loaded_model = joblib.load(filename)
        # score_sav = eval(loaded_model, model.classes_, testing_X, testing_y)
        # with open(r"./results/bic_scores_score_sav" + str(ind) + ".json", 'w') as f:
        #     json.dump({**clf.combination, **score_sav}, f)

        # dense_model(loaded_model, "load", testing_X, testing_y)
        # dense_model(model, "real", testing_X, testing_y)


if __name__ == "__main__":
    ind = None
    ind = sys.argv[2]
    import ast
    rf = ast.literal_eval(sys.argv[3])
    main(sys.argv[1], ind, rf)
