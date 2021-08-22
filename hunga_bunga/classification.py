import random
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterSampler

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.utils.testing import all_estimators
from functools import reduce
from .core import *
from .params import *
import itertools as it

# region models
linear_models_n_params = [
    (SGDClassifier,
     {'loss': ['log', 'modified_huber'],
      'alpha': [0.0001, 0.001, 0.1],
      'penalty': penalty_12none
      }),

    (LogisticRegression,
     {'penalty': penalty_12, 'max_iter': max_iter, 'tol': tol, 'warm_start': warm_start, 'C': C, 'solver': ['liblinear']
      }),

    (Perceptron,
     {'penalty': penalty_all, 'alpha': alpha, 'eta0': eta0, 'warm_start': warm_start
      }),

    (PassiveAggressiveClassifier,
     {'C': C, 'warm_start': warm_start,
      'loss': ['hinge', 'squared_hinge'],
      })
]

linear_models_n_params_small = linear_models_n_params

svm_models_n_params = [
    (SVC,
     {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol,
      'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     {'C': C, 'penalty_12': penalty_12, 'tol': tol, 'max_iter': max_iter,
      'loss': ['hinge', 'squared_hinge'],
      })
]

svm_models_n_params_small = [
    (SVC,
     {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol,
      'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     {'C': C, 'penalty': penalty_12, 'tol': tol, 'max_iter': max_iter,
      'loss': ['hinge', 'squared_hinge'],
      })
]

neighbor_models_n_params = [

    (KMeans,
     {'algorithm': ['auto', 'full', 'elkan'],
      'init': ['k-means++', 'random']}),

    (KNeighborsClassifier,
     {'n_neighbors': n_neighbors, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2]
      }),

    (NearestCentroid,
     {'metric': neighbor_metric,
      'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
      }),

    (RadiusNeighborsClassifier,
     {'radius': neighbor_radius, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      'outlier_label': [-1]
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessClassifier,
     {'warm_start': warm_start,
      'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel(), Matern(), StationaryKernelMixin()],
      'max_iter_predict': [20, 50, 100, 200, 500, 1000],
      'n_restarts_optimizer': [0, 1, 2, 3, 4],
      })
]

bayes_models_n_params = [
    (GaussianNB, {})
]

nn_models_n_params_small = [
    (MLPClassifier,
     {'hidden_layer_sizes': [(64,), (32, 64)],
      'batch_size': ['auto', 50],
      'activation': ['identity', 'tanh', 'relu'],
      'max_iter': [500],
      'early_stopping': [True],
      'learning_rate': learning_rate_small
      })
]

tree_models_n_params = [

    (RandomForestClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'min_samples_leaf': min_samples_leaf,
      }),

    (DecisionTreeClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
      'min_impurity_split': min_impurity_split, 'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
      'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'criterion': ['gini', 'entropy']})
]

tree_models_n_params_small = [

    (RandomForestClassifier,
     {'max_features': max_features_small, 'n_estimators': n_estimators_small, 'min_samples_split': min_samples_split,
      'max_depth': max_depth_small, 'min_samples_leaf': min_samples_leaf
      }),

    (DecisionTreeClassifier,
     {'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators': n_estimators_small, 'max_features': max_features_small, 'max_depth': max_depth_small,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf})
]

best = [
    (RadiusNeighborsClassifier,
     {'radius': neighbor_radius, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2, 3, 4, 5],
      'outlier_label': [-1]
      }),
    (GaussianProcessClassifier,
     {'warm_start': warm_start,
      'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel(), Matern(), StationaryKernelMixin()],
      'max_iter_predict': [20, 50, 100, 200, 500, 1000],
      'n_restarts_optimizer': [0, 1, 2, 3, 4],
      })
]

# endregion


# best

# nn_models_n_params = [
#     (MLPClassifier,
#      {'hidden_layer_sizes': [(512, 1024,)],
#       # 'activation': ['relu'],
#       #  'solver': ['adam'],
#       # 'alpha': alpha,  # L2 penalty (regularization term)
#       'alpha': [0.0001],
#       # 'learning_rate': learning_rate, # Only used when solver='sgd',
#       #   'learning_rate_init': [0.001],
#       # 'tol': tol,
#       # 'warm_start': warm_start,
#         'warm_start': [False],
#       # 'batch_size': ['auto', 64, 32, 128],
#     'batch_size': [128],
#
#       'max_iter': [2],
#
#       # 'early_stopping': [True, False],
#       'early_stopping': [False],
#       # 'shuffle': [False, True],
#       'shuffle': [True],
#       'random_state': [1]
#       })
# ]


rf_models_n_params = [

    (RandomForestClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features,  # The number of features to consider when looking for the best split:
      'n_estimators': n_estimators,
      'max_depth': max_depth,
      'min_samples_split': min_samples_split,
      'min_impurity_split': min_impurity_split,  # Threshold for early stopping in tree growth
      'warm_start': warm_start,
      'min_samples_leaf': min_samples_leaf,
      })
]

nn_models_n_params = [
    (MLPClassifier,
     {
         'activation': ['tanh'],
         'alpha': [0.0001, 0.001],
         'batch_size': [128, 64, 32],
         'early_stopping': [False],
         'hidden_layer_sizes': [(64, 128, 512,)],
         'learning_rate_init': [0.001],
         'max_iter': [5000, 3000, 1000, 500],
         'random_state': [12],
         'shuffle': [True],
         'solver': ['adam'],
        'tol': [0.0001, 0.001],

         # 'hidden_layer_sizes': [(128, 512,), (64, 128, 512, 1024,), (128, 128,), (512, 512,), (1024, 512,), (64, 128, 512,),
         #                        (128, 512, 512,), (128, 128, 512,), (512, 1024, 128,), (512, 512, 1024, 128,)],
         # 'activation': ['tanh', 'relu'],
         #
         # # 'solver': ['adam', 'sgd'],
         # 'solver': ['adam'],
         #
         # 'alpha': alpha,  # L2 penalty (regularization term)
         #
         # # 'batch_size': ['auto', 64, 32, 128],
         # 'batch_size': [64],
         #
         # # 'learning_rate': learning_rate, # Only used when solver='sgd'
         #
         # 'learning_rate_init': [0.001, 0.0001], #Only used when solver=’sgd’ or ‘adam’.
         #
         # 'max_iter': [3000],  # adam and sgd = epoch, other - gradient steps
         #
         # # 'shuffle': [False, True], # Only used when solver=’sgd’ or ‘adam’.
         # 'shuffle': [True],
         #
         # 'random_state': [1],
         #
         # 'tol': tol,
         #
         # # 'warm_start': warm_start,
         # # 'warm_start': [False],
         #
         # # 'batch_size': [128],
         #
         #
         # # 'early_stopping': [True, False],
         # 'early_stopping': [False],
         #
         # # 'early_stopping': [False],

     })
]


def run_all_classifiers(x, y, small=False, normalize_x=False, n_jobs=cpu_count() - 1, brain=False, test_size=0.2,
                        n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=False, ind=0, RF=False):
    all_params = (nn_models_n_params if not RF else rf_models_n_params)
    # all_grid_params = dict(reduce(list.__add__, list(map(lambda x: list(x[1].items()), all_params)), []))
    # estimators = all_estimators()
    # estimators = [('MLP', MLPClassifier)]

    varNames = sorted(all_params[0][1])
    combinations = [dict(zip(varNames, prod)) for prod in
                    it.product(*(all_params[0][1][varName] for varName in varNames))]
    clf = all_params[0][0](**combinations[ind])

    return combinations[ind], main_loop(clf, StandardScaler().fit_transform(x) if normalize_x else x, y,
                                        isClassification=True, n_jobs=n_jobs, verbose=verbose, brain=brain,
                                        test_size=test_size, n_splits=n_splits, upsample=upsample, scoring=scoring,
                                        grid_search=grid_search)


def run_one_classifier(x, y, small=True, normalize_x=True, n_jobs=cpu_count() - 1, brain=False, test_size=0.2,
                       n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) + (
        nn_models_n_params_small if small else nn_models_n_params) + (
                     [] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (
                     svm_models_n_params_small if small else svm_models_n_params) + (
                     tree_models_n_params_small if small else tree_models_n_params)
    all_params = random.choice(all_params)
    return all_params[0](**(list(ParameterSampler(all_params[1], n_iter=1))[0]))


class HungaBungaClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size=0.2, n_splits=5, random_state=None, upsample=True, scoring=None,
                 verbose=False, normalize_x=True, n_jobs=cpu_count() - 1, grid_search=True, ind=None):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        self.ind = ind
        self.res = None
        self.combination = None
        super(HungaBungaClassifier, self).__init__()

    def fit(self, x, y, RF=False):
        ans = run_all_classifiers(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits,
                                  upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain,
                                  n_jobs=self.n_jobs, grid_search=self.grid_search, ind=self.ind, RF=RF)
        self.combination, self.model, self.res = ans[0], ans[1][0], ans[1][1]
        return self

    def predict(self, x):
        return self.model.predict(x)


class HungaBungaRandomClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size=0.2, n_splits=5, random_state=None, upsample=True, scoring=None,
                 verbose=False, normalize_x=True, n_jobs=cpu_count() - 1, grid_search=True):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        super(HungaBungaRandomClassifier, self).__init__()

    def fit(self, x, y):
        self.model = run_one_classifier(x, y, normalize_x=self.normalize_x, test_size=self.test_size,
                                        n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring,
                                        verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs,
                                        grid_search=self.grid_search)
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = HungaBungaClassifier()
    clf.fit(X, y)
    print(clf.predict(X).shape)
