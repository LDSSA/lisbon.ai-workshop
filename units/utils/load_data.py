import pandas as pd
import numpy as np
from itertools import product
from sklearn.datasets import make_moons, make_circles, make_classification


def _get_scikit_datasets(n_points=100):
    X, y = make_classification(n_samples=n_points, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    datasets = [make_moons(n_samples=n_points, noise=0.3, random_state=0),
                make_circles(n_samples=n_points, noise=0.2, factor=0.5,
                             random_state=1),
                linearly_separable
                ]
    return datasets


def get_circle(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    i = 1
    q = pd.DataFrame(datasets[i][0])
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_ying_yang(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    # print(datasets[0])
    i = 0
    q = pd.DataFrame(datasets[i][0])
    # print(q.shape)
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_linear_separable(n_points=100):
    datasets = _get_scikit_datasets(n_points)
    i = 2
    q = pd.DataFrame(datasets[i][0])
    q['c'] = datasets[i][1]
    q.columns = ['a', 'b', 'c']
    return q


def get_cross_data(n_points=10):
    x = np.linspace(1, n_points, n_points)
    y = np.linspace(1, n_points * 100, n_points)
    q = pd.DataFrame(list(product(x, y)), columns=['a', 'b'])
    q['c'] = 0
    q = q.fillna(0)
    q.loc[((q['b'] < q['b'].median()) & (q['a'] < q['a'].median())), 'c'] = 1
    q.loc[((q['b'] > q['b'].median()) & (q['a'] > q['a'].median())), 'c'] = 1
    return q


def get_loan_data(n_points=1000):
    data = pd.read_csv('../data/loans_workshop.csv')
    features = [f for f in data.columns if f != 'defaulted']
    target = 'defaulted'
    data = data.sample(n_points)
    return data, features, target


def get_correlated_data(n_points=100):
    xx = np.array([-0.51, 51.2])
    yy = np.array([0.33, 51.6])
    means = [xx.mean(), yy.mean()]
    stds = [xx.std() / 3, yy.std() / 3]
    corr = 0.9  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
            [stds[0] * stds[1] * corr, stds[1] ** 2]]

    m = np.random.multivariate_normal(means, covs, n_points).T

    data = pd.DataFrame(m).T

    data.columns = ['a', 'b']
    return data


def get_temperature_data():
    data = pd.DataFrame({
        'Summer': [15, 25, 39],
        'Winter': [-2, 5, 16],
        'Autumn': [7, 14, 25],
        'Spring': [6, 20, 29]
    }).T

    data.columns = ['Min', 'Mean', 'Max']
    return data
