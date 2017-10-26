from bokeh.models import formatters
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper
import pandas as pd
import numpy as np
from itertools import product
from bokeh.plotting import figure, show
from sklearn.model_selection import train_test_split


def plot_scatter_3_features(data, feature1, feature2, feature3, title):
    p = figure(
        title=title,
        x_axis_label=feature1,
        y_axis_label=feature2,
        x_range=(data[feature1].min(), data[feature1].max()),
        y_range=(data[feature2].min(), data[feature2].max()),
        plot_width=800,
        plot_height=500,
    )

    colors = data[feature3].map({0: 'red', 1: 'blue'})

    p.scatter(x=data[feature1],
              y=data[feature2],
              fill_color=colors,
              fill_alpha=1,
              line_color='black',
              size=7)
    p.title.align = "center"
    show(p)


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
    data = pd.read_csv('../../data/loans_workshop.csv')
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
