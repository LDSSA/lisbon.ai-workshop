from bokeh.models import formatters
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper
import pandas as pd
import numpy as np
from itertools import product
from bokeh.plotting import figure, show
from sklearn.model_selection import train_test_split


def plot_data(model, data, target, feature1, feature2, scatter=True,
              area=True, out_of_sample=False, probabilities=True, n_points=500):
    p = figure(
        title='%s predictions ' % model.__class__.__name__,
        x_axis_label=feature1,
        y_axis_label=feature2,
        x_range=(data[feature1].min(), data[feature1].max()),
        y_range=(data[feature2].min(), data[feature2].max()),
        plot_width=800,
        plot_height=500,
    )

    color_mapper = LinearColorMapper(palette='RdYlGn11', low=0, high=1)

    if out_of_sample:
        train, test, = train_test_split(data, test_size=0.5, random_state=1000)

    else:
        train = data
        test = data

    if area:
        get_area(p, model, train, target, feature1, feature2, color_mapper,
                 probabilities, n_points)

    if scatter:

        if area:
            black_and_white = True
        else:
            black_and_white = False

        get_scatterplot(p, test, target, feature1, feature2, black_and_white)
        # get the color bar
        cbar = ColorBar(color_mapper=color_mapper, location=(0, 0))
        p.add_layout(cbar, 'right')
        p.title.align = "center"

    # styling
    p.xaxis.formatter = formatters.NumeralTickFormatter(format="0.0")
    p.yaxis.formatter = formatters.NumeralTickFormatter(format="0.0")
    p.toolbar.logo = None
    p.toolbar_location = None

    show(p)


def get_area(p, model, train, target, feature1, feature2, color_mapper,
             probabilities=True, n_points=500):
    # fit the model to the target
    model.fit(train[[feature1, feature2]], train[target])

    # create a dataset for predicting
    x = np.linspace(train[feature1].min(), train[feature1].max(), n_points)
    y = np.linspace(train[feature2].min(), train[feature2].max(), n_points)

    # here is the dataset
    area_space = pd.DataFrame(list(product(x, y)),
                              columns=[feature1, feature2])

    # Now predict it
    if probabilities:
        area_space['predictions'] = model.predict_proba(
            area_space[[feature1, feature2]])[:, 1]
    else:
        area_space['predictions'] = model.predict(
            area_space[[feature1, feature2]])

    array_data = area_space.set_index([feature1, feature2]).unstack().values

    p.image(
        image=[array_data.T],
        x=x.min(),
        y=y.min(),
        dw=(x.max() - x.min()),
        dh=(y.max() - y.min()),

        color_mapper=color_mapper,
        alpha=.5)


def get_scatterplot(p, data, target, feature1, feature2, black_and_white):
    # Scatter plot
    if black_and_white:
        colors = data[target].map({0: 'white', 1: 'black'})
    else:
        colors = data[target].map({0: 'green', 1: 'red'})
    p.scatter(x=data[feature1],
              y=data[feature2],
              fill_color=colors,
              fill_alpha=1,
              line_color='black',
              size=7)


def get_toy_data(n_points=10):
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
