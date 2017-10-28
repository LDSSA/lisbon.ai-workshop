from bokeh.models import formatters
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper
import pandas as pd
import numpy as np
from itertools import product
from bokeh.plotting import figure, show
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# TODO: add crosses with the train set
# TODO: clean up API, making sure not to break the notebooks
# TODO: add the accuracy score to the chart
# TODO: add more datasets for the smarty pants out there

def plot_data(data, target, feature1, feature2, model=None, scatter=True,
              out_of_sample=False, probabilities=True, n_points=500):
    # TODO: call these things x and y, x.name and y.name, much less error prone
    # TODO: make the image smaller for some reason it seems to be cutting in jupyter

    if isinstance(model, BaseEstimator):
        title = '%s predictions ' % model.__class__.__name__
    else:
        title = 'Scatter plot'

    x_min = data[feature1].min()
    x_max = data[feature1].max()
    y_min = data[feature2].min()
    y_max = data[feature2].max()

    lims = [x_min, x_max, y_min, y_max]

    p = figure(
        title=title,
        x_axis_label=feature1,
        y_axis_label=feature2,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        plot_width=600,
        plot_height=500,
    )

    color_mapper = LinearColorMapper(palette='RdYlGn11', low=0, high=1)

    if out_of_sample:
        train, test, = train_test_split(data, test_size=0.5, random_state=1000)

    else:
        train = data
        test = data

    if isinstance(model, BaseEstimator):
        _get_area(p, model, train, target, feature1, feature2,
                  color_mapper, lims, probabilities, n_points)

    if scatter:
        if isinstance(model, BaseEstimator):
            black_and_white = True
        else:
            black_and_white = False

        _get_scatterplot(p, test, target, feature1, feature2, black_and_white)
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


def _get_area(p, model, train, target, feature1, feature2, color_mapper, lims,
              probabilities=True, n_points=500, ):
    # fit the model to the target
    model.fit(train[[feature1, feature2]], train[target])

    # create a dataset for predicting
    x = np.linspace(lims[0], lims[1], n_points)
    y = np.linspace(lims[2], lims[3], n_points)

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

    # TODO: understand why the transpose. I suspect the problem might
    # be the axis names
    array_data = area_space.set_index([feature1, feature2]).unstack().T.values

    p.image(
        image=[array_data],
        x=x.min(),
        y=y.min(),
        dw=(x.max() - x.min()),
        dh=(y.max() - y.min()),
        color_mapper=color_mapper,
        alpha=.5)


def _get_scatterplot(p, data, target, feature1, feature2, black_and_white):
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


def plot_hyper_parameters(gs):
    if len(list(gs.param_grid.keys())) != 2:
        raise ValueError('Two parameters at the time please')
    temp = pd.DataFrame(gs.grid_scores_)
    keys = list(gs.param_grid.keys())

    unstacked = pd.concat([temp['parameters'].apply(pd.Series),
                           temp['mean_validation_score']],
                          axis=1).set_index(
        keys)['mean_validation_score'].unstack()

    plt.title('Accuracy at different hyper parameters')
    sns.heatmap(unstacked, annot=True)
    plt.show()


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

    _get_scatterplot(p, data, feature3, feature1, feature2,
                     black_and_white=False)

    show(p)


def calculate_quartet_stats(df_list):
    """
    Why in the name of Satan would anyone code this way?

    :param df_list:
    :return:
    """
    stats = {}
    stats['corr x and y'] = {}
    stats['mean x'] = {}
    stats['mean y'] = {}
    stats['std x'] = {}
    stats['std y'] = {}
    stats['r2'] = {}
    stats['max x'] = {}
    stats['max y'] = {}
    stats['min x'] = {}
    stats['min y'] = {}

    for df in df_list:
        stats['corr x and y'][df.name] = df['x'].corr(df['y'])
        stats['mean x'][df.name] = df['x'].mean()
        stats['mean y'][df.name] = df['y'].mean()
        stats['std x'][df.name] = df['x'].std()
        stats['std y'][df.name] = df['y'].std()
        stats['r2'][df.name] = r2_score(df['x'], df['y'])
        stats['max x'][df.name] = df['x'].max()
        stats['max y'][df.name] = df['y'].max()
        stats['min x'][df.name] = df['x'].min()
        stats['min y'][df.name] = df['y'].min()

    return pd.DataFrame(stats)


def plot_scatter_and_linreg(df, col='b'):
    """
    This is hardcoded for our example in the tutorial
    If you want to generalize, abstract away the feature names

    :param df: the data from the quartet
    :param col: the color to use in the chart
    :return:
    """
    lr = LinearRegression()
    lr.fit(df['x'].reshape(-1, 1), df['y'])
    df.plot(kind='scatter', x='x', y='y', c=col, s=50)
    x_pred = np.linspace(df['x'].min(), df['x'].max(), 10)
    y_pred = lr.predict(x_pred.reshape(-1, 1))
    plt.plot(x_pred, y_pred, ls=':', c=col)

    plt.title(df.name)
