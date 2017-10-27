from bokeh.models import formatters
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper
import pandas as pd
import numpy as np
from itertools import product
from bokeh.plotting import figure, show
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import  BaseEstimator

# TODO: add crosses with the train set
# TODO: clean up API, making sure not to break the notebooks
# TODO: add the accuracy score to the chart
# TODO: predict the area of the whole thing not just training set area
# TODO: pass the data as a multi-index Series, to keep consistency
# TODO: add more datasets for the smarty pants out there
# TODO: add graphviz

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
