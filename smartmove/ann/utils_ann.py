'''
This module contains utility functions for use in the ANN modules
'''

def ppickle(obj, file_path):
    '''Write routine for saving output

    Args
    ----
    obj: object
        Python object to be pickled
    file_path: str
        Path and filename of pickle file
    '''
    import pandas

    with open(file_path, 'wb') as f:
        pandas.to_pickle(obj, file_path)

    return None


def plot_confusion_matrix(cm, tick_labels, xlabel='Predicted',
        ylabel='Observed', normalize=False, title='', cmap=None,
        xlabel_rotation=0, path_plot=None):
    '''This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    Args
    ----
    cm: ndarray
        Confusion matrix as a numpy ndarray
    tick_labels: list of str
        Tick labels for both y-axis and x-axis
    xlabel: str
        Axis label for x-axis (Default: 'Predicted')
    ylabel: str
        Axis label for y-axis (Default: 'Observed')
    normalize: bool
        Switch for normalizing the count data (i.e. from 0 to 1, Default: False)
    title: str
        Main title above plot (Default: '')
    cmap: matplotlib.cmap
        Colormap object for color coding the number of correct predictions
    xlabel_rotation: int
        Degrees to rotate the x-axis tick labels (Default: 0)
    path_plot: str
        Path and filename for plot to be saved. If 'None', no plot will be
        saved (Defualt: None)

    Note
    ----
    Plotting routind modified from this code: https://goo.gl/kYHMxk
    '''
    import itertools
    import matplotlib.pyplot as plt
    import numpy
    import os
    from pyotelem.plots import plotutils

    from ..visuals import latex
    from ..visuals import utils

    # Set colomap if not passed
    if cmap is None:
        cmap = plt.cm.PuBuGn

    # Create normalized
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    # Creat matrix plot from confusion matric
    fig, ax = plt.subplots()
    cm_mag = plotutils.magnitude(cm.max())
    cm_max = plotutils.roundup(int(cm.max()), cm_mag)
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0,
                    vmax=cm_max)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Turn off grid (overide `seaborn`)
    ax.grid(False)

    # Create color bar and update ticks
    cb = fig.colorbar(img, label='Number predicted')
    cb_labels = cb.ax.get_yticklabels()
    cb_labels = numpy.linspace(0, cm_max, len(cb_labels)).astype(int)
    cb.ax.set_yticklabels(cb_labels)

    # Configure matrix ticks
    tick_marks = numpy.arange(len(tick_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_labels, rotation=xlabel_rotation)
    ax.set_yticklabels(tick_labels)

    # Draw annotations with varying colors given `thresh`
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    # Save plot
    if path_plot is not None:
        fname = 'confusion-matrix'
        ext = 'eps'
        if normalize:
            fname += '_normalized'

        file_path = os.path.join(path_plot, '{}.{}'.format(fname, ext))
        plt.savefig(file_path, format=ext, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return None


def get_confusion_matrices(net, train, valid, targets, plots=True):
    '''Print and return an sklearn confusion matrix from the input net

    Args
    ----
    net: theanets.Classifier
        Theanets feedfoward classifier object (i.e. the ANN object)
    train: tuple of ndarrays
        Train data set tuple of the input features (train[0]), and their target
        values (train[1]).
    valid: tuple of ndarrays
        Validation data set tuple of the input features (valid[0]), and their
        target values (valid[1]).
    test: tuple of ndarrays
        Test data set tuple of the input features (test[0]), and their target
        values (test[1]).

    Returns
    -------
    cms: OrderedDict
        Dictionary of confusion matrices for the 'training' and 'validation'
        data sets.
    '''
    from collections import OrderedDict
    import numpy
    import sklearn.metrics

    # Show confusion matrices on the training/validation splits.
    cms = OrderedDict()
    cms['targets'] = targets
    for label, (X, y) in (('training', train), ('validation', valid)):

        cms[label] = OrderedDict()
        # Filter targets to only those that were assigned to training values
        cms[label]['targets'] = targets[sorted(list(numpy.unique(y)))]
        # Create confusion matrix as numpy array
        cms[label]['cm'] = sklearn.metrics.confusion_matrix(y, net.predict(X))

        if plots:
            title = '{} confusion matrix'.format(label.capitalize())
            plot_confusion_matrix(cms[label]['cm'], cms[label]['targets'], title=title)

    return cms
