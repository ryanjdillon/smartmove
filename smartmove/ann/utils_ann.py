def ppickle(obj, file_path):
    '''Write routine for saving output'''
    import pandas

    with open(file_path, 'wb') as f:
        pandas.to_pickle(obj, file_path)

    return None


def plot_confusion_matrix(cm, targets, normalize=False, title='', cmap=None,
        xlabel_rotation=0):
    '''This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    Plotting routind modified from this code: https://goo.gl/kYHMxk
    '''
    import itertools
    import matplotlib.pyplot as plt
    import numpy

    if cmap is None:
        cmap = plt.cm.PuBuGn

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print('\n{}, normalized'.format(title))
    else:
        print('\n{}, without normalization'.format(title))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Number of predictions')
    n_targets = len(targets)
    yticks = numpy.arange(n_targets)
    xticks = yticks #- (numpy.diff(yticks)[0]/3)
    plt.xticks(xticks, numpy.round(targets, 1), rotation=xlabel_rotation)
    plt.yticks(yticks, numpy.round(targets,1))

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Observed bin')
    plt.xlabel('Predicted bin')
    plt.tight_layout()
    plt.show()

    return None


def get_confusion_matrices(net, train, valid, targets):
    '''Print and return an sklearn confusion matrix from the input net'''
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

        title = '{} confusion matrix'.format(label.capitalize())
        plot_confusion_matrix(cms[label]['cm'], cms[label]['targets'], title=title)

    return cms


def n_hidden(n_input, n_output, n_train_samples, alpha):
    # http://stats.stackexchange.com/a/136542/16938
    # Alpha is scaling factor between 2-10
    n_hidden = n_samples/(alpha*(n_input+n_output))
    return n_hidden
