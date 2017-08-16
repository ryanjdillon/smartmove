from os.path import join as _join

def time_prediction(net, features):
    import timeit
    return timeit.timeit('net.predict(features)', number=10000, globals=locals())


def calculate_precision(cm):
    '''Calculate the precision for each class in a confusion matrix'''
    import numpy

    n_classes = len(cm[0])
    precision = numpy.zeros(n_classes)
    for i in range(n_classes):
        precision[i] = cm[i,i]/sum(cm[i,:])

    return precision


def process(cfg_project, cfg_ann):
    from collections import OrderedDict
    import numpy
    import os
    import pandas
    import pyotelem
    import yamlord

    from . import pre

    paths = cfg_project['paths']
    fnames = cfg_project['fnames']

    path_output = _join(paths['project'], paths['ann'],
                               cfg_project['ann_analyses'][-1])

    fname_field = fnames['csv']['field']
    fname_isotope = fnames['csv']['isotope']

    file_field = _join(paths['project'], paths['csv'],
                              fnames['csv']['field'])
    file_isotope = _join(paths['project'], paths['csv'],
                                fnames['csv']['isotope'])
    field, isotope = pre.add_rhomod(file_field, file_isotope)

    # EXPERIMENT INPUT
    post = OrderedDict()
    post['input_exp'] = OrderedDict()

    # n experiments and animals
    post['n_field'] = len(field)
    post['n_animals'] = len(field['animal'].unique())

    # Min max values of rho_mod and % lipid for each seal
    post['exp'] = OrderedDict()
    post['iso'] = OrderedDict()
    for a in numpy.unique(field['animal']):
        # Field experiment values
        post['exp'][a] = OrderedDict()
        mask = field['animal'] == a
        post['exp'][a]['min_rhomod'] = field[mask]['rho_mod'].min()
        post['exp'][a]['max_rhomod'] = field[mask]['rho_mod'].max()

        # lipid range for each seal

        # Isotope experiment values
        post['iso'][a] = OrderedDict()
        mask = isotope['animal'] == a
        min_mass = isotope[mask]['mass_kg'].min()
        max_mass = isotope[mask]['mass_kg'].max()

    # ANN CONFIG
    results = pandas.read_pickle(_join(path_output, fnames['ann']['tune']))

    post['ann'] = OrderedDict()

    # Number of network configurations
    post['ann']['n_configs'] = len(results)

    # Number of samples compiled, train, valid, test
    post['ann']['n'] = OrderedDict()

    file_train = _join(path_output, 'data_train.p')
    file_valid = _join(path_output, 'data_valid.p')
    file_test = _join(path_output, 'data_test.p')
    train = pandas.read_pickle(file_train)
    valid = pandas.read_pickle(file_valid)
    test = pandas.read_pickle(file_test)

    for set_name in ['train', 'valid', 'test']:
        post['ann']['n'][set_name] = len(data[0])

    n_all = sum([n for n in post['ann']['n'].values()])
    post['ann']['n']['all'] = n_all

    # percentage of compiled dataset in train, valid, test
    for set_name in ['train', 'valid', 'test']:
        n_set = post['ann']['n'][set_name]
        post['ann']['n']['perc_{}'.format(set_name)] = n_set/n_all

    # Total tuning time
    post['ann']['total_train_time'] = results['train_time'].sum()

    # POSTPROCESS VALUES
    # Best/worst classification accuracies
    mask_best = results['accuracy'] == results['accuracy'].max()
    best_idx = results['train_time'][mask_best].idxmin()

    mask_worst = results['accuracy'] == results['accuracy'].min()
    worst_idx = results['train_time'][mask_worst].idxmax()

    post['ann']['best_idx'] = best_idx
    post['ann']['worst_idx'] = worst_idx

    # Get min/max accuracy and training time for all configurations
    post['ann']['metrics'] = OrderedDict()
    for key in ['accuracy', 'train_time']:
        post['ann']['metrics'][key] = OrderedDict()
        post['ann']['metrics'][key]['max_idx'] = results[key].argmax()
        post['ann']['metrics'][key]['min_idx'] = results[key].argmin()

        post['ann']['metrics'][key]['max'] = results[key].max()
        post['ann']['metrics'][key]['min'] = results[key].min()

        post['ann']['metrics'][key]['best'] = results[key][best_idx]
        post['ann']['metrics'][key]['worst'] = results[key][worst_idx]

    # Optimal network results
    post['ann']['opt'] = OrderedDict()

    net = results['net'][best_idx]

    # Loop 10 times taking mean prediction time
    # Each loop, 100k iterations of timing
    file_test = _join(path_output, fnames['ann']['test'])
    test = pandas.read_pickle(file_test)
    t_pred = numpy.mean([time_prediction(net, test[0][:0]) for _ in range(10)])
    post['ann']['opt']['t_pred'] = t_pred

    # Filesize of trained NN
    file_net_best = './net.tmp'
    pandas.to_pickle(net, file_net_best)
    st = os.stat(file_net_best)
    os.remove(file_net_best)
    post['ann']['opt']['trained_size'] = st.st_size/1000 #kB

    # %step between subsets of test for dataset size test
    post['ann']['dataset'] = 'numpy.arange(0,1,0.03))[1:]'

    # Tune confusion matrices (cms) from most optimal configuration
    # one field per dataset `train`, `valid`, and `test`
    # first level `targets` if for all datasets
    post['ann']['bins'] = OrderedDict()
    file_tune_cms = _join(path_output, fnames['ann']['cms_tune'])

    tune_cms = pandas.read_pickle(file_tune_cms)
    bins = tune_cms['targets']

    # Range of each bin, density, lipid percent
    bin_range = range(len(bins)-1)

    rho_hi = numpy.array([bins[i] for i in bin_range])
    rho_lo = numpy.array([bins[i+1] for i in bin_range])
    # Note density is converted from kg/m^3 to g/cm^3 for `dens2lip`
    lip_hi = pyotelem.physio_seal.dens2lip(rho_hi*0.001)['perc_lipid'].values
    lip_lo = pyotelem.physio_seal.dens2lip(rho_lo*0.001)['perc_lipid'].values

    # Generate bin ranges as strings
    fmt_bin = r'{:7.2f} <= rho_mod < {:7.2f}'
    fmt_lip = r'{:6.2f} <= lipid % < {:6.2f}'
    str_bin = [fmt_bin.format(lo, hi) for lo, hi in zip(rho_lo, rho_hi)]
    str_lip = [fmt_lip.format(lo, hi) for lo, hi in zip(lip_lo, lip_hi)]

    path_sgls = _join(path_output, fnames['ann']['sgls'])
    sgls_ann = pandas.read_pickle(path_sgls)

    post['ann']['bins']['values'] = list(bins)
    post['ann']['bins']['value_range'] = str_bin
    post['ann']['bins']['value_diff'] = list(numpy.diff(bins))

    # Note density is converted from kg/m^3 to g/cm^3 for `dens2lip`
    lipid_perc = pyotelem.physio_seal.dens2lip(bins*0.001)['perc_lipid'].values
    post['ann']['bins']['lipid_perc'] = list(lipid_perc)
    post['ann']['bins']['lipid_range'] = str_lip
    post['ann']['bins']['lipid_diff'] = list(numpy.diff(lipid_perc))

    precision = calculate_precision(tune_cms['validation']['cm'])
    post['ann']['bins']['precision'] = [None,] * len(bins)
    targets = tune_cms['validation']['targets']
    for i in range(len(bins)):
        if bins[i] in targets:
            post['ann']['bins']['precision'][i] = precision[bins[i]==targets]
        else:
            post['ann']['bins']['precision'][i] = 'None'

    # Save post processing results as YAML
    file_post = _join(path_output, fnames['ann']['post'])
    yamlord.write_yaml(post, file_post)

    # Create dataframes with summary stats of input and output features
    feature_cols = ['abs_depth_change',
                    'dive_phase_int',
                    'mean_a',
                    'mean_depth',
                    'mean_pitch',
                    'mean_speed',
                    'mean_swdensity',
                    'total_depth_change',
                    'total_speed_change',]
    input_stats = input_feature_stats(sgls_ann, feature_cols)
    target_stats = target_value_stats(train, valid, test)
    input_stats.to_pickle(_join(path_output, fnames['ann']['stats_features']))
    target_stats.to_pickle(_join(path_output, fnames['ann']['stats_targets']))

    return post


def target_value_stats(train, valid, test):
    import numpy
    import pandas

    allbins = numpy.hstack([train[1], valid[1], test[1]])
    ubins = numpy.unique(allbins)
    columns = ['bin', 'n', 'perc']

    dfout = pandas.DataFrame(index=range(len(ubins)), columns=columns)

    for i in range(len(ubins)):
        n = len(numpy.where(allbins==ubins[i])[0])
        dfout['bin'][i] = ubins[i]
        dfout['n'][i] = n
        dfout['perc'][i] = n/len(allbins)*100

    for key in ['n', 'perc']:
        dfout[key] = pandas.to_numeric(dfout[key])

    return dfout


def input_feature_stats(df, feature_cols):
    '''Produce stats for paper tables

    Args
    ----
    df: pandas.dataframe
        Final `sgls_all.p` dataframe filtered and saved to model output dir
    feature_cols: ndarray
        List of string names of feature columns

    Returns
    -------
    df_out: pandas_dataframe
        Dataframe containing statistics for input features used in ANN
    '''
    import pandas

    features = [r'Absolute depth change (m)',
                r'Dive phase',
                r'Mean acceleration (g)',
                r'Mean depth (m)',
                r'Mean pitch (\degree)',
                r'Mean speed (m s\textsuperscript{-1})',
                r'Mean seawater density (kg m\textsuperscript{3})',
                r'Total depth change (m)',
                r'Total speed change (m s\textsuperscript{-1})']

    columns = ['feature', 'min', 'max', 'mean', 'std']
    dfout = pandas.DataFrame(index=range(len(feature_cols)), columns=columns)
    dfout['feature'][:] = features
    for i in range(len(feature_cols)):
        d = df[feature_cols[i]]
        dfout['min'][i] = d.min()
        dfout['max'][i] = d.max()
        dfout['mean'][i] = d.mean()
        dfout['std'][i] = d.std()

    for key in ['min', 'max', 'mean', 'std']:
        dfout[key] = pandas.to_numeric(dfout[key])

    print('Input feature statistics')
    print(dfout)

    return dfout


def last_monitors(results_dataset):
    '''Extract the last value from each monitor in results_dataset

    Args
    ----
    results_dataset: OrderedDict
        Contains results from each trained ANN with varying fraction of train

    Returns
    -------
    m: dict
        Dict of `train` and `valid` dicts with `err`, `loss`, and `acc`
    '''
    import numpy


    m = dict()
    m['n_train'] = results_dataset['n_train']

    set_keys = ['train', 'valid']
    monitor_keys = ['err', 'loss', 'acc']

    # Process for each dataset in `set_keys`
    for skey in set_keys:
        n_networks = len(results_dataset['monitors'])
        m[skey] = dict()

        # Create output array for each monitor in `monitor_keys`
        for mkey in monitor_keys:
            m[skey][mkey]  = numpy.zeros(n_networks, dtype=float)

        # Iterate through results from network trainings
        for i in range(n_networks):
            # Pointer to monitor dicts for `i`th network training
            monitors = results_dataset['monitors'][i][skey]
            # Get last value from list of monitors from training
            for mkey in monitor_keys:
                m[skey][mkey][i]  = monitors[mkey][-1]

    return m


def plot_learning_curves(m):
    '''Plot learning curve of train and test accuracy/error

    Args
    ----
    net: theanets.feedforward.Classifier
        trained Theanets classifier network
    m: dict()
        Dict of `train` and `valid` dicts with `err`, `loss`, and `acc`
    mkey: str
        key for monitor to plot
    '''
    import matplotlib.pyplot as plt
    import numpy
    import scipy.optimize
    import seaborn

    seaborn.set_style('whitegrid')

    linear = lambda x, m, c: m*x + c

    fig, ax = plt.subplots()

    ax.plot(m['train']['err'], label='Train')
    ax.plot(m['valid']['err'], label='Validation')
    ax.set_ylabel('Cross-entropy error')
    ax.set_xlabel('Number of samples')

    # Fit a linear regression througn n_train to get ticks at regular locs
    x = m['n_train'].values.astype(float)
    y = numpy.arange(len(m['n_train']))
    popt, pcov = scipy.optimize.curve_fit(linear, x, y)
    new_labels = numpy.arange(0, 6500, 1000)
    new_ticks = linear(new_labels, *popt)

    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels)

    ax.legend()
    plt.tight_layout()
    plt.savefig('learning_curve_datasetsize.svg')
    plt.show()

    return None


def plot_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    Args
    ----
    ax : matplotlib.axes.AxesSubplot
      The axes on which to plot the cartoon (get e.g. by plt.gca())
    left : float
      The center of the leftmost node(s) will be placed here
    right : float
      The center of the rightmost node(s) will be placed here
    bottom : float
      The center of the bottommost node(s) will be placed here
    top : float
      The center of the topmost node(s) will be placed here
    layer_sizes : list of int
      List of layer sizes, including input and output dimensionality

    Example
    -------
    ```
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    plot_neural_net(ax, .1, .9, .1, .9, [9, 15, 10])
    fig.savefig('neuralnet.svg')
    plt.show()
    ```

    Author: Colin Raffel
    Gist link: https://gist.github.com/craffel/2d727968c3aaebd10359
    '''
    import matplotlib.pyplot as plt

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing),
                                v_spacing/4., color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    enum_layer_sizes = enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
    for n, (layer_size_a, layer_size_b) in enum_layer_sizes:
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                  c='k')
                ax.add_artist(line)

    return None
