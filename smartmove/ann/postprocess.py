import matplotlib.pyplot as plt
import numpy

def preprocess_ann():
    # Generate experiments/isotope pandas from csv
    # Compile datasets
      # Add rho_mod to each sample
    return None

def postprocess_ann():
    import os
    import pandas

    import yamlord

    # TODO update after model path saved to cfg_paths.yaml
    path_data = '/home/ryan/code/python/projects/bodycondition/'

    cfg_paths = yamlord.read_yaml(os.path.join(path_data, 'cfg_paths.yaml'))

    path_data = cfg_paths['root']
    path_csv   = cfg_paths['csv']
    path_ann  = cfg_paths['ann']
    path_model = 'theanets_20170316_132225'

    path_out = os.path.join(path_data, path_ann, path_model)

    cfg_ann = yamlord.read_yaml(os.path.join(path_out, 'cfg_ann.yaml'))

    fname_field = 'field_experiments.p'
    fname_isotope = 'isotope_experiments.p'
    field = pandas.read_pickle(os.path.join(path_data, path_csv, fname_field))
    isotope = pandas.read_pickle(os.path.join(path_data, path_csv, fname_isotope))

    # EXPERIMENT INPUT
    # n experiments and animals
    n_field = len(field)
    n_animals = len(field['animal'].unique())

    # Min max values of rho_mod and % lipid for each seal
    # TODO change `total_dens` to `rho_mod`
    mask_skinny = field['animal']=='skinny'
    mask_notag = field['animal']=='notag'
    min_rhomod_skinny = field[mask_skinny]['total_dens'].min()
    max_rhomod_skinny = field[mask_skinny]['total_dens'].max()
    min_rhomod_notag = field[mask_notag]['total_dens'].min()
    max_rhomod_notag = field[mask_notag]['total_dens'].max()

    # lipid range for each seal

    # ISOTOPE INPUT
    # Min/max body mass of each seal
    mask_skinny = isotope['animal']=='skinny'
    mask_notag = isotope['animal']=='notag'
    min_mass_skinny = isoptope[mask_skinny]['mass_kg'].min()
    max_mass_skinny = isoptope[mask_skinny]['mass_kg'].max()
    min_mass_notag = isoptope[mask_notag]['mass_kg'].min()
    max_mass_notag = isoptope[mask_notag]['mass_kg'].max()


    # ANN CONFIG
    # Number of network configurations
    results_tuning = pandas.read_pickle(os.path.join(path_out, 'results_tuning.p'))
    n_configs = len(results_tuning)

    # Number of samples compiled, train, valid, test
    train = pandas.read_pickle(os.path.join(path_out, 'data_train.p'))
    valid = pandas.read_pickle(os.path.join(path_out, 'data_valid.p'))
    test  = pandas.read_pickle(os.path.join(path_out, 'data_test.p'))
    n_train = len(train[0])
    n_valid = len(valid[0])
    n_test = len(test[0])
    n_all = n_train + n_valid + n_test

    # percentage of compiled dataset in train, valid, test
    perc_train = n_train/n_all
    perc_valid = n_valid/n_all
    perc_test  = n_test/n_all

    # TODO
    # Range of each bin, density, lipid percent
    # avg change in body density/%lipid comp. between bins
    # Total tuning time

    # POSTPROCESS VALUES
    # Get min/max accuracy and training time for all configurations
    hi_acc_idx = results_tuning['accuracy'].argmax()
    lo_acc_idx = results_tuning['accuracy'].argmin()
    hi_acc = results_tuning['accuracy'].max()
    lo_acc = results_tuning['accuracy'].min()

    hi_time_idx = results_tuning['training_time'].argmax()
    lo_time_idx = results_tuning['training_time'].argmin()
    hi_time = results_tuning['training_time'].max()
    lo_time = results_tuning['training_time'].min()

    # Best/worst classification accuracies
    mask_best = results_tuning['accuracy'] == results_tuning['accuracy'].max()
    best_idx = results_tuning['train_time'][mask_acc].idxmin()

    mask_worst = results_tuning['accuracy'] == results_tuning['accuracy'].max()
    worst_idx = results_tuning['train_time'][mask_acc].idxmin()

    best_acc = results_tuning['accuracy'][best_idx]
    best_time = results_tuning['train_time'][best_idx]
    worst_acc = results_tuning['accuracy'][worst_idx]
    worst_time = results_tuning['train_time'][worst_idx]

    # Time for prediction
    import numpy
    import pandas
    # Uset best net
    # TODO loop x times take lowest?
    net = results_tuning['net'][best_idx]
    t = time_prediction(net, test[0][0])

    # Filesize of trained NN
    import os
    st = os.stat(file_best_ann)
    trained_size = st.st_size/1000 #kB

    # Precision of each bin
    tune_cms = pandas.read_pickle('./tune_cms.p')
    precision = calculate_precision(tune_cms['Validation'])

    # %step between subsets of test for dataset size test
    # TODO

    # save as txt/yaml
    return None


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


def target_value_stats(train, valid, test):
    import numpy
    import pandas

    allbins = numpy.hstack([train[1], valid[1], test[1]])
    ubins = numpy.unique(allbins)
    columns = ['bin', 'n', 'perc']

    dfout = pandas.DataFrame(index=range(len(ubins)), columns=columns)

    for i in range(len(ubins)):
        n = len(numpy.where(allbins==ubins[i])[0])
        dfout.ix[i,'bin']  = ubins[i]
        dfout.ix[i,'n']    = n
        dfout.ix[i,'perc'] = n/len(allbins)*100

    for key in ['n', 'perc']:
        dfout[key] = pandas.to_numeric(dfout[key])

    print(dfout)

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
    dfout.ix[:, 'feature'] = features
    for i in range(len(feature_cols)):
        d = df[feature_cols[i]]
        dfout.ix[i,'min']  = d.min()
        dfout.ix[i,'max']  = d.max()
        dfout.ix[i,'mean'] = d.mean()
        dfout.ix[i,'std']  = d.std()

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
