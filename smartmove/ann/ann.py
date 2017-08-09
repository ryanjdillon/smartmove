def get_attr(results, group, attr):
    import numpy

    n_results = len(results)
    attrs = [results['monitors'][i][group][attr][-1] for i in range(n_results)]

    return numpy.asarray(attrs)


def _normalize_data(df, features, target, n_targets):
    '''Normalize features and target values

    Args
    ----
    df: pandas.DataFrame
        dataframe with input features columns and target value columns
    features: str
        list of string names of columns to be used as features
    target: str
        string name of column to be used as target vales
    n_targets: int
        Number of target targets (bins) to split `target` into

    Returns
    -------
    df: pandas.DataFrame
        Data frame containing normalized features and target columns
    bins: ndarray
        List of unique targets (bins) generated during data splitting

    Note
    ----
    This should be performed before splitting the dataset into train/validation
    sets.

    Could also consider using sklearn for normalization:

    from sklearn.preprocessing import normalize
    data_norm = normalize(data_array, norm='l2', axis=1).astype('f4')
    '''

    def _mean_normalize(df, keys):
        return df[keys].div(df.sum(axis=0)[keys], axis=1)[keys]

    def _unit_normalize(df, keys):
        data = df[keys].values
        #(df[keys] - df[keys].min())/(df[keys].max()-df[keys].min())
        return (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))

    def _bin_column(y, n_targets):
        import numpy

        ymin =  numpy.floor(numpy.min(y),) - 1
        ymax =  numpy.ceil(numpy.max(y)) + 1

        # Make one additional bin for use in digitize()
        bins = numpy.linspace(ymin, ymax, n_targets+1)

        # No values are assigned to last bin, making bin widths even across
        # values. Digitize would otherwise assign only y.max() to last bin.
        # Theno messes up when digits don't correspond with python indexing
        ybins = numpy.digitize(y, bins)-1

        # Remove buffer bin before returning
        bins = bins[:n_targets]

        return ybins, bins

    # Normalize inputs
    df[features] = _unit_normalize(df, features)

    # Bin outputs
    df['y_binned'], bins = _bin_column(df[target], n_targets)

    return df, bins


def _split_indices(df, valid_frac):
    '''Get randomly selected row indices for train, validation, and test data
    Args
    ----
    df: pandas.DataFrame

    features: list
        List of string column names in `df` to be used as feature values
    target: str
        Name of the column in `df` to use as the target value
    valid_frac: float
        Fraction of dataset that should be reserved for validation/testing.
        This slice of dataframe `df` is then split in half into the validation
        and testing datasets

    Returns
    -------
    ind_train: pandas.index
        Index of values to be sliced for training
    ind_valid: pandas.index
        Index of values to be sliced for validation
    ind_test: pandas.index
        Index of values to be sliced for testing
    '''
    # Sample into train and validation sets
    ind_train = df.sample(frac=valid_frac).index
    mask_rest = df.index.isin(ind_train)
    df_rest = df.loc[~mask_rest]

    # Split valid to valid & test sets
    # http://stats.stackexchange.com/a/19051/16938
    ind_valid = df_rest.sample(frac=0.5).index
    ind_test  = df_rest[~df_rest.index.isin(ind_valid)].index

    return ind_train, ind_valid, ind_test


def _create_datasets(df, ind_train, ind_valid, ind_test, features, target):
    '''

    Args
    ----
    df: pandas.DataFrame


    Returns
    -------
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    '''

    df_train = df.loc[ind_train]
    df_valid = df.loc[ind_valid]
    df_test  = df.loc[ind_test]

    # Extract to numpy arrays - typecast to float32
    X_train  = (df_train[features].values).astype('f4')
    train_labels = (df_train['y_binned'].values).astype('f4')

    X_valid  = (df_test[features].values).astype('f4')
    valid_labels = (df_test['y_binned'].values).astype('f4')

    X_test  = (df_test[features].values).astype('f4')
    test_labels = (df_test['y_binned'].values).astype('f4')

    # Make into tuple (features, label)
    # Pivot 1-D target value arrays to match 0dim of X
    train = X_train, train_labels.astype('i4')
    valid = X_valid, valid_labels.astype('i4')
    test  = X_test, test_labels.astype('i4')

    return train, valid, test


def _get_configs(tune_params):
    '''Generate list of all possible configuration dicts from tuning params'''
    import itertools

    #tune_idxs = list()
    #for key in tune_params.keys():
    #    tune_idxs.append(range(len(tune_params[key])))

    # Create list of configuration permutations
    config_list = list(itertools.product(*tune_params.values()))

    configs = list()
    for l in config_list:
        configs.append(dict(zip(tune_params.keys(), l)))

    return configs


def create_algorithm(train, valid, config, n_features, n_targets, plots=False):
    '''Configure and train a theanets neural network

    Args
    ----
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    config: dict
        Dictionary of network configuration parameters
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    net: theanets.Classifier object
        Neural network object
    accuracy: float
        Accuracy value of the network configuration from the validation dataset
    monitors: dict
        Dictionary of "monitor" objects produced during network training
        Contains two labels 'train' and 'valid' with the following attributes:
            - 'loss': percentage from loss function (default: cross-entropy)
            - 'err': percentage of error (default: )
            - 'acc': percentage of accuracy (defualt: true classifications)
    '''
    from collections import OrderedDict
    import theanets

    def plot_monitors(attrs, monitors_train, monitors_valid):
        import matplotlib.pyplot as plt
        import seaborn

        seaborn.set_style('whitegrid')

        labels = {'loss':'Loss', 'err':'Error', 'acc':'Accuracy'}

        fig, axes = plt.subplots(1, len(attrs), sharex=True)
        legend_on = True
        for ax, attr in zip(axes, attrs):
            ax.yaxis.label.set_text(labels[attr])
            ax.xaxis.label.set_text('Epic')
            ax.plot(monitors['train'][attr], label='Training')
            ax.plot(monitors['valid'][attr], label='Validation')
            if legend_on:
                ax.legend(loc='upper right')
                legend_on = False
        plt.show()

        # Keep seaborn from messing up confusion matrix plots
        seaborn.reset_orig()

        return None

    # SGD converges to minima/maxima faster with momentum
    # NAG, ADADELTA, RMSProp have equivalents of parameter specific momentum
    if config['algorithm'] is 'sgd':
        config['momentum'] = 0.9

    # Create dictionary for storing monitor lists
    attrs = ['loss', 'err', 'acc']
    monitors = OrderedDict()
    for mtype in ['train', 'valid']:
        monitors[mtype] = dict()
        for attr in attrs:
            monitors[mtype][attr] = list()

    print('')
    print('Train samples:       {:8d}'.format(len(train[0])))
    print('Valididation samples:{:8d}'.format(len(valid[0])))
    print('Hidden layers:       {:8d}'.format(config['hidden_layers']))
    print('Hidden nodes/layer:  {:8d}'.format(config['hidden_nodes']))
    print('')

    kwargs = {'train':train,
              'valid':valid,
              'algo':config['algorithm'],
              #'patience':config['patience'],
              #'min_improvement':config['min_improvement'],
              #'validate_every':config['validate_every'],
              #'batch_size':batch_size,
              #'train_batches':train_batches,
              #'valid_batches':valid_batches,
              'learning_rate':config['learning_rate'],
              'momentum':config['momentum'],
              'hidden_l1':config['hidden_l1'],
              'weight_l2':config['weight_l2'],
              }

    # Run with monitors if `plots` flag set to true
    for t_monitors, v_monitors in net.itertrain(**kwargs):
        for key in attrs:
            monitors['train'][key].append(t_monitors[key])
            monitors['valid'][key].append(v_monitors[key])

    if plots == True:
        plot_monitors(attrs, monitors['train'], monitors['valid'])

    # Run with `train` wrapper of `itertrain`
    #else:
    #    net.train(**kwargs)

    # Classify features against label/target value to get accuracy
    # where `valid` is a tuple with validation (features, label)
    accuracy = net.score(valid[0], valid[1])

    return net, accuracy, monitors


def get_best(results, key):
    '''Return results column 'key''s value from model with best accuracy'''
    mask_acc = results['accuracy'] == results['accuracy'].max()
    best_idx = results['train_time'][mask_acc].idxmin()
    return results[key][best_idx]


def _tune_net(train, valid, test, targets, configs, n_features, n_targets, plots):
    '''Train nets with varying configurations and `validation` set

    The determined best configuration is then used to find the resulting
    accuracy with the `test` dataset

    Args
    ----
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    targets: ndarray
        List of unique targets (bins) generated during data splitting
    configs: dict
        Dictionary of all permutation of network configuration parameters
        defined in `cfg_ann.yml` file
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    results_tune: pandas.DataFrame (dtype=object)
        Dataframe with columns for each different configuration:
            * parameter configuration
            * network object
            * accuracy value from validation set
            * training time.
    test_accuracy: float
        Accuracy value of best configuration from test dataset
    cms: dict
        Dictionary of confusion matrices for labels 'train' & 'valid'. These
        matrices are generated from the most optimal tuning network
        configuration.
    '''
    import numpy
    import pandas
    import time

    from . import utils_ann

    tune_cols = ['config', 'net', 'accuracy', 'train_time', 'monitors']
    results_tune = pandas.DataFrame(index=range(len(configs)),
                                    columns=tune_cols, dtype=object)
    for i in range(len(configs)):
        t0 = time.time()
        print('CONFIG', i, n_targets, len(targets))
        net, accuracy, monitors = create_algorithm(train, valid, configs[i],
                                                   n_features, n_targets, plots)
        t1 = time.time()

        results_tune['config'][i]     = configs[i]
        results_tune['net'][i]        = net
        results_tune['accuracy'][i]   = accuracy
        results_tune['train_time'][i] = t1 - t0
        results_tune['monitors'][i]   = monitors

    # Get neural net with best accuracy
    best_net = get_best(results_tune, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])
    print('Tuning test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    cms = utils_ann.get_confusion_matrices(best_net, train, test, targets)

    return results_tune, test_accuracy, cms


def _test_dataset_size(best_config, train, valid, test, targets, n_features,
        n_targets):
    '''Train nets with best configuration and varying dataset sizes

    Args
    ----
    best_config: dict
        Dictionary of configuration parameters for best performing network
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    targets: ndarray
        List of unique targets (bins) generated during data splitting
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network

    Returns
    -------
    results_dataset: pandas.DataFrame (dtype=object)
        Dataframe with columns for each different configuration:
            * parameter configuration
            * network object
            * accuracy value from validation set
            * fraction of original dataset
            * training time.
    test_accuracy: float
        Accuracy value of best configuration from test dataset
    cms: dict
        Dictionary of confusion matrices for labels 'train' & 'valid'. These
        matrices are generated from the most optimal dataset size network
        configuration.
    '''

    def _truncate_data(data, frac):
        '''Reduce data rows to `frac` of original

        Args
        ----
        data: Tuple containing numpy array of feature data and labels
        frac: percetange of original data to return

        Returns
        -------
        subset_frac: pandas.DataFrame
            Slice of original dataframe with len(data)*n rows
        '''

        n = len(data[0])
        subset_frac = (data[0][:int(round(n*frac))], data[1][:int(round(n*frac))])

        return subset_frac

    import numpy
    import pandas
    import time

    from . import utils_ann

    # Make array for storing results
    data_cols = ['config', 'net', 'accuracy', 'subset_frac', 'train_time',
                 'monitors', 'n_train', 'n_valid']

    subset_fractions = (numpy.arange(0,1,0.03))[1:]
    results_dataset = pandas.DataFrame(index=range(len(subset_fractions)),
                                    columns=data_cols, dtype=object)

    # Generate net and save results for each data subset
    for i in range(len(subset_fractions)):

        # Trim data sets to `frac` of original
        train_frac = _truncate_data(train, subset_fractions[i])

        #valid_frac = _truncate_data(valid, subset_fractions[i])
        #test_frac  = _truncate_data(test, subset_fractions[i])
        valid_frac = valid
        test_frac = test

        t0 = time.time()

        net, accuracy, monitors = create_algorithm(train_frac, valid_frac, best_config,
                                                   n_features, n_targets)
        t1 = time.time()

        results_dataset['config'][i]      = best_config
        results_dataset['net'][i]         = net
        results_dataset['accuracy'][i]    = accuracy
        results_dataset['subset_frac'][i] = subset_fractions[i]
        results_dataset['train_time'][i]  = t1 - t0
        results_dataset['monitors'][i]    = monitors
        results_dataset['n_train'][i]     = len(train_frac[0])
        results_dataset['n_valid'][i]     = len(valid_frac[0])

    # Get neural net with best accuracy
    best_net = get_best(results_dataset, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])

    # Print confusion matrices for train and test
    cms = utils_ann.get_confusion_matrices(best_net, train, test, targets)

    return results_dataset, test_accuracy, cms


def run(cfg_project, cfg_ann, debug=False, plots=False):
    '''
    Compile subglide data, tune network architecture and test dataset size

    Args
    ----
    cfg_project: OrderedDict
        Dictionary of configuration parameters for the current project
    cfg_ann: OrderedDict
        Dictionary of configuration parameters for the ANN
    debug: bool
        Swith for running single network configuration
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    cfg: dict
        Dictionary of network configuration parameters used
    data: tuple
        Tuple collecting training, validation, and test sets. Also includes bin
        deliniation values
    results: tuple
        Tuple collecting results dataframes and confusion matrices

    Note
    ----
    The validation set is split into `validation` and `test` sets, the
    first used for initial comparisons of various net configuration
    accuracies and the second for a clean test set to get an true accuracy,
    as reusing the `validation` set can cause the routine to overfit to the
    validation set.
    '''

    from collections import OrderedDict
    import climate
    import numpy
    import os
    import pandas
    import theano
    import yamlord

    from . import utils_ann
    from . import pre
    from . import post

    from .utils_ann import ppickle

    paths = cfg_project['paths']
    fnames = cfg_project['fnames']

    # Environment settings - logging, Theano, load configuration, set paths
    #---------------------------------------------------------------------------
    climate.enable_default_logging()
    theano.config.compute_test_value = 'ignore'

    # Configuration settings
    if debug is True:
        for key in cfg_ann['net_tuning'].keys():
            cfg_ann['net_tuning'][key] = [cfg_ann['net_tuning'][key][0],]

    # Load data from pre-processing
    path_model, sgls = pre.process(cfg_project, cfg_ann)
    sgls = sgls.dropna()
    cfg_project['ann_analyses'].append(path_model)

    print('\nSplit and normalize input/output data')
    features   = cfg_ann['net_all']['features']
    target     = cfg_ann['net_all']['target']
    n_targets  = cfg_ann['net_all']['n_targets']
    valid_frac = cfg_ann['net_all']['valid_frac']

    # Normalize input (features) and output (target)
    nsgls, bins = _normalize_data(sgls, features, target, n_targets)

    # Get indices of train, validation and test datasets
    ind_train, ind_valid, ind_test = _split_indices(nsgls, valid_frac)

    # Split dataframes into train, validation and test  (features, targets) tuples
    train, valid, test = _create_datasets(nsgls, ind_train, ind_valid, ind_test,
                                         features, target)
    print('train', len(train[0]), len(train[1]))
    print('valid', len(valid[0]), len(valid[1]))
    print('test',  len( test[0]), len( test[1]))

    # Save information on input data to config
    cfg_ann['net_all']['targets'] = [float(b) for b in bins]


    # Tuning - find optimal network architecture
    #---------------------------------------------------------------------------
    print('\nTune netork configuration')

    # Get all dict of all configuration permutations of params in `tune_params`
    configs = _get_configs(cfg_ann['net_tuning'])

    # Cycle through configurations storing configuration, net in `results_tune`
    n_features = len(cfg_ann['net_all']['features'])
    n_targets = cfg_ann['net_all']['n_targets']

    print('\nNumber of features: {}'.format(n_features))
    print('Number of targets: {}\n'.format(n_targets))

    results_tune, tune_accuracy, cms_tune = _tune_net(train,
                                                      valid,
                                                      test,
                                                      bins,
                                                      configs,
                                                      n_features,
                                                      n_targets,
                                                      plots)

    # Get neural net configuration with best accuracy
    best_config = get_best(results_tune, 'config')


    # Test effect of dataset size
    #---------------------------------------------------------------------------
    print('\nRun percentage of datasize tests')

    # Get randomly sorted and subsetted datasets to test effect of dataset_size
    # i.e. - a dataset with the first `subset_fraction` of samples.
    results_dataset, data_accuracy, cms_data = _test_dataset_size(best_config,
                                                                  train,
                                                                  valid,
                                                                  test,
                                                                  bins,
                                                                  n_features,
                                                                  n_targets,
                                                                  )

    print('\nTest data accuracy (Configuration tuning): {}'.format(tune_accuracy))
    print('Test data accuracy (Datasize test):        {}'.format(data_accuracy))


    # Save results and configuration to output directory
    #---------------------------------------------------------------------------

    # Create output directory if it does not exist
    path_output = os.path.join(paths['project'], paths['ann'],
                               path_model)
    os.makedirs(path_output, exist_ok=True)

    # Save config as a `*.yml` file to the output directory
    file_cfg_ann = os.path.join(path_output, fnames['ann']['cfg_ann'])
    file_cfg_project = os.path.join(path_output, fnames['project']['cfg'])
    yamlord.write_yaml(cfg_project, os.path.join(path_output, file_cfg_ann))
    yamlord.write_yaml(cfg_ann, os.path.join(path_output, file_cfg_ann))

    # Compiled SGLs after NaN drop
    utils_ann.ppickle(nsgls, os.path.join(path_output, fnames['ann']['ann_sgls']))

    # Save output data to analysis output directory
    tune_fname = fnames['ann']['tune']
    datasize_fname = fnames['ann']['dataset_size']
    ppickle(results_tune, os.path.join(path_output, tune_fname))
    ppickle(results_dataset, os.path.join(path_output, datasize_fname))

    # Save train, validation, test datasets
    ppickle(sgls, os.path.join(path_output, fnames['ann']['sgls']))
    ppickle(train, os.path.join(path_output, fnames['ann']['train']))
    ppickle(valid, os.path.join(path_output, fnames['ann']['valid']))
    ppickle(test, os.path.join(path_output, fnames['ann']['test']))

    ppickle(cms_tune, os.path.join(path_output, fnames['ann']['cms_tune']))
    ppickle(cms_data, os.path.join(path_output, fnames['ann']['cms_data']))

    return cfg_ann, (train, valid, test, bins), (results_tune, results_dataset,
                                                 cms_tune, cms_data)

if __name__ == '__main__':
    import yamlord
    #cfg, results_tune, results_dataset = run()

    debug = False
    plots = False

    cfg_project = yamlord.read_yaml('./cfg_project.yml')
    cfg_ann = yamlord.read_yaml('./cfg_ann.yml')

    cfg_ann, data, results = run(cfg_project, cfg_ann, debug=debug, plots=plots)

    train = data[0]
    valid = data[1]
    test  = data[2]
    bins  = data[3]

    results_tune    = results[0]
    results_dataset = results[1]
    cms_tune        = results[2]
    cms_data        = results[3]
