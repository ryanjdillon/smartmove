from os.path import join as _join


def roundup(x, order):
    return x if x % 10**order == 0 else x + 10**order - x % 10**order


def magnitude(x):
    import math
    return int(math.floor(math.log10(x)))


def compile_exp_data(path_project, field, cfg_ann):
    '''Walk root tag directory and compile derived values to dataframe

    Args
    ----
    field: pandas.DataFrame
        Field experiments with added rho_mod
    paths: OrderedDict
        Dictionary of smartmove project paths
    fnames: OrderedDict
        Dictionary of smartmove project filenames
    cfg_ann: OrderedDict
        Configuration dictionary for ANN analysis

    match data file from data_accelerometer/
    '''
    from collections import OrderedDict
    import numpy
    import os
    import pandas

    from ..ann import pre
    from ..config import paths, fnames

    # TODO all of this info could be put into cfg_filter.yaml, then just walk
    # through directories based on cfg_ann.yaml in model path

    def get_sgl_path(path_project, path_glide, path_exp, cfg):
        '''Concatenate path to subglide output directory'''
        from .. import utils

        glide_data_path = _join(path_project, path_glide, path_exp)
        subdir_path = utils.get_subdir(glide_data_path, cfg)
        glide_data_path = _join(glide_data_path, subdir_path)

        return glide_data_path

    cols = [ 'id', 'date', 'animal', 'mod_str', 'duration', 'n_dives',
            'n_sgls_asc', 'n_sgls_des', 'sgls_asc_str', 'sgls_des_str',
            'perc_des', 'perc_asc', 'tritium_id', 'density_kgm3', 'rho_mod']

    data = OrderedDict()
    exp_ids = field['exp_id'].values
    for c in cols:
        data[c] = numpy.array([None,]*len(exp_ids))

    # TODO rewrite this to just use `field` columns
    for i in range(len(exp_ids)):
        path = _join(path_project, paths['tag'], exp_ids[i])

        if os.path.isdir(path):

            exp = field.ix[i, 'exp_id']

            data['date'][i] = '{}-{}-{}'.format(exp[:4], exp[4:6], exp[6:8])

            data['animal'][i] = exp.split('_')[3]

            file_tag = _join(path, 'pydata_{}.p'.format(exp))
            tag = pandas.read_pickle(file_tag)

            file_mask = _join(path, 'mask_tag.p')
            masks = pandas.read_pickle(file_mask)

            start = tag['datetimes'][masks['exp']].iloc[0]
            stop = tag['datetimes'][masks['exp']].iloc[-1]

            hours, remainder = divmod((stop - start).total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            data['duration'][i] = '{:.0f}hr {:2.0f}min'.format(hours, minutes)

            # Create string for mod type
            block_type = field.ix[i, 'block_type']
            n_blocks = int(field.ix[i, 'n_blocks'])

            if n_blocks > 0:
                data['mod_str'][i] = '{:1d} {:>7}'.format(n_blocks, block_type)
            else:
                data['mod_str'][i] = '{:>9}'.format(block_type)

            # Get number of subglides during descent & ascent
            cfg = cfg_ann['data']
            glide_data_path = get_sgl_path(path_project, paths['glide'], exp, cfg)

            sgls = pandas.read_pickle(_join(glide_data_path,
                                            fnames['glide']['sgls']))
            # TODO perhaps dive ids being set to NaN incorrectly in glidepy
            sgls = sgls[~numpy.isnan(sgls['dive_id'].astype(float))]

            # TODO hackfix, correcting for incorrectly assigned phases in
            # calc_glide_des_asc(), swap back
            n_sgls_des = len(sgls[sgls['dive_phase'] == 'ascent'])
            n_sgls_asc = len(sgls[sgls['dive_phase'] == 'descent'])
            #n_sgls_des = len(sgls[sgls['dive_phase'] == 'descent'])
            #n_sgls_asc = len(sgls[sgls['dive_phase'] == 'ascent'])

            # Save number of SGLs with phase
            data['n_sgls_des'][i] = float(n_sgls_des)
            data['n_sgls_asc'][i] = float(n_sgls_asc)

            # Calculate percent of SGLs with phase per total N
            perc_des = n_sgls_des / (n_sgls_des+n_sgls_asc)*100
            perc_asc = n_sgls_asc / (n_sgls_des+n_sgls_asc)*100

            # Create string with number of SGLs with phase and (% of total N)
            fmt = '{:>4} ({:2.0f})'
            data['sgls_des_str'][i] = fmt.format(n_sgls_des, perc_des)
            data['sgls_asc_str'][i] = fmt.format(n_sgls_asc, perc_asc)

            # Save numeric percent of SGLs with phase per total N
            data['perc_des'][i] = perc_des
            data['perc_asc'][i] = perc_asc

            # Number of dives
            n_dives = len(numpy.unique(sgls['dive_id']))
            data['n_dives'][i] = n_dives

            # Isotop analysis ID
            data['tritium_id'][i] = field.ix[i, 'tritium_id']

            # Total original and modified body density
            data['density_kgm3'][i] = field.ix[i, 'density_kgm3']
            data['rho_mod'][i] = field.ix[i, 'rho_mod']

    data['id'] = numpy.array(list(range(len(data['date'])))) + 1

    field_all = pandas.DataFrame(data)

    for key in field_all.columns:
        try:
            field_all.ix[:,key] = pandas.to_numeric(field_all[key])
        except:
            pass

    return field_all


def filter_dataframe(df, ignore):
    '''Filter dataframe to columns not in ignore list'''
    return df[[c for c in df.columns if c not in ignore]][:]


def parse_col_txt(cols):
    # Unit string conversion
    unit_dict = {'kg':'kg', 'cm':'cm', 'l':'L', 'perc':'\%', 'kgm3':'kgm3'}

    names = ['']*len(cols)
    units = ['']*len(cols)
    for i in range(len(cols)):
        c = cols[i].split('_')
        names[i] = c[0][0].upper() + c[0][1:]
        if len(c) > 1:
            units[i] = unit_dict[c[1]]

    return names, units


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

#def perc_with_time(field_all):
#    ''''Dive and sub-glide plot with increasing durations'''
#    import datetime
#    import numpy
#    import pandas
#    import seaborn
#    import matplotlib.pyplot as plt
#
#    dates = numpy.zeros(len(field_all), dtype=object)
#    for i in range(len(field_all)):
#        dates[i] = datetime.datetime.strptime(field_all['duration'][i],
#                                              '%Hhr %Mmin')
#    deltas = dates - datetime.datetime(1900,1,1)
#    deltas = numpy.array(list(map(lambda x: x.seconds, deltas)))
#    sort_ind = deltas.argsort()
#
#    colors = seaborn.color_palette()
#
#    cols = {'n_dives':'Dives', 'n_sgls_des':'subglides (descent)',
#            'n_sgls_asc':'subglides (ascent)'}
#
#    fig, (ax0, ax1) = plt.subplots(1, 2)
#
#    i = 0
#    pos = 0
#    width = 0.25
#    for key in ['n_dives', 'n_sgls_des', 'n_sgls_asc']:
#        if i == 0:
#            ax = ax0
#            offset = width
#        else:
#            ax = ax1
#            pos += 1
#            offset = (pos*width) + width
#        labels = field_all.ix[sort_ind, 'label']
#        xs = numpy.arange((len(labels))) + offset
#        ys = field_all.ix[sort_ind, key]
#        ax.bar(xs, ys, width, color=colors[i], label=cols[key])
#        if i == 0:
#            for l, x, y in zip(durs[sort_ind], xs, ys):
#                text = ax.annotate(l, xy=(x,y),
#                                   horizontalalignment='left',
#                                   verticalalignment='bottom')
#                text.set_rotation(45)
#        ax.set_xticks(xs)
#        ax.set_xticklabels(field_all.ix[sort_ind, 'id'])
#        ax.set_ylabel('No. {}'.format(cols[key].split()[0]))
#        plt.xlabel('Exp. Duration (minutes)')
#        ax.legend(loc='upper left')
#        i += 1
#
#    plt.legend()
#    plt.show()
#
#    return None


