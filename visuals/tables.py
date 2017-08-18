from os.path import join as _join

def table_isotope(name, attrs, path_table, data):
    from collections import OrderedDict

    from . import latex
    from . import utils

    # Create and write experiment table
    print('Creating isotope experiment table')
    cols = OrderedDict()
    #cols['id']           = '%.0f'
    cols['experiments']  = '%str'
    cols['date']         = '%str'
    cols['animal']       = '%str'
    cols['mass_kg']      = '%.1f'
    #cols['length_cm']    = '%.0f'
    #cols['girth_cm']     = '%.0f'
    cols['water_l']      = '%.1f'
    cols['water_perc']   = '%.1f'
    cols['fat_kg']       = '%.1f'
    cols['fat_perc']     = '%.1f'
    cols['protein_kg']   = '%.1f'
    cols['protein_perc'] = '%.1f'
    cols['density_kgm3'] = '%.1f'

    names, units = utils.parse_col_txt(data.columns)
    names = ['Experiments', 'Date', 'Animal', 'Mass', 'Water', 'Water',
             'Fat', 'Fat', 'Protein', 'Protein', 'Density']
    units = ['', '', '', 'kg', 'L', '\%', 'kg', '\%', 'kg', '\%',
             r'kg m\textsuperscript{-3}']
    headers = [names, units]

    table = latex.tables.write_table(path_table,
                                     name,
                                     data,
                                     cols,
                                     headers,
                                     adjustwidth=attrs['adjustwidth'],
                                     tiny=False,
                                     title=attrs['title'],
                                     caption=attrs['caption'],
                                     centering=True,
                                     extrarowheight=attrs['extrarowheight'],
                                     label=name,
                                     notes=attrs['notes'])

    return table


def table_exps(name, attrs, path_table, data):
    '''
    '''
    from collections import OrderedDict

    from . import latex

    # Create and write experiment table
    print('Creating experiment table')
    cols = OrderedDict()

    cols['id']           = '%.0f'
    cols['date']         = '%str'
    cols['animal']       = '%str'
    cols['mod_str']      = '%str'
    cols['duration']     = '%str'
    cols['n_dives']      = '%.0f'
    cols['sgls_des_str'] = '%str'
    cols['sgls_asc_str'] = '%str'
    cols['density_kgm3'] = '%.1f'
    cols['rho_mod']   = '%.1f'

    # Column names to print in table
    names = ['ID', 'Date', 'Animal', 'Mod.', 'Duration', 'Dives',
             'Descent SGLs', 'Ascent SGLs', 'Density',
             r'$\rho\textsubscript{mod}$']

    units = ['', '', '', '', '', '\#', '\# (\%)', '\# (\%)',
            r'kg m\textsuperscript{-3}', r'kg m\textsuperscript{-3}']
    headers = [names, units]

    table = latex.tables.write_table(path_table,
                                     name,
                                     data,
                                     cols,
                                     headers,
                                     adjustwidth=attrs['adjustwidth'],
                                     tiny=False,
                                     title=attrs['title'],
                                     caption=attrs['caption'],
                                     centering=True,
                                     extrarowheight=attrs['extrarowheight'],
                                     label=name,
                                     notes=attrs['notes'])

    return table


def table_ann_params(name, attrs, path_table, cfg_ann):
    from collections import OrderedDict

    from . import latex

    cfg = cfg_ann['net_tuning']
    print('Creating ANN parameter table')

    ignore = ['momentum', 'patience', 'min_improvement', 'validate_every']
    keys = [k for k in cfg.keys() if k not in ignore]

    cols = OrderedDict()
    cols['Hyperparameter'] = '%str'
    cols['Values']         = '%str'

    data = pandas.DataFrame(index=range(len(keys)), columns=cols.keys())

    for i in range(len(keys)):
        # Make string of list values in dictionary of params
        values = cfg[keys[i]]
        values = ', '.join([str(j) for j in values])

        data['Hyperparameter'].iloc[i] = keys[i].replace('_',' ')
        data['Values'].iloc[i]         = values

    names = data.columns
    headers = [names,]

    table = latex.tables.write_table(path_table,
                                     name,
                                     data,
                                     cols,
                                     headers,
                                     adjustwidth=attrs['adjustwidth'],
                                     tiny=False,
                                     title=attrs['title'],
                                     caption=attrs['caption'],
                                     centering=True,
                                     extrarowheight=attrs['extrarowheight'],
                                     label=name,
                                     notes=attrs['notes'])

    return table


def table_ann_feature_stats(name, attrs, path_table, data):
    from collections import OrderedDict

    from . import latex
    from . import utils

    # Create and write experiment table
    print('Creating input feature stats table')
    cols = OrderedDict()
    cols['feature'] = '%str'
    cols['min']     = '%.2f'
    cols['max']     = '%.2f'
    cols['mean']    = '%.2f'
    cols['std']     = '%.2f'

    names, units = utils.parse_col_txt(data.columns)
    names = ['Input feature', 'Min.', 'Max.', 'Mean', 'STD']

    headers = [names,]
    table = latex.tables.write_table(path_table,
                                     name,
                                     data,
                                     cols,
                                     headers,
                                     adjustwidth=attrs['adjustwidth'],
                                     tiny=False,
                                     title=attrs['title'],
                                     caption=attrs['caption'],
                                     centering=True,
                                     extrarowheight=attrs['extrarowheight'],
                                     label=name,
                                     notes=attrs['notes'])

    return table


def table_ann_target_stats(name, attrs, path_table, data):
    from collections import OrderedDict

    from . import latex
    from . import utils

    # Create and write experiment table
    print('Creating target value stats table')
    cols = OrderedDict()
    cols['bin']  = '%str'
    cols['n']    = '%.0f'
    cols['perc'] = '%.2f'

    names, units = utils.parse_col_txt(data.columns)
    names = ['Bin', 'Number', '\% of compiled']
    headers = [names,]

    table = latex.tables.write_table(path_table,
                                     name,
                                     data,
                                     cols,
                                     headers,
                                     adjustwidth=attrs['adjustwidth'],
                                     tiny=False,
                                     title=attrs['title'],
                                     caption=attrs['caption'],
                                     centering=True,
                                     extrarowheight=attrs['extrarowheight'],
                                     label=name,
                                     notes=attrs['notes'])

    return table


def make_all(path_project, path_analysis):
    import numpy
    import os
    import pandas
    import yamlord

    from ..ann import pre
    from ..config import paths, fnames

    from . import utils
    from . import table_attributes

    # Load ANN configuration for selection analysis
    file_cfg_ann = _join(path_project, paths['ann'], path_analysis,
                         fnames['cfg']['ann'])
    cfg_ann = yamlord.read_yaml(file_cfg_ann)

    # Load table titlesi and captions
    table_attrs  = table_attributes.get_all()

    # Create table output directory
    path_table = _join(path_project, 'paper/tables')
    os.makedirs(path_table, exist_ok=True)

    # http://stackoverflow.com/a/6605085/943773
    # TODO add duration (h) and tissue density (kg/m^3)

    # Load field and isotop data
    file_field   = _join(path_project, paths['csv'], fnames['csv']['field'])
    file_isotope = _join(path_project, paths['csv'], fnames['csv']['isotope'])
    field, isotope = pre.add_rhomod(file_field, file_isotope)

    # Compile experiments adding columns necessary for tables/figures
    exps_all = utils.compile_exp_data(path_project, field, cfg_ann)

    # Create isotope experiments table
    ignore_isotope = ['length', 'girth', 'contributer1', 'contributer2', 'notes']
    name_isotope   = table_isotope.__name__
    isotope = utils.filter_dataframe(isotope, ignore_isotope)
    table_isotope(name_isotope, table_attrs[name_isotope], path_table, isotope)

    # Create compiled experiment table
    name_exps = table_exps.__name__
    table_exps(name_exps, table_attrs[name_exps], path_table, exps_all)

    ## Create ANN hyperparameter table
    #name_ann_params = table_ann_params.__name__
    #table_ann_params(name_ann_params, table_attrs[name_ann_params], path_table,
    #                 cfg_ann)

    # Create input feature statistics table
    name_fstats  = table_ann_feature_stats.__name__
    fname_sgls   = fnames['ann']['sgls']
    file_sgls    = _join(path_project, paths['ann'], path_analysis, fname_sgls)
    sgls_all = pandas.read_pickle(file_sgls)
    sgls_all = sgls_all.dropna()
    feature_cols = ['abs_depth_change',
                    'dive_phase_int',
                    'mean_a',
                    'mean_depth',
                    'mean_pitch',
                    'mean_speed',
                    'mean_swdensity',
                    'total_depth_change',
                    'total_speed_change',]
    input_stats = utils.input_feature_stats(sgls_all, feature_cols)
    table_ann_feature_stats(name_fstats, table_attrs[name_fstats], path_table,
                            input_stats)

    # Create target value statistics table
    name_tstats  = table_ann_target_stats.__name__
    file_train  = _join(path_project, paths['ann'], path_analysis,
                        fnames['ann']['train'])
    file_valid  = _join(path_project, paths['ann'], path_analysis,
                        fnames['ann']['valid'])
    file_test   = _join(path_project, paths['ann'], path_analysis,
                        fnames['ann']['test'])
    train = pandas.read_pickle(file_train)
    valid = pandas.read_pickle(file_train)
    test  = pandas.read_pickle(file_train)

    target_stats = utils.target_value_stats(train, valid, test)
    table_ann_target_stats(name_tstats, table_attrs[name_tstats], path_table,
                           target_stats)
    return None
