
def calc_mod_density(mass_kg, dens_kgm3, n_blocks, block_type):
    '''Calculate the modified density with attached blocks

    Args
    ----
    mass_kg: float
        mass of the seal (kg)
    dens_kgm3: float
        mass of the seal (kg/m^3)
    block_type: str
        Type of modification block for experiment (`weight` or `float`)
    n_blocks: int
        number of modifying blocks attached

    Returns
    -------
    rho_mod: float
        combined density of seal and attached blocks

    Notes
    -----
    All blocks dimensions were manually measured and then drawn in `FreeCAD`
    and the volumes calculated using the `FCInfo` macro.

    FLOAT:
      Float blocks were made from syntactic, non-compressible foam. They were
      attached with a neoprene pouch but were not compensated by a lead strip.
      The effect of the pouch buoyancy was not measured by considered to be
      negligible.

      Float block dimensions (with and without sailcloth)
        L:150mm, W:40mm, H:30mm (41, 29)
        shaved 45deg on all edges, 5.5 mm from edge

      Volume float           - 0.000174510114214 m³
      Mass float             - 0.034 kg
      Mass float w/sailcloth - 0.042 kg

    WEIGHT AND NEUTRAL:
      Neutral and weight blocks were made from PE plastic with 2 16mm diameter
      holes drilled length-wise to a depth of 144mm. These holes were
      left empty for neutral blocks. For weight blocks, these holes were filled
      with 140mm long 16mm diameter copper pipe that were filled with lead.

      The buoyancy of the neoprene pouch with which they were attached was
      compensated by a thin strip of lead that allowed the pouch to be neutral
      at surface. As the neoprene compresses with depth, this strip should
      overcompensate to a small degree.

      PE block dimensions:
        L:150mm, W:41mm, H:30mm
        rounded on one width edge - 15mm from edge length-wise and height-wise
      Hole dimensions (2 holes):
        L:144mm length, D:16mm
      Weight rod dimensions (one rod per hole):
        L:140mm, D:15mm

      Mass block w/holes   - 0.118 kg
      Mass weight bar      - 0.260 kg
      Mass weight block    - 0.118 kg + 2(0.260 kg) = 0.638 kg
      Volume block w/holes - 0.00012016274768 m³
      Volume weight bar    - 0.00000176714586764 m³
      Volume weight+2bars  - 1.202e-4 + 2(1.767e-6) = 1.237e-4

      4 weight tubes = 2 weights
    '''

    # Modifier block attributes
    block_types = ['weight', 'float', 'neutral']
    mod_vol_m3 = {'weight':1.237e-4, 'float':1.745e-4, 'neutral':1.202e-4}
    mod_mass_kg = {'weight': 0.638,  'float': 0.034, 'neutral':0.118}

    seal_vol_m3 = mass_kg / dens_kgm3

    # Calculate combined density
    if block_type in block_types:
        total_mass = (mass_kg + (n_blocks * mod_mass_kg[block_type]))
        total_vol = (seal_vol_m3 + (n_blocks * mod_vol_m3[block_type]))
        rho_mod = total_mass / total_vol

    elif block_type == 'control':
        rho_mod = mass_kg / seal_vol_m3

    # Raise error if incorrect `block_type` passed
    else:
        print(block_type)
        raise KeyError("`block_type` must be 'weight', 'float', or 'neutral'")

    return rho_mod


def simulate_density(mass_kg=40, bd_start=1000, n_bd=101, block_start=1,
        n_blocks=8):
    '''Produce a range of body densities given an initial mass and density'''
    import numpy
    import pandas

    # Range of body densities to test
    types = ['weight', 'float']
    bd_range = bd_start+numpy.arange(0, n_bd)
    bodydensities = numpy.tile(numpy.repeat(bd_range, n_blocks), len(types))

    block_range = block_start+numpy.arange(0, n_blocks)
    blocks = numpy.tile(numpy.tile(block_range, n_bd), len(types))

    types = numpy.repeat(types, n_bd*n_blocks)

    columns = ['type', 'dens_kgm3', 'n_blocks', 'rho_mod', 'delta_rho']
    df = pandas.DataFrame(index=range(len(bodydensities)), columns=columns)

    for i in range(len(df)):
        print(i, df.index[i])
        df.loc[df.index[i], 'type'] = types[i]
        df.loc[df.index[i], 'dens_kgm3'] = bodydensities[i]
        df.loc[df.index[i], 'n_blocks'] = blocks[i]
        rho_mod = calc_mod_density(mass_kg, bodydensities[i], blocks[i], t)
        df.loc[df.index[i], 'rho_mod'] = rho_mod
        df.loc[df.index[i], 'delta_ro'] = rho_mod - bodydensities[i]

    return df


def add_rhomod(file_field, file_isotope):
    import numpy
    import pandas

    import pyotelem

    # Load experiments and convert datetimes to datetime
    field = pandas.read_csv(file_field, comment='#')
    field['date'] = pandas.to_datetime(field['date'])

    # Remove rows without an ID (experiments not to be used)
    field = field[~numpy.isnan(field['id'])]

    # Load isotope analysis and isotopemetric data, skip first 4 rows
    isotope = pandas.read_csv(file_isotope, comment='#')

    # Get percent body compositions, including density - what we want
    perc_comps = pyotelem.physio_seal.lip2dens(isotope['fat_perc'])
    isotope['density_kgm3'] = perc_comps['density']*1000

    # List of columns to add to experiments from isotope-isotope data
    cols = ['mass_kg', 'length_cm', 'girth_cm','water_l', 'water_perc', 'fat_kg',
            'fat_perc', 'protein_kg', 'protein_perc', 'density_kgm3']

    # Create new columns in experiment dataframe
    for col in cols:
        field[col] = numpy.nan
    field['rho_mod'] = numpy.nan

    # Add data from isotope-morpho dataframe to experiments dataframe
    for i in range(len(field)):
        idx = int(field['tritium_id'].iloc[i])
        midx = numpy.where(isotope['id'] == idx)[0][0]
        field.loc[i, cols] = isotope.ix[midx,cols]

        # Cacluate total density with modification, buoyant forces
        rho_mod = calc_mod_density(field['mass_kg'][i],
                                      field['density_kgm3'][i],
                                      field['n_blocks'][i],
                                      field['block_type'][i])

        field.loc[i, 'rho_mod'] = rho_mod

    return field, isotope


def _add_ids_to_df(df, exp_id, animal_id=None, tag_id=None):
    '''Add columns to dataframe with experiment ID and animal ID

    if list of ids passed, must be equal to number of rows in `df`
    '''

    df['exp_id'] = exp_id

    # Add parameter if passed
    if animal_id is not None:
        df['animal_id']  = animal_id

    if tag_id is not None:
        df['tag_id']  = tag_id

    return df


def _compile_experiments(path_project, path_glide, cfg, fname_sgls,
        fname_mask_sgls_filt, manual_selection=True):
    '''Compile data from experiments into three dataframes for MCMC input'''
    import numpy
    import os
    import pandas
    import pyotelem

    from .. import utils

    # List of paths to process
    path_exps = list()

    # Empty lists for appending IDs of each experiment
    exp_ids    = list()
    animal_ids = list()
    tag_ids    = list()

    print('''
          ┌----------------------------------------------------------------┐
          | Compiling glide analysis output to single file for model input |
          └----------------------------------------------------------------┘
          ''')

    # Iterate through experiment directories in glide analysis path
    first_iter = True

    # Generate list of possible paths to process in glide directory
    glide_paths_found = False
    for path_exp in os.listdir(os.path.join(path_project, path_glide)):
        path_data_glide = os.path.join(path_project, path_glide, path_exp)
        if os.path.isdir(path_data_glide):
            path_exps.append(path_exp)
            glide_paths_found = True

    # Throw exception if no data found in glide path
    if not glide_paths_found:
        raise SystemError('No glide paths found, check input directories '
                          'for errors\n'
                          'path_project: {}\n'
                          'path_glide: {}\n'.format(path_project, path_glide))

    # Run manual input data path selection, else process all present paths
    path_exps = sorted(path_exps)
    if manual_selection:
        msg = 'path numbers to compile to single dataset.\n'
        process_ind = pyotelem.utils.get_dir_indices(msg, path_exps)
    else:
        process_ind = range(len(path_exps))

    # Process user selected paths
    for i in process_ind:
        path_exp = path_exps[i]

        # Concatenate data path
        path_data_glide = os.path.join(path_project, path_glide, path_exp)
        path_subdir = utils.get_subdir(path_data_glide, cfg)
        path_data_glide = os.path.join(path_data_glide, path_subdir)

        print('Processing {}'.format(path_exp))

        # Get experiment/animal ID from directory name
        exp_id    = path_exp
        tag_id    = exp_id.split('_')[2]
        animal_id = exp_id.split('_')[3]

        # Append experiment/animal id to list for `exps` df creation
        exp_ids.append(exp_id)
        animal_ids.append(animal_id)
        tag_ids.append(tag_id)

        # Read sgls dataframe, filter out only desired columns
        file_sgls = os.path.join(path_data_glide, fname_sgls)
        sgls_exp  = pandas.read_pickle(file_sgls)

        # Filter with saved mask meeting criteria
        file_mask_sgls_filt = os.path.join(path_data_glide, fname_mask_sgls_filt)
        mask_sgls = numpy.load(file_mask_sgls_filt)
        sgls_exp  = sgls_exp[mask_sgls]

        # Get unique dives in which all subglides occur
        dive_ids_exp = numpy.unique(sgls_exp['dive_id'][:])
        dives_exp = pandas.DataFrame(index=range(len(dive_ids_exp)))
        dives_exp['dive_id'] = dive_ids_exp

        # Add exp_id/animal_id fields
        sgls_exp  = _add_ids_to_df(sgls_exp, exp_id)
        dives_exp = _add_ids_to_df(dives_exp, exp_id)

        # Append experiment sgl array to array with all exps to analyze
        if first_iter is True:
            first_iter = False
            sgls_all   = sgls_exp
            dives_all  = dives_exp
        else:
            sgls_all  = pandas.concat([sgls_all, sgls_exp], ignore_index = True)
            dives_all = pandas.concat([dives_all, dives_exp], ignore_index = True)

    # Create experiments dataframe
    exps_all = pandas.DataFrame(index=range(len(exp_ids)))
    exps_all = _add_ids_to_df(exps_all, exp_ids, animal_id=animal_ids,
                               tag_id=tag_ids)

    return exps_all, sgls_all, dives_all


def _create_ann_inputs(cfg_ann, path_project, path_tag, path_glide, path_ann, path_csv,
        field, fname_sgls, fname_mask_sgls_filt, sgl_cols,
        manual_selection=True):
    '''Compile all experiment data for ann model input'''
    import numpy
    import os
    import pandas

    def insert_field_col_to_sgls(sgls, field):
        '''Insert bodycondition from nearest date in field to sgls dataframes'''
        import numpy

        col_name = 'rho_mod'

        # Create empty column for body condition target values
        sgls = sgls.assign(**{col_name:numpy.full(len(sgls), numpy.nan)})

        exp_ids = numpy.unique(sgls['exp_id'].values)

        for exp_id in exp_ids:
            mask_sgl = sgls['exp_id'] == exp_id
            mask_field = field['exp_id'] == exp_id

            try:
                value = field.ix[mask_field, 'rho_mod'].values[0]
                sgls.ix[mask_sgl, col_name] = value
            except Exception as e:
                if exp_id not in numpy.unique(field['exp_id']):
                    raise Exception('{} not in '
                                    '`field_experiments.csv`'.format(exp_id))
                elif 'rho_mod' not in field.columns:
                    raise Exception('`rho_mod` field not found in '
                                    'generated `field` dataframe with added '
                                    'body density.')
                else:
                    raise Exception(e)
        return sgls

    # Compile subglide inputs for all experiments
    exps_all, sgls_all, dives_all = _compile_experiments(path_project,
                                                         path_glide,
                                                         cfg_ann['data'],
                                                         fname_sgls,
                                                         fname_mask_sgls_filt)

    # Add integer dive_phase column
    des = sgls_all['dive_phase'] == 'descent'
    asc = sgls_all['dive_phase'] == 'ascent'

    sgls_all['dive_phase_int'] = 0
    sgls_all.ix[des, 'dive_phase_int'] = -1
    sgls_all.ix[asc, 'dive_phase_int'] = 1
    sgls_all.ix[~des&~asc, 'dive_phase_int'] = 0

    # Extract only columns useful for ann
    sgls = sgls_all[sgl_cols]

    # Add column with body condition target values to `sgls`
    sgls = insert_field_col_to_sgls(sgls, field)

    # Save output
    sgls.to_pickle(os.path.join(path_project, path_ann, 'sgls_all.p'))

    return exps_all, sgls, dives_all


def _print_dict_values(cfg_ann):
    '''Print parameterization of input data to be analyzed'''

    labels = ['glides', 'sgls', 'filter']

    line = '-' * max([len(l) for l in labels]) * 4

    print('\nInput data configuration:')

    for label in labels:
        pad_front = ' '*((len(line) - len(label))//2)
        pad_back  = ' '*((len(line) - len(label))%2)
        print('\n' + pad_front + label.upper() + pad_back)
        print(line)
        space = str(max([len(key) for key in list(cfg_ann[label])]))
        for key, value in cfg_ann[label].items():
            print(('{:>'+space+'}: {:04.2f}').format(key, value))

        print(line)

    return None


def process(cfg_project, cfg_ann):
    '''Compile data with modified body density, generate output directory

    The ANN input data generated in this routine is saved in python pickle
    format (binary) to the generated output directory `path_model`.

    Args
    ----
    cfg_project: OrderedDict
        Dictionary of configuration parameters for the current project
    cfg_ann: OrderedDict
        Dictionary of configuration parameters for the ANN

    Returns
    -------
    path_output: str
        Absolute path to ANN output directory
    sgls: pandas.DataFrame
        A dataframe of all subglide data compiled from the glide identification
        performed by `glideid`.
    '''
    import datetime
    import os
    import yamlord

    from . import utils_ann

    # Set paths
    #---------------------------------------------------------------------------
    # Load project paths and input/output filenames
    paths = cfg_project['paths']
    fnames = cfg_project['fnames']

    # Define output directory and create if does not exist
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path_model = 'theanets_{}'.format(now)

    # Print input data configuration
    _print_dict_values(cfg_ann['data'])

    # Generate experiments/isotope pandas from csv
    file_field = os.path.join(paths['project'], paths['csv'],
                              fnames['csv']['field'])
    file_isotope = os.path.join(paths['project'], paths['csv'],
                                fnames['csv']['isotope'])

    field, isotope = add_rhomod(file_field, file_isotope)


    # Compile experiment data
    #---------------------------------------------------------------------------
    sgl_cols = cfg_ann['data']['sgl_cols'] + cfg_ann['net_all']['features']

    # Compile output from glides into single input dataframe
    _, sgls, _ = _create_ann_inputs(cfg_ann,
                                    paths['project'],
                                    paths['tag'],
                                    paths['glide'],
                                    paths['ann'],
                                    paths['csv'],
                                    field,
                                    fnames['glide']['sgls'],
                                    fnames['glide']['mask_sgls_filt'],
                                    sgl_cols,
                                    manual_selection=True)

    return path_model, sgls
