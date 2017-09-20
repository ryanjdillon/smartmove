'''
This module contains utility functions using in Smartmove
'''

def mask_from_noncontiguous_indices(n, start_ind, stop_ind):
    '''Create boolean mask from start stop indices of noncontiguous regions

    Args
    ----
    n: int
        length of boolean array to fill
    start_ind: numpy.ndarray
        start index positions of non-contiguous regions
    stop_ind: numpy.ndarray
        stop index positions of non-contiguous regions

    Returns
    -------
    mask: numpy.ndarray, shape (n,), dtype boolean
        boolean mask array
    '''
    import numpy

    mask = numpy.zeros(n, dtype=bool)

    for i in range(len(start_ind)):
        mask[start_ind[i]:stop_ind[i]] = True

    return mask


def get_n_lines(file_path):
    '''Get number of lines by calling bash command wc

    Args
    ----
    file_path: str
        File whose lines to count

    Returns
    -------
    n_lines: int
        Number of lines in file
    '''
    import os
    import subprocess

    cmd = 'wc -l {0}'.format(file_path)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    n_lines = int((output).readlines()[0].split()[0])

    return n_lines


def get_versions(module_name):
    '''Return versions for repository and packages in requirements file

    Args
    ----
    module_name: str
        Name of module calling this routine, stored with local git hash

    Returns
    -------
    versions: OrderedDict
        Dictionary of module name and dependencies with versions
    '''
    from collections import OrderedDict
    import importlib
    import os

    versions = OrderedDict()

    module = importlib.util.find_spec(module_name)

    # Get path to pylleo requirements file
    module_path  = os.path.split(module.origin)[0]
    requirements = os.path.join(module_path, 'requirements.txt')

    # Add git hash for module to dict
    cwd = os.getcwd()
    os.chdir(module_path)
    try:
        versions[module_name] = get_githash('long')
    except:
        versions[module_name] = module.__version__

    os.chdir(cwd)

    return versions


def get_githash(hash_type):
    '''Add git commit for reference to code that produced data

    Args
    ----
    hash_type: str
        keyword determining length of has. 'long' gives full hash, 'short'
        gives 6 character hash

    Returns
    -------
    git_hash: str
        Git hash as a 6 or 40 char string depending on keywork `hash_type`
    '''
    import subprocess

    cmd = dict()
    cmd['long']  = ['git', 'rev-parse', 'HEAD']
    cmd['short'] = ['git', 'rev-parse', '--short', 'HEAD']

    return subprocess.check_output(cmd[hash_type]).decode('ascii').strip()


def symlink(src, dest):
    '''Failsafe creation of symlink if symlink already exists

    Args
    ----
    src: str
        Path or file to create symlink to
    dest: str
        Path of new symlink
    '''
    import os

    # Attempt to delete existing symlink
    try:
        os.remove(dest)
    except:
        pass

    os.symlink(src, dest)

    return None


def cat_path(d, ignore):
    '''Concatenate dictionary key, value pairs to a single string

    Args
    ----
    d: dict
        Dictionary for which key, value pairs should be concatenated to str
    ignore: iterable
        List of keys to exclude from concatenated string

    Returns
    -------
    s: str
        String with concatenated key, value pairs
    '''

    items = list(d.items())
    s = ''
    for i in range(len(items)):
        key, value = items[i]
        if key not in set(ignore):
            s += '{}_{}__'.format(key, value)

    return s[:-2]


def _parse_subdir(path):
    '''Parse parameters in names of child directories to pandas dataframe

    Child directories in `path` are parsed so that the parameter values in
    their directory names can be easily searched using a pandas.DataFrame.
    Parameters are separated by double `__` and values by single `_`. Names
    that include an `_` are joined back together after they are split

    Args
    ----
    path: str
        Parent path with directories names with parameters to parse

    Returns
    -------
    paths_df: pandas.DataFrame
        Dataframe with one row for each respective child directory and one
        column for each parameter.

    '''
    import os
    import numpy
    import pandas

    dir_list = numpy.asarray(os.listdir(path), dtype=object)

    # Search root directory for directories to parse
    for i in range(len(dir_list)):
        if os.path.isdir(os.path.join(path,dir_list[i])):
            name = dir_list[i]
            # Split parameters in name
            dir_list[i] = dir_list[i].split('__')
            for j in range(len(dir_list[i])):
                param = dir_list[i][j].split('_')
                # Join names with `_` back together, make key/value tuple
                key = '_'.join(param[:-1])
                value = param[-1]
                if value == 'None':
                    value = numpy.nan
                param = (key, float(value))
                dir_list[i][j] = param
            # Convert list of tuples to dictionary
            dir_list[i] = dict(dir_list[i])
            # Add directory name to dict for later retrieval
            dir_list[i]['name'] = name
        else:
            dir_list[i] = ''

    # Remove entries that are files
    dir_list = dir_list[~(dir_list == '')]

    # Convert list of dictionaries to dictionary of lists
    keys = dir_list[0].keys()
    params = dict()
    for i in range(len(dir_list)):
        for key in dir_list[i]:
            if key not in params:
                params[key] = numpy.zeros(len(dir_list), object)
            params[key][i] = dir_list[i][key]

    return pandas.DataFrame(params)


def get_subdir(path, cfg):
    '''Get path to glide output data for a given `cfg_glide`

    Args
    ----
    path: str
        Tag data parent path
    cfg: OrderedDict
        Composite dictions of cfg dicts

    Returns
    -------
    path_data: str
        Absolute path to glide data output path
    '''
    import os

    import pyotelem

    def match_subdir(path, cfg):
        import numpy

        n_subdirs = 0
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                n_subdirs += 1

        if n_subdirs == 0:
            raise SystemError('No data subdirectories in {}'.format(path))

        params = _parse_subdir(path)
        mask = numpy.zeros(n_subdirs, dtype=bool)

        # Evalute directory params against configuration params
        # Set directory mask to True where all parameters are matching
        for i in range(len(params)):
            match = list()
            for key, val in cfg.items():
                if params[key].iloc[i] == val:
                    match.append(True)
                else:
                    match.append(False)
            mask[i] = all(match)

        idx = numpy.where(mask)[0]
        if idx.size > 1:
            raise SystemError('More than one matching directory found')
        else:
            idx = idx[0]
            return params['name'].iloc[idx]


    # TODO this requires that each exp have same paramter values as in
    # cfg dict (i.e. cfg_ann and cfg_mcmc yaml)
    subdir_glide = match_subdir(path, cfg['glides'])

    path = os.path.join(path, subdir_glide)
    subdir_sgl   = match_subdir(path, cfg['sgls'])

    path = os.path.join(path, subdir_sgl)
    subdir_filt  = match_subdir(path, cfg['filter'])

    return os.path.join(subdir_glide, subdir_sgl, subdir_filt)


def filter_sgls(n_samples, exp_ind, sgls, max_pitch, min_depth,
        max_depth_delta, min_speed, max_speed, max_speed_delta):
    '''Create mask filtering only glides matching criterea
    
    Args
    ----
    n_samples: int
        Total number of samples in tag data
    exp_ind: ndarray
        Boolean array to slice tag data to only experimental period
    sgls: 
        
    max_pitch: float
        Maximum allowable pitch during sub-glide
    min_depth: float
        Minimum allowable depth during sub-glide
    max_depth_delta: float
        Maximum allowable change in depth during sub-glide
    min_speed: float
        Minimum allowable speed during sub-glide
    max_speed: float
        Maximum allowable speed during sub-glide
    max_speed_delta: float
        Maximum allowable change in speed during sub-glide

    Returns
    -------
    mask_data_sgl: ndarray
        Boolean mask to slice tag dataframe to filtered sub-glides
    mask_sgls: ndarray
        Boolean mask to slice sgls dataframe to filtered sub-glides
    '''
    import numpy

    import pyotelem

    # Defined experiment indices
    mask_exp = (sgls['start_idx'] >= exp_ind[0]) & \
               (sgls['stop_idx'] <= exp_ind[-1])

    # Found within a dive
    mask_divid = ~numpy.isnan(sgls['dive_id'].astype(float))

    # Uniformity in phase (dive direction)
    mask_phase = (sgls['dive_phase'] == 'descent') | \
                 (sgls['dive_phase'] == 'ascent')

    # Depth change and minimum depth constraints
    mask_depth = (sgls['total_depth_change'] < max_depth_delta) & \
                 (sgls['total_depth_change'] > min_depth)

    # Pitch angle constraint
    mask_deg = (sgls['mean_pitch'] <  max_pitch) & \
               (sgls['mean_pitch'] > -max_pitch)

    # Speed constraints
    mask_speed = (sgls['mean_speed'] > min_speed) & \
                 (sgls['mean_speed'] < max_speed) & \
                 (sgls['total_speed_change'] < max_speed_delta)

    # Concatenate masks
    mask_sgls = mask_divid & mask_phase & mask_exp & \
                mask_deg    & mask_depth & mask_speed

    # Extract glide start/stop indices within above constraints
    start_ind = sgls[mask_sgls]['start_idx'].values
    stop_ind  = sgls[mask_sgls]['stop_idx'].values

    # Create mask for all data from valid start/stop indices
    mask_data_sgl = mask_from_noncontiguous_indices(n_samples, start_ind,
                                                               stop_ind)
    # Catch error with no matching subglides
    num_valid_sgls = len(numpy.where(mask_sgls)[0])
    if num_valid_sgls == 0:
        raise SystemError('No sublides found meeting filter criteria')

    return mask_data_sgl, mask_sgls
