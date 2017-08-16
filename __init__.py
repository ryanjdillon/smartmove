from . import ann
from . import glideid
from . import utils
from .analysis import Analysis

def create_project(path_project):
    """Generate project based on values in *d*."""
    from collections import OrderedDict
    import datetime
    import importlib
    import os
    import shutil
    import yamlord

    from .config import paths, fnames

    # Get path to pylleo requirements file
    module = importlib.util.find_spec('smartmove')
    module_path = os.path.split(module.origin)[0]

    # Copy configuration files from `smartmove/_templates/` to `project_path`
    fname_cfg_project = fnames['cfg']['project']
    fname_cfg_exp = fnames['cfg']['exp_bounds']
    fname_cfg_ann = fnames['cfg']['ann']
    fname_cfg_glide = fnames['cfg']['glide']
    fname_cfg_filt = fnames['cfg']['filt']
    for fname in [fname_cfg_project, fname_cfg_exp, fname_cfg_ann,
                  fname_cfg_glide, fname_cfg_filt]:
        src = os.path.join(module_path, '_templates', fname)
        dst = os.path.join(path_project, fname)
        shutil.copyfile(src, dst)

    # Add creation datetime and versions to `cfg_project`
    d = yamlord.read_yaml(os.path.join(path_project, fname_cfg_project))
    d['meta'] = OrderedDict()
    d.move_to_end('meta', last=False)
    d['meta']['created'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    d['meta']['versions'] = utils.get_versions('smartmove')

    yamlord.write_yaml(d, os.path.join(path_project, fname_cfg_project))

    # Create project sub-paths if not existing
    for key in paths.keys():
        p = os.path.join(path_project, paths[key])
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

    print('\nYour project directory has been created at {}.\n'
          'You must now copy your datalogger data to the `{}` directory, '
          'the body condition `.csv` files to the `{}` directory, and the CTD '
          '`.mat` file to the `{}` directory'.format(path_project,
                                                     paths['tag'],
                                                     paths['csv'],
                                                     paths['ctd']))

    return None
