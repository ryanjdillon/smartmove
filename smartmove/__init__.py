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

    # Get path to pylleo requirements file
    module = importlib.util.find_spec('smartmove')
    module_path = os.path.split(module.origin)[0]

    fname_cfg_project = 'cfg_project.yml'
    fname_cfg_exp = 'cfg_experiments.yml'
    fname_cfg_ann = 'cfg_ann.yml'
    fname_cfg_glide = 'cfg_glide.yml'
    fname_cfg_filt = 'cfg_sgl-filt.yml'
    for fname in [fname_cfg_project, fname_cfg_exp, fname_cfg_ann,
                  fname_cfg_glide, fname_cfg_filt]:
        src = os.path.join(module_path, 'templates', fname)
        dst = os.path.join(path_project, fname)
        shutil.copyfile(src, dst)

    d = yamlord.read_yaml(os.path.join(path_project, fname_cfg_project))
    d['paths']['project'] = os.path.abspath(path_project)
    d['paths'].move_to_end('project', last=False)

    d['meta'] = OrderedDict()
    d.move_to_end('meta', last=False)
    d['meta']['created'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    d['meta']['versions'] = utils.get_versions('smartmove')

    yamlord.write_yaml(d, os.path.join(path_project, fname_cfg_project))

    # Create paths if not existing
    for key in d['paths'].keys():
        p = os.path.join(path_project, d['paths'][key])
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

    print('\nYour project directory has been created at {}.\n'
          'You must now copy your datalogger data to the `{}` directory, '
          'the body condition `.csv` files to the `{}` directory, and the CTD '
          '`.mat` file to the `{}` directory'.format(path_project,
                                                     d['paths']['tag'],
                                                     d['paths']['csv'],
                                                     d['paths']['ctd']))

    return None
