from os.path import join as _join
from pandas import read_pickle as _rpickle

class Analysis(object):
    '''Smartmove helper class for running analyses and output to project path

    Args
    ----
    path_project: str
        Path to project directory created with `smartmove.create_project()`
        method.
    '''

    def __init__(self, path_project, sgl_dur=2):
        '''Initiate the Analysis object

        Args
        ----
        sgl_dur: int
            Duration of sub-glide splits (seconds, Defaults to `2`)
        '''
        import os
        import yamlord

        from .config import paths, fnames

        self.path_project = os.path.abspath(path_project)

        file_cfg_project = _join(path_project, fnames['cfg']['project'])
        self.cfg_project = yamlord.read_yaml(file_cfg_project)

        file_cfg_ann = _join(path_project, fnames['cfg']['ann'])
        file_cfg_glide = _join(path_project, fnames['cfg']['glide'])
        file_cfg_filt = _join(path_project, fnames['cfg']['filt'])
        file_cfg_experiments = _join(path_project, fnames['cfg']['exp_bounds'])

        self.cfg_ann = yamlord.read_yaml(file_cfg_ann)
        self.cfg_glide = yamlord.read_yaml(file_cfg_glide)
        self.cfg_filt = yamlord.read_yaml(file_cfg_filt)
        self.cfg_experiments = yamlord.read_yaml(file_cfg_experiments)

        self.sgl_dur = sgl_dur

        return None

    def run_glides(self, plots=True, debug=False):
        '''Peform glide identification analysis

        Args
        ----
        plots: bool
            Switch for turning on plots (Default `True`). When activated plots for
            reviewing signal processing will be displayed.
        debug: bool
            Switch for turning on debugging (Default `False`). When activated values
            for `cutoff_freq` and `J` will be set to generic values and diagnostic
            plots of the `speed` parameter in `tag` will be displayed.
        '''
        from . import glideid

        glideid.glideid.run(self.path_project, self.cfg_project,
                            self.cfg_glide, self.cfg_filt, self.sgl_dur,
                            plots=plots, debug=debug)

        return None

    def _update_ann_analyses(self):
        import os

        from .config import paths, fnames

        ann_analyses = list()
        path_ann = _join(self.path_project, paths['ann'])
        analyses = sorted(os.listdir(path_ann))
        for p in analyses:
            if os.path.isdir(_join(path_ann, p)):
                ann_analyses.append(p)

        self.ann_analyses = ann_analyses

        return None

    def _update_ann_data(self):
        '''Load ANN output data from currently selected ANN analysis'''
        import os
        import pandas
        import yamlord

        from .config import paths, fnames

        path_output = _join(self.path_project, paths['ann'], self.current_ann)

        self.cfg_ann = yamlord.read_yaml(_join(path_output,
                                               fnames['cfg']['ann']))

        self.train = _rpickle(_join(path_output, fnames['ann']['train']))
        self.valid = _rpickle(_join(path_output, fnames['ann']['valid']))
        self.test = _rpickle(_join(path_output, fnames['ann']['test']))

        self.results_tune = _rpickle(_join(path_output, fnames['ann']['tune']))
        self.results_dataset = _rpickle(_join(path_output, fnames['ann']['dataset']))
        self.tune_cms = _rpickle(_join(path_output, fnames['ann']['cms_tune']))
        self.tune_cms = _rpickle(_join(path_output, fnames['ann']['cms_data']))

        file_post = _join(self.path_project, paths['ann'], self.current_ann,
                          fnames['ann']['post'])
        self.post = yamlord.read_yaml(file_post)

        return None

    def run_ann(self, plots=False, debug=False):
        '''Perfom ANN analysis

        Args
        ----
        plots: bool
            Plots loss, accuracy, and error during training
        debug: bool
            Runs a single configuration of ANN hyperparameters
        '''
        import datetime
        import yamlord

        from . import ann
        from .config import paths, fnames

        # Define output directory
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_ann = 'theanets_{}'.format(now)

        # Pre-process sub-glide data for ANN (compile, add `rho_mod`)
        sgls_all = ann.pre.process(self.path_project, self.cfg_project,
                                   self.cfg_ann)
        # Run the ANN analysis
        ann.ann.run(self.path_project, self.current_ann, self.cfg_project,
                    self.cfg_ann, sgls_all, plots=plots, debug=debug)

        # Reload `cfg_project` after ANN analyses additions in `ann.ann.run()`
        file_cfg_project = _join(self.path_project, fnames['cfg']['project'])
        self.cfg_project = yamlord.read_yaml(file_cfg_project)

        # Reload `cfg_ann` from analysis output directory, load output data
        file_cfg_ann = _join(self.path_project, paths['ann'], self.current_ann,
                             fnames['cfg']['ann'])
        self.cfg_ann = yamlord.read_yaml(file_cfg_ann)

        # Post process data
        self.post = ann.post.process(self.path_project, self.current_ann,
                                     self.cfg_ann)

        # Update ANN output directory list and load data
        self._update_ann_analyses()
        self._update_ann_data()

        return None

    def set_ann_analysis(self):
        '''Set the ANN analysis to work with'''
        import os
        import pyotelem

        self._update_ann_analyses()

        # Get user selection for ANN analysis to load
        print('\nAvailable ANN analyses selections:')
        print('')
        print('{:>3}. {}'.format('No', 'Path'))
        print('-----------------------------')
        for i, path in enumerate(self.ann_analyses):
            print('{:3.0f}. {}'.format(i, path))
        print('')
        idx = pyotelem.utils.recursive_input('ANN analysis number', int)

        self.current_ann = self.ann_analyses[idx]

        self._update_ann_data()

        return None
