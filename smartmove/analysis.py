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

        self.path_project = os.path.abspath(path_project)

        file_cfg_project = _join(path_project, 'cfg_project.yml')

        # Check that the project environment has been created
        if not os.path.isfile(file_cfg_project):
            msg = ('The configuration file `{}` was not found. Please be '
                   'refer to the documentation on how to setup your project '
                   'directory'.format(fname_cfg_project))
            raise SystemError(msg)

        self.cfg_project = yamlord.read_yaml(file_cfg_project)

        fnames = self.cfg_project['fnames']
        file_cfg_ann = _join(path_project, fnames['ann']['cfg_ann'])
        file_cfg_filt = _join(path_project, fnames['glide']['cfg_filt'])
        file_cfg_experiments = _join(path_project, fnames['tag']['cfg_exp'])

        self.cfg_ann = yamlord.read_yaml(file_cfg_ann)
        self.cfg_filt = yamlord.read_yaml(file_cfg_ann)
        self.cfg_experiments = yamlord.read_yaml(file_cfg_experiments)

        self.sgl_dur = sgl_dur

        return None

    def set_sgl_duration(self, sgl_dur):
        '''Set the duration to be used for splitting sub-glides

        Args
        ----
        sgl_dur: int
            Duration of sub-glide splits (seconds)
        '''
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

        glideid.glideid.run(self.cfg_project, self.cfg_glide, self.cfg_filt,
                            self.sgl_dur, plots=plots, debug=debug)
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
        import yamlord

        from . import ann

        paths = self.cfg_project['paths']
        fnames = self.cfg_project['fnames']

        # Run the ANN analysis
        ann.ann.run(self.cfg_project, self.cfg_ann, plots=plots, debug=debug)

        # Reload `cfg_project` after ANN analyses additions in `ann.ann.run()`
        file_cfg_project = _join(paths['project'], 'cfg_project.yml')
        self.cfg_project = yamlord.read_yaml(file_cfg_project)

        # Set the current ANN analysis path name
        self.current_ann = self.cfg_project['ann_analyses'][-1]

        # Reload `cfg_ann` from analysis output directory, load output data
        file_cfg_ann = _join(paths['project'], paths['ann'], self.current_ann,
                             fnames['ann']['cfg_ann'])
        self.cfg_ann = yamlord.read_yaml(file_cfg_ann)

        # Post process data
        self.post = ann.post.process(self.cfg_project, self.cfg_ann)

        # Update ANN output data
        self.update_ann()

        return None

    def update_ann(self):
        '''Load ANN output data from currently selected ANN analysis'''
        import os
        import pandas
        import yamlord

        paths = self.cfg_project['paths']
        fnames = self.cfg_project['fnames']

        path_output = _join(paths['project'], paths['ann'], self.current_ann)

        self.train = _rpickle(_join(path_output, fnames['ann']['train']))
        self.valid = _rpickle(_join(path_output, fnames['ann']['valid']))
        self.test = _rpickle(_join(path_output, fnames['ann']['test']))

        self.results_tune = _rpickle(_join(path_output, fnames['ann']['tune']))
        self.results_dataset = _rpickle(_join(path_output, fnames['ann']['dataset']))
        self.tune_cms = _rpickle(_join(path_output, fnames['ann']['cms_tune']))
        self.tune_cms = _rpickle(_join(path_output, fnames['ann']['cms_data']))

        file_post = _join(paths['project'], paths['ann'], self.current_ann,
                          fnames['ann']['post'])
        self.post = yamlord.read_yaml(file_post)

        return None

    def set_ann_analysis(self):
        '''Set the ANN analysis to work with'''
        import pyotelem

        paths = self.cfg_project['paths']
        fnames = self.cfg_project['fnames']

        path_ann = _join(paths['project'], paths['ann'])

        # Get user selection for ANN analysis to load
        print('\nANN analyses selections:')
        i = 0
        for path in os.listdir(path_ann):
            if os.path.isdir(path):
                print('{:3.0f}. {}'.format(i, path))
        print('')
        idx = pyotelem.utils.recursive_input('ANN model number', int)

        self.current_ann = self.cfg_project['ann_analyses'][idx]

        self.update_ann()

        return None
