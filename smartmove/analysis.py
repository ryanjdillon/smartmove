class Analysis(object):

    def __init__(self, path_project):
        import os
        import yamlord

        self.path_project = path_project

        fname_cfg_project = 'cfg_project.yaml'
        fname_cfg_project = 'cfg_ann.yaml'
        path_cfg_project = os.path.join(path_project, fname_cfg_project)
        path_cfg_ann = os.path.join(path_project, fname_cfg_ann)

        # Check that the project environment has been created
        if not os.path.isfile(path_cfg_project):
            msg = ('The configuration file `{}` was not found. Please be '
                   'refer to the documentation on how to setup your project '
                   'directory'.format(fname_cfg_project))
            raise SystemError(msg)

        self.cfg_project = yamlord.read_yaml(path_cfg_project)
        self.cfg_ann = yamlord.read_yaml(path_cfg_ann)

        return None

    def process_glides(self):
        return None

    def run_ann(self):
        from ann import ann
        cfg, data, results = ann.run(self.cfg_project, self.cfg_ann)

        self.train = data[0]
        self.valid = data[1]
        self.test  = data[2]
        self.bins  = data[3]

        self.results_tune    = results[0]
        self.results_dataset = results[1]
        self.tune_cms        = results[2]
        self.data_cms        = results[3]

        return None
