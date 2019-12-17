from collections import OrderedDict
import os
import yamlord

import pandas
import datetime
import pyotelem.utils

from smartmove.config import paths, fnames
from smartmove import ann, visuals
from smartmove.glideid import glideid


class Analysis(object):
    """Smartmove helper class for running analyses and output to project path
    """

    def __init__(self, path_project: str, sgl_dur: int = 2):
        """Initiate the Analysis object

        Args
        ----
        path_project: str
            Path to project directory created with `smartmove.create_project()`
            method.
        sgl_dur: int
            Duration of sub-glide splits (seconds, Defaults to `2`)
        """

        self.path_project = os.path.abspath(path_project)
        self.sgl_dur = sgl_dur

        self.cfg_project = self._read_file(fnames["cfg"]["project"])
        self.cfg_ann = self._read_file(fnames["cfg"]["ann"])
        self.cfg_glide = self._read_file(fnames["cfg"]["glide"])
        self.cfg_filt = self._read_file(fnames["cfg"]["filt"])
        self.cfg_experiments = self._read_file(fnames["cfg"]["exp_bounds"])

    def run_glides(self, plots: bool = True, debug: bool = False):
        """Peform glide identification analysis

        Args
        ----
        plots: bool
            Switch for turning on plots (Default `True`). When activated plots for
            reviewing signal processing will be displayed.
        debug: bool
            Switch for turning on debugging (Default `False`). When activated values
            for `cutoff_freq` and `J` will be set to generic values and diagnostic
            plots of the `speed` parameter in `tag` will be displayed.
        """
        glideid.run(
            self.path_project,
            self.cfg_project,
            self.cfg_glide,
            self.cfg_filt,
            self.sgl_dur,
            plots=plots,
            debug=debug,
        )

    def run_ann(self, plots=False, debug=False):
        """Perfom ANN analysis

        Args
        ----
        plots: bool
            Plots loss, accuracy, and error during training
        debug: bool
            Runs a single configuration of ANN hyperparameters
        """

        # Define output directory
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_ann = "theanets_{}".format(now)

        # Pre-process sub-glide data for ANN (compile, add `rho_mod`)
        sgls_all = ann.pre.process(self.path_project, self.cfg_project, self.cfg_ann)

        # Run the ANN analysis
        ann.ann.run(
            self.path_project,
            self.current_ann,
            self.cfg_project,
            self.cfg_ann,
            sgls_all,
            plots=plots,
            debug=debug,
        )

        # Reload `cfg_project` after ANN analyses additions in `ann.ann.run()`
        self.cfg_project = self._read_file(fnames["cfg"]["project"])

        # Reload `cfg_ann` from analysis output directory, load output data
        fp_cfg_ann = os.path.join(
            self.path_project, paths["ann"], self.current_ann, fnames["cfg"]["ann"]
        )
        self.cfg_ann = yamlord.read_yaml(fp_cfg_ann)

        # Post process data
        self.post = ann.post.process(self.path_project, self.current_ann, self.cfg_ann)

        # Update ANN output directory list and load data
        self._update_ann_analyses()
        self._update_ann_data()

    def set_ann_analysis(self):
        """Set the ANN analysis to work with"""

        self._update_ann_analyses()

        # Get user selection for ANN analysis to load
        print("\nAvailable ANN analyses selections:")
        print("")
        print("{:>3}. {}".format("No", "Path"))
        print("-----------------------------")
        for i, path in enumerate(self.ann_analyses):
            print("{:3.0f}. {}".format(i, path))
        print("")
        idx = pyotelem.utils.recursive_input("ANN analysis number", int)

        self.current_ann = self.ann_analyses[idx]

        self._update_ann_data()

    def make_figures(self):
        """Generate figures used in Smartmove paper"""

        visuals.figures.make_all(self.path_project, self.current_ann)

    def make_tables(self):
        """Generate tables used in Smartmove paper

        Note
        ----
        The following tables require manual adjustments:
        * `table_ann_params`
        * `table_ann_target_descr`

        The table `table_ann_feature_descr` is created entirely manually.
        """

        visuals.tables.make_all(self.path_project, self.current_ann)

    def tex_compile(self, path_tex: str):
        """Helper function to compile all .tex in a path and convert to png"""

        path = os.path.join(self.path_project, path_tex)
        for p in os.listdir(path):
            pp = os.path.join(path, p)
            if pp.endswith(".tex"):
                fname = os.path.splitext(os.path.split(pp)[1])[0]
                visuals.latex.utils.compile_latex(path, fname, dpi=300)
                visuals.latex.utils.pdf_to_img(path, fname, in_ext="pdf", out_ext="png")

    def _read_file(self, fname: str) -> OrderedDict:
        return yamlord.read_yaml(os.path.join(self.path_project, fname))

    @staticmethod
    def _read_pickle(path_output: str, fname: str) -> pandas.DataFrame:
        return pandas.read_pickle(os.path.join(path_output, fname))

    def _update_ann_analyses(self):
        """Update list of available ANN analyses"""
        ann_analyses = list()
        path_ann = os.path.join(self.path_project, paths["ann"])
        analyses = sorted(os.listdir(path_ann))
        for p in analyses:
            if os.path.isdir(os.path.join(path_ann, p)):
                ann_analyses.append(p)

        self.ann_analyses = ann_analyses

    def _update_ann_data(self):
        """Load ANN output data from currently selected ANN analysis"""

        path_output = os.path.join(self.path_project, paths["ann"], self.current_ann)

        self.cfg_ann = yamlord.read_yaml(
            os.path.join(path_output, fnames["cfg"]["ann"])
        )

        self.train = Analysis._read_pickle(path_output, fnames["ann"]["train"])
        self.valid = Analysis._read_pickle(path_output, fnames["ann"]["valid"])
        self.test = Analysis._read_pickle(path_output, fnames["ann"]["test"])

        self.results_tune = Analysis._read_pickle(path_output, fnames["ann"]["tune"])
        self.results_dataset = Analysis._read_pickle(path_output, fnames["ann"]["dataset"])
        self.tune_cms = Analysis._read_pickle(path_output, fnames["ann"]["cms_tune"])
        self.tune_cms = Analysis._read_pickle(path_output, fnames["ann"]["cms_data"])

        fp_post = os.path.join(
            self.path_project, paths["ann"], self.current_ann, fnames["ann"]["post"]
        )
        self.post = yamlord.read_yaml(fp_post)
