"""
This module has functions for post-processing ANN results and preparing a
summary of them.
"""
from os.path import join as _join


def time_prediction(net, features):
    """
    `timeit.timeit` returns the total time in seconds (float) to run the test,
    not including the time used for setup. The time for each execution is the
    total time divided by the number of executions `number`. The default value
    for `number` is 1e6, but set here to `100000`.
    """
    from timeit import timeit as tt

    n = 100000
    return tt("net.predict(features)", number=n, globals=locals()) / n


def calculate_precision(cm):
    """Calculate the precision for each class in a confusion matrix"""
    import numpy

    n_classes = len(cm[0])
    precision = numpy.zeros(n_classes)
    for i in range(n_classes):
        precision[i] = cm[i, i] / sum(cm[i, :])

    return precision


def process(path_project, path_analysis, cfg_ann):
    from collections import OrderedDict
    import numpy
    import os
    import pandas
    import pyotelem
    import yamlord

    from . import pre
    from ..config import paths, fnames

    print(path_analysis)
    path_output = _join(path_project, paths["ann"], path_analysis)

    file_field = _join(path_project, paths["csv"], fnames["csv"]["field"])
    file_isotope = _join(path_project, paths["csv"], fnames["csv"]["isotope"])
    field, isotope = pre.add_rhomod(file_field, file_isotope)

    # EXPERIMENT INPUT
    post = OrderedDict()
    post["input_exp"] = OrderedDict()

    # n experiments and animals
    post["n_field"] = len(field)
    post["n_animals"] = len(field["animal"].unique())

    # Min max values of rho_mod and % lipid for each seal
    post["exp"] = OrderedDict()
    post["iso"] = OrderedDict()
    for a in numpy.unique(field["animal"]):
        # Field experiment values
        post["exp"][a] = OrderedDict()
        mask = field["animal"] == a
        post["exp"][a]["min_rhomod"] = field[mask]["rho_mod"].min()
        post["exp"][a]["max_rhomod"] = field[mask]["rho_mod"].max()

        # Isotope experiment values
        post["iso"][a] = OrderedDict()
        mask = isotope["animal"] == a.capitalize()
        post["iso"][a]["min_mass"] = isotope[mask]["mass_kg"].min()
        post["iso"][a]["max_mass"] = isotope[mask]["mass_kg"].max()

    # ANN CONFIG
    results = pandas.read_pickle(_join(path_output, fnames["ann"]["tune"]))

    post["ann"] = OrderedDict()

    # Number of network configurations
    post["ann"]["n_configs"] = len(results)

    # Load training data
    file_train = _join(path_output, "data_train.p")
    file_valid = _join(path_output, "data_valid.p")
    file_test = _join(path_output, "data_test.p")
    train = pandas.read_pickle(file_train)
    valid = pandas.read_pickle(file_valid)
    test = pandas.read_pickle(file_test)

    # Number of samples compiled, train, valid, test
    post["ann"]["n"] = OrderedDict()
    post["ann"]["n"]["train"] = len(train[0])
    post["ann"]["n"]["valid"] = len(valid[0])
    post["ann"]["n"]["test"] = len(test[0])
    post["ann"]["n"]["all"] = len(train[0]) + len(valid[0]) + len(test[0])

    # percentage of compiled dataset in train, valid, test
    post["ann"]["n"]["perc_train"] = len(train[0]) / post["ann"]["n"]["all"]
    post["ann"]["n"]["perc_valid"] = len(valid[0]) / post["ann"]["n"]["all"]
    post["ann"]["n"]["perc_test"] = len(test[0]) / post["ann"]["n"]["all"]

    # Total tuning time
    post["ann"]["total_train_time"] = results["train_time"].sum()

    # POSTPROCESS VALUES
    # Best/worst classification accuracies
    mask_best = results["accuracy"] == results["accuracy"].max()
    best_idx = results["train_time"][mask_best].idxmin()

    mask_worst = results["accuracy"] == results["accuracy"].min()
    worst_idx = results["train_time"][mask_worst].idxmax()

    post["ann"]["best_idx"] = best_idx
    post["ann"]["worst_idx"] = worst_idx

    # Get min/max accuracy and training time for all configurations
    post["ann"]["metrics"] = OrderedDict()
    for key in ["accuracy", "train_time"]:
        post["ann"]["metrics"][key] = OrderedDict()
        post["ann"]["metrics"][key]["max_idx"] = results[key].argmax()
        post["ann"]["metrics"][key]["min_idx"] = results[key].argmin()

        post["ann"]["metrics"][key]["max"] = results[key].max()
        post["ann"]["metrics"][key]["min"] = results[key].min()

        post["ann"]["metrics"][key]["best"] = results[key][best_idx]
        post["ann"]["metrics"][key]["worst"] = results[key][worst_idx]

    # Optimal network results
    post["ann"]["opt"] = OrderedDict()

    net = results["net"][best_idx]

    # Loop 10 times taking mean prediction time
    # Each loop, 100k iterations of timing
    file_test = _join(path_output, fnames["ann"]["test"])
    test = pandas.read_pickle(file_test)
    features = numpy.expand_dims(test[0][0], axis=0)
    t_pred = time_prediction(net, features)
    post["ann"]["opt"]["t_pred"] = t_pred

    # Filesize of trained NN
    file_net_best = "./net.tmp"
    pandas.to_pickle(net, file_net_best)
    st = os.stat(file_net_best)
    os.remove(file_net_best)
    post["ann"]["opt"]["trained_size"] = st.st_size / 1000  # kB

    # %step between subsets of test for dataset size test
    post["ann"]["dataset"] = "numpy.arange(0,1,0.03))[1:]"

    # Tune confusion matrices (cms) from most optimal configuration
    # one field per dataset `train`, `valid`, and `test`
    # first level `targets` if for all datasets
    post["ann"]["bins"] = OrderedDict()
    file_tune_cms = _join(path_output, fnames["ann"]["cms_tune"])

    tune_cms = pandas.read_pickle(file_tune_cms)
    bins = tune_cms["targets"]

    # Range of each bin, density, lipid percent
    bin_range = range(len(bins) - 1)

    rho_lo = numpy.array([bins[i] for i in bin_range])
    rho_hi = numpy.array([bins[i + 1] for i in bin_range])
    # Note density is converted from kg/m^3 to g/cm^3 for `dens2lip`
    lip_lo = pyotelem.physio_seal.dens2lip(rho_lo * 0.001)
    lip_hi = pyotelem.physio_seal.dens2lip(rho_hi * 0.001)

    # Generate bin ranges as strings
    fmt_bin = r"{:7.2f} <= rho_mod < {:7.2f}"
    fmt_lip = r"{:6.2f} >= lipid % > {:6.2f}"
    str_bin = [fmt_bin.format(lo, hi) for lo, hi in zip(rho_lo, rho_hi)]
    str_lip = [fmt_lip.format(lo, hi) for lo, hi in zip(lip_lo, lip_hi)]

    post["ann"]["bins"]["values"] = list(bins)
    post["ann"]["bins"]["value_range"] = str_bin
    post["ann"]["bins"]["value_diff"] = list(numpy.diff(bins))

    # Note density is converted from kg/m^3 to g/cm^3 for `dens2lip`
    lipid_perc = pyotelem.physio_seal.dens2lip(bins * 0.001)
    post["ann"]["bins"]["lipid_perc"] = list(lipid_perc)
    post["ann"]["bins"]["lipid_range"] = str_lip
    post["ann"]["bins"]["lipid_diff"] = list(numpy.diff(lipid_perc))

    precision = calculate_precision(tune_cms["validation"]["cm"])
    post["ann"]["bins"]["precision"] = [None] * len(bins)
    targets = tune_cms["validation"]["targets"]
    for i in range(len(bins)):
        if bins[i] in targets:
            post["ann"]["bins"]["precision"][i] = precision[bins[i] == targets]
        else:
            post["ann"]["bins"]["precision"][i] = "None"

    # Save post processing results as YAML
    file_post = _join(path_output, fnames["ann"]["post"])
    yamlord.write_yaml(post, file_post)

    return post
