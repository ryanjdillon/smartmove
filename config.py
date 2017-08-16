'''Paths and filenames used by `smartmove` modules. These are not to be changed'''

paths = {
    # Contains datalogger data for each experiment to be analyzed
    'tag': 'data_tag',
    # Contains `.csv` files for field and isotope experiments
    'csv': 'data_csv',
    # Contains data from CTD measurements
    'ctd': 'data_ctd',
    # Contains sub-glide data directories for each processed experiment
    'glide': 'data_glide',
    # Path for ANN model input (copied) and output files
    'ann': 'model_ann',
    }

fnames = {
    # Configuration YAML files
    'cfg': {
        'project':    'cfg_project.yml',
        'exp_bounds': 'cfg_experiments.yml',
        'glide':      'cfg_glide.yml',
        'ann':        'cfg_ann.yml',
        'filt':       'cfg_subglide-filter.yml',
        },
    # Datalogger data processing
    'tag': {
        'cal':  'cal.yml',
        'data': 'pydata_{}.p',
        },
    # CSV input filenames
    'csv': {
        'cal_prop': 'propeller_calibrations.csv',
        'field':    'field_experiments.csv',
        'isotope':  'isotope_experiments.csv',
        },
    # Glide analysis input/output
    'glide': {
        'dives':           'data_dives.p',
        'sgls':            'data_sgls.p',
        'glide_ratio' :    'data_ratios.p',
        'mask_tag':        'mask_tag.p',
        'mask_tag_glides': 'mask_tag_glides.p',
        'mask_tag_sgls':   'mask_tag_sgls.p',
        'mask_tag_filt':   'mask_tag_filt.p',
        'mask_sgls_filt':  'mask_sgls_filt.p',
        },
    # ANN configuration and output filenames
    'ann': {
        'sgls':           'data_sgls.p',
        'sgls_norm':      'data_sgls_norm.p',
        'train':          'data_train.p',
        'valid':          'data_valid.p',
        'test':           'data_test.p',
        'cms_tune':       'cms_tune.p',
        'cms_data':       'cms_data.p',
        'tune':           'results_tuning.p',
        'dataset':        'results_dataset_size.p',
        'post':           'postprocessing.yml',
        'stats_features': 'stats_input_features.p',
        'stats_targets':  'stats_target_values.p',
        },
    }
