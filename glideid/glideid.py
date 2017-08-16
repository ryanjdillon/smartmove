'''
Functions for performing glide identification on data collected from tags (i.e.
data loggers)

This code is an adaptation of the script written by Lucia Martina Martin Lopez,
presented at a body density estimation workshop at the University of Tokyo,
Japan in May, 2016.
'''

def run(path_project, cfg_project, cfg_glide, cfg_filt, sgl_dur, plots=True, debug=False):
    '''Run glide identification on data in configuration paths

    Args
    ----
    path_project:
        Parent path for project
    cfg_project: OrderedDict
        Dictionary of configuration parameters for the current project
    cfg_glide: OrderedDict
        Dictionary of configuration parameters for glide identification
    cfg_filt: OrderedDict
        Dictionary of configuration parameters for filtering sub-glides
    sgl_dur: int
        Duration of sub-glide splits (seconds)
    plots: bool
        Switch for turning on plots (Default `True`). When activated plots for
        reviewing signal processing will be displayed.
    debug: bool
        Switch for turning on debugging (Default `False`). When activated values
        for `cutoff_freq` and `J` will be set to generic values and diagnostic
        plots of the `speed` parameter in `tag` will be displayed.

    Attributes
    ----------
    cutoff_frq: float
        Cutoff frequency for separating low and high frequency signals
    stroke_frq: float
        Frequency at which maximum power is seen in accelerometer PSD
    J:
        Frequency of stroke signal in accelerometer data (m/s2)
    t_max: int
        Maximum duration allowable for a fluke stroke in seconds, it can be set as
        1/`stroke_frq`
    J:
        Magnitude threshold for detecting a fluke stroke in [m/s2]

    Returns
    -------
    tag: pandas.DataFrame
        Data loaded from tag with associated sensors
    dives: pandas.DataFrame
        Start and stop indices and attributes for dive events in `tag` data,
        including: start_idx, stop_idx, dive_dur, depths_max, depths_max_idx,
        depths_mean, compr_mean.
    GL: ndarray, (n, 2)
        Start and stop indices and attributes of glide events in `tag` data,
    sgls: pandas.DataFrame
        Contains sub-glide summary information of `tag` data
    '''
    from collections import OrderedDict
    import numpy
    import os
    from os.path import join as _join
    import pandas
    import pyotelem
    from pyotelem.plots import plotdynamics, plotglides
    import yamlord

    from ..config import paths, fnames
    from .. import utils
    from . import utils_lleo

    # Input filenames
    fname_cal = fnames['tag']['cal']
    fname_cal_prop = fnames['csv']['cal_prop']

    # Output filenames
    fname_cfg_glide = fnames['cfg']['glide']
    fname_cfg_filt  = fnames['cfg']['filt']

    fname_dives           = fnames['glide']['dives']
    fname_glide_ratio     = fnames['glide']['glide_ratio']
    fname_mask_tag        = fnames['glide']['mask_tag']
    fname_mask_tag_glides = fnames['glide']['mask_tag_glides']
    fname_sgls            = fnames['glide']['sgls']
    fname_mask_tag_sgls   = fnames['glide']['mask_tag_sgls']
    fname_mask_tag_filt   = fnames['glide']['mask_tag_filt']
    fname_mask_sgls_filt  = fnames['glide']['mask_sgls_filt']

    # Fields to ignore when concatenating output path names
    ignore = ['nperseg', 'peak_thresh', 'alpha', 'min_depth', 't_max',
              'last_modified']

    # Generate list of paths in tag data directory
    path_exps = list()
    for path_exp in os.listdir(_join(path_project, paths['tag'])):

        # Only process directories
        if os.path.isdir(_join(path_project, paths['tag'], path_exp)):

            path_exps.append(path_exp)

    # Get user selection of tag data paths to process
    path_exps = sorted(path_exps)
    msg = 'paths numbers to process:\n'
    process_ind = pyotelem.utils.get_dir_indices(msg, path_exps)

    # Process selected tag experiments
    for i in process_ind:
        path_exp = path_exps[i]
        fname_tag = fnames['tag']['data'].format(path_exp)

        # Get correct calibration path given tag ID number
        tag_model = path_exp.replace('-','').split('_')[1].lower()
        tag_id = int(path_exp.split('_')[2])
        year   = int(path_exp[:4])
        month  = int(path_exp[4:6])
        path_cal_acc = cfg_project['cal'][tag_model][tag_id][year][month]

        print('Tag calibration file path: {}\n'.format(path_cal_acc))

        # Currently creating a new configuration for each exp
        path_cfg_glide = path_exp

        print('Processing: {}\n'.format(path_exp))

        # Run glide analysis

        # Output paths
        out_data  = _join(path_project, paths['tag'], path_exp)
        os.makedirs(out_data, exist_ok=True)

        # LOAD DATA
        #----------
        # linearly interpolated tag to accelerometer sensor
        path_data_tag = _join(path_project, paths['tag'], path_exp)
        file_cal_acc  = _join(path_project, paths['tag'],
                                     path_cal_acc, fname_cal)
        file_cal_prop = _join(path_project, paths['csv'],
                                     fname_cal_prop)

        tag, dt_a, fs_a = utils_lleo.load_lleo(path_data_tag, file_cal_acc,
                                               file_cal_prop)

        # Plot speed if debug is on
        if debug:
            exp_ind = range(len(tag))
            plotdynamics.plot_swim_speed(exp_ind, tag['speed'].values)


        # Signal process data, calculate derived data and find stroke frequencies
        cfg_glide_exp, tag, dives, masks, exp_ind = _process_tag_data(path_project,
                                                                      cfg_project,
                                                                      cfg_glide,
                                                                      path_exp,
                                                                      tag,
                                                                      fs_a,
                                                                      plots=plots,
                                                                      debug=debug)
        # Save data
        tag.to_pickle(_join(out_data, fname_tag))
        dives.to_pickle(_join(out_data, fname_dives))
        masks.to_pickle(_join(out_data, fname_mask_tag))


        # Find Glides
        #------------
        GL, masks = _process_glides(cfg_glide_exp, tag, fs_a, dives, masks,
                                    plots=plots, debug=debug)

        # Create output path from concatenating parameters in `cfg_glide_exp`
        dname_glide = utils.cat_path(cfg_glide_exp, ignore)
        out_glide = _join(path_project, paths['glide'], path_exp,
                          dname_glide)
        os.makedirs(out_glide, exist_ok=True)

        # Save glide data to concatenated path
        masks['glides'].to_pickle(_join(out_glide, fname_mask_tag_glides))

        # Save glide analysis configuration
        cfg_glide_exp['last_modified'] = _now_str()
        file_cfg_glide_exp = _join(out_glide, fname_cfg_glide)
        yamlord.write_yaml(cfg_glide_exp, file_cfg_glide_exp)


        # SPLIT GLIDES TO SUB-GLIDES
        #--------------------------
        # Split into sub-glides, generate summary tables
        sgls, masks['sgls'] = _process_sgls(tag, fs_a, dives, GL, sgl_dur)

        # Create output path from passed `sgls` duration
        out_sgls = _join(out_glide, 'dur_{}'.format(sgl_dur))
        os.makedirs(out_sgls, exist_ok=True)

        # Save sgls data to path for passed `sgls` duration
        sgls.to_pickle(_join(out_sgls, fname_sgls))
        masks['sgls'].to_pickle(_join(out_sgls, fname_mask_tag_sgls))

        # FILTER AND PLOT SUB-GLIDES
        #--------------------------
        # Get masks of `tag` and `sgls` data for sgls matching constraints
        exp_ind = numpy.where(masks['exp'])[0]
        mask_tag_filt, mask_sgls_filt = utils.filter_sgls(len(tag),
                                                          exp_ind,
                                                          sgls,
                                                          cfg_filt['pitch_thresh'],
                                                          cfg_filt['min_depth'],
                                                          cfg_filt['max_depth_delta'],
                                                          cfg_filt['min_speed'],
                                                          cfg_filt['max_speed'],
                                                          cfg_filt['max_speed_delta'])

        # Plot filtered sgls
        plotglides.plot_sgls(tag['depth'].values, mask_tag_filt, sgls,
                             mask_sgls_filt, tag['p_lf'].values,
                             tag['r_lf'].values, tag['h_lf'].values)

        # Create output path from concatenating parameters in `cfg_filt`
        dname_filt = utils.cat_path(cfg_filt, ignore)
        out_filt = _join(out_sgls, dname_filt)
        os.makedirs(out_filt, exist_ok=True)

        # Save filtered sgls data to concatenated path
        pandas.to_pickle(mask_tag_filt, _join(out_filt, fname_mask_tag_filt))
        pandas.to_pickle(mask_sgls_filt, _join(out_filt, fname_mask_sgls_filt))

        # Save symlink to data and masks in filter directory
        out_paths = [out_data, out_data, out_glide, out_glide, out_sgls,
                     out_sgls, out_sgls]
        sym_fnames = [fname_tag, fname_mask_tag, fname_cfg_glide,
                      fname_mask_tag_glides, fname_cfg_filt,
                      fname_mask_tag_sgls, fname_sgls]
        for out_path, fname in zip(out_paths, sym_fnames):
            rel_path = os.path.relpath(_join(out_path, fname), out_filt)
            utils.symlink(rel_path, _join(out_filt, fname))

        # Save sub-glide analysis configuration
        cfg_filt['last_modified'] = _now_str()
        file_cfg_filt = _join(out_sgls, fname_cfg_filt)
        yamlord.write_yaml(cfg_filt, file_cfg_filt)

    return tag, dives, GL, sgls


def _process_tag_data(path_project, cfg_project, cfg_glide, path_exp, tag,
        fs_a, plots=True, debug=False):
    '''Calculate body conditions summary statistics

    Args
    ----
    path_project:
        Parent path for project
    cfg_project: OrderedDict
        Dictionary of configuration parameters for the current project
    cfg_glide: OrderedDict
        Dictionary of configuration parameters for glide identification
    path_exp: str
        Directory name of `tag` data being processed
    tag: pandas.DataFrame
        Data loaded from tag with associated sensors
    fs_a: float
        Sampling frequency (i.e. number of samples per second)
    plots: bool
        Switch for turning on plots (Default `True`). When activated plots for
        reviewing signal processing will be displayed.
    debug: bool
        Switch for turning on debugging (Default `False`). When activated values
        for `cutoff_freq` and `J` will be set to generic values and diagnostic
        plots of the `speed` parameter in `tag` will be displayed.

    Returns
    -------
    cfg: OrderedDict
    tag: pandas.DataFrame
        Data loaded from tag with associated sensors, with added fields from
        signal processing
    dives: pandas.DataFrame
        Start and stop indices and attributes for dive events in `tag` data,
        including: start_idx, stop_idx, dive_dur, depths_max, depths_max_idx,
        depths_mean, compr_mean.
    masks: pandas.DataFrame
        Boolean masks for slicing identified dives, glides, and sub-glides from
        the `tag` dataframe.
    exp_ind: OrderedDict
        Start and stop indices of `tag` data to be analyzed
    '''
    from collections import OrderedDict
    import numpy
    from os.path import join as _join
    import pandas
    import pyotelem
    from pyotelem.plots import plotdives, plotdsp
    import yamlord
    import copy

    from .. import utils
    from . import utils_ctd
    from ..config import paths, fnames

    exp_idxs = [None, None]

    file_cfg_exp  = _join(path_project, fnames['cfg']['exp_bounds'])
    cfg = copy.deepcopy(cfg_glide)

    try:
        cfg_exp = yamlord.read_yaml(file_cfg_exp)
    except:
        cfg_exp = OrderedDict()

    # 1 Select indices for analysis
    #--------------------------------------------------------------------------
    print('* Select indices for analysis\n')

    if path_exp in cfg_exp:
        exp_idxs[0] = cfg_exp[path_exp]['start_idx']
        exp_idxs[1] = cfg_exp[path_exp]['stop_idx']
    else:
        # Plot accelerometer axes, depths, and propeller speed
        plotdives.plot_triaxial_depths_speed(tag)

        # Get indices user input - mask
        exp_idxs[0] = pyotelem.utils.recursive_input('Analysis start index', int)
        exp_idxs[1] = pyotelem.utils.recursive_input('Analysis stop index', int)

        cfg_exp[path_exp] = OrderedDict()
        cfg_exp[path_exp]['start_idx'] = exp_idxs[0]
        cfg_exp[path_exp]['stop_idx']  = exp_idxs[1]
        yamlord.write_yaml(cfg_exp, file_cfg_exp)

    # Create dataframe for storing masks for various views of the data
    masks = pandas.DataFrame(index=range(len(tag)), dtype=bool)

    # Create mask of values to be considered part of the analysis
    masks['exp'] = False
    masks['exp'][exp_idxs[0]:exp_idxs[1]] = True

    # Create indices array `exp_ind` for analysis
    exp_ind = numpy.where(masks['exp'])[0]


    # 1.3 Calculate pitch, roll, and heading
    #--------------------------------------------------------------------------
    print('* Calculate pitch, roll, heading\n')
    tag['p'], tag['r'], tag['h'] = pyotelem.dynamics.prh(tag['Ax_g'].values,
                                                         tag['Ay_g'].values,
                                                         tag['Az_g'].values)

    # 2 Define dives
    #--------------------------------------------------------------------------
    print('* Define dives\n')
    dives, masks['dive'] = pyotelem.dives.finddives2(tag['depth'].values,
                                                     cfg_glide['min_depth'])


    # 3.2.1 Determine `stroke_frq` fluking rate and cut-off frequency
    #--------------------------------------------------------------------------
    print('* Get stroke frequency\n')
    # calculate power spectrum of the accelerometer data at the whale frame
    Ax_g = tag['Ax_g'][masks['exp']].values
    Az_g = tag['Az_g'][masks['exp']].values

    # NOTE change `stroke_ratio` here to modify selectio method
    # should be OK other than t_max, these values are too high
    if debug is False:
        cutoff_frq, stroke_frq, stroke_ratio = pyotelem.glides.get_stroke_freq(Ax_g,
                                                       Az_g,
                                                       fs_a,
                                                       cfg_glide['nperseg'],
                                                       cfg_glide['peak_thresh'],
                                                       stroke_ratio=None)
        # Store user input cutoff and stroke frequencies
        cfg['cutoff_frq']   = cutoff_frq
        cfg['stroke_frq']   = stroke_frq
        cfg['stroke_ratio'] = stroke_ratio

        # Calculate maximum duration of glides from stroke frequency
        cfg['t_max']  = 1 /cfg['stroke_frq']  # seconds
    else:
        cutoff_frq = 0.3
        cfg['cutoff_frq'] = cutoff_frq


    # 3.2.2 Separate low and high frequency signals
    #--------------------------------------------------------------------------
    print('* Separate accelerometry to high and low-pass signals\n')
    order = 5
    cutoff_str = str(cfg['cutoff_frq'])
    for btype, suffix in zip(['low', 'high'], ['lf', 'hf']):
        b, a, = pyotelem.dsp.butter_filter(cfg['cutoff_frq'], fs_a, order=order,
                btype=btype)
        for param in ['Ax_g', 'Ay_g', 'Az_g']:
            key = '{}_{}_{}'.format(param, suffix, cutoff_str)
            tag[key] = pyotelem.dsp.butter_apply(b, a, tag[param].values)

    # Plot low and high frequency accelerometer signals
    if plots is True:
        plotdsp.plot_lf_hf(tag['Ax_g'][masks['exp']],
                           tag['Ax_g_lf_'+cutoff_str][masks['exp']],
                           tag['Ax_g_hf_'+cutoff_str][masks['exp']],
                           title='x axis')

        plotdsp.plot_lf_hf(tag['Ay_g'][masks['exp']],
                           tag['Ay_g_lf_'+cutoff_str][masks['exp']],
                           tag['Ay_g_hf_'+cutoff_str][masks['exp']],
                           title='y axis')

        plotdsp.plot_lf_hf(tag['Az_g'][masks['exp']],
                           tag['Az_g_lf_'+cutoff_str][masks['exp']],
                           tag['Az_g_hf_'+cutoff_str][masks['exp']],
                           title='z axis')


    # 3.2.3 Calculate the smooth pitch from the low pass filter acceleration
    #       signal to avoid incorporating signals above the stroking periods
    #--------------------------------------------------------------------------
    print('* Calculate low-pass pitch, roll, heading\n')
    prh_lf = pyotelem.dynamics.prh(tag['Ax_g_lf_'+cutoff_str].values,
                                   tag['Ay_g_lf_'+cutoff_str].values,
                                   tag['Az_g_lf_'+cutoff_str].values,)

    tag['p_lf'], tag['r_lf'], tag['h_lf'] = prh_lf


    # 4 Define precise descent and ascent phases
    #--------------------------------------------------------------------------
    print('* Get precise indices of descents, ascents, phase and bottom\n')
    masks['des'], masks['asc'] = pyotelem.dives.get_des_asc2(tag['depth'].values,
                                                             masks['dive'].values,
                                                             tag['p_lf'].values,
                                                             cfg['cutoff_frq'],
                                                             fs_a,
                                                             order=5)
    # Typecast `des` and `asc` columns to `bool`
    masks = masks.astype(bool)
    if plots is True:
        plotdives.plot_dives_pitch(tag['depth'][masks['exp']],
                                   masks['dive'][masks['exp']],
                                   masks['des'][masks['exp']],
                                   masks['asc'][masks['exp']],
                                   tag['p'][masks['exp']],
                                   tag['p_lf'][masks['exp']])


    # 8 Estimate seawater density around the tagged animal
    #--------------------------------------------------------------------------
    print('* Estimate seawater density\n')

    # Study location and max depth to average salinities
    lon = cfg_project['experiment']['coords']['lon']
    lat = cfg_project['experiment']['coords']['lat']
    lat = cfg_project['experiment']['coords']['lat']
    max_depth = cfg_project['experiment']['net_depth']

    # Read data
    fname_ctd = cfg_project['experiment']['fname_ctd']
    file_ctd_mat = _join(path_project, paths['ctd'], fname_ctd)

    t = tag['temperature'].values

    tag['dsw'] = utils_ctd.get_seawater_densities(file_ctd_mat, t, lon, lat,
                                                  max_depth)

    # 6.1 Extract strokes and glides using heave
    #     high-pass filtered (HPF) acceleration signal, axis=3
    #--------------------------------------------------------------------------
    # Two methods for estimating stroke frequency `stroke_frq`:
    # * from the body rotations (pry) using the magnetometer method
    # * from the dorso-ventral axis of the HPF acceleration signal.

    # For both methods, t_max and J need to be determined.

    # Choose a value for J based on a plot showing distribution of signals:
    #   hpf-x, when detecting glides in the next step use Ahf_Anlf() with axis=0
    #   hpf-z when detecting glides in the next step use Ahf_Anlf() with axis=2

    print('* Get fluke signal threshold\n')

    if debug is False:
        # Plot PSD for J selection
        Ax_g_hf = tag['Ax_g_hf_'+cutoff_str][masks['exp']].values
        Az_g_hf = tag['Az_g_hf_'+cutoff_str][masks['exp']].values

        f_wx, Sx, Px, dpx = pyotelem.dsp.calc_PSD_welch(Ax_g_hf, fs_a, nperseg=512)
        f_wz, Sz, Pz, dpz = pyotelem.dsp.calc_PSD_welch(Az_g_hf, fs_a, nperseg=512)

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(f_wx, Sx, label='hf-x PSD')
        ax1.plot(f_wz, Sz, label='hf-z PSD')
        ax1.legend(loc='upper right')
        ax2.plot(tag['datetimes'][masks['exp']], Ax_g_hf, label='hf-x')
        ax2.plot(tag['datetimes'][masks['exp']], Az_g_hf, label='hf-z')
        ax2.legend(loc='upper right')

        fig.autofmt_xdate()
        plt.show()

        # Get user selection for J - select one for both axes
        cfg['J'] = pyotelem.utils.recursive_input('J (fluke magnitude)', float)
    else:
        cfg['J'] = 0.4

    return cfg, tag, dives, masks, exp_ind


def _process_glides(cfg_glide, tag, fs_a, dives, masks, plots=True,
        debug=False):
    import numpy

    from .. import utils
    import pyotelem.glides

    cutoff_str = str(cfg_glide['cutoff_frq'])

    # Get GL from dorso-ventral axis of the HPF acc signal
    GL = pyotelem.glides.get_stroke_glide_indices(tag['Az_g_hf_'+cutoff_str].values,
                                                  fs_a,
                                                  cfg_glide['J'],
                                                  cfg_glide['t_max'])

    # check glides duration and +/- zero crossings from selected J and t_max
    masks['glides'] = utils.mask_from_noncontiguous_indices(len(tag), GL[:,0], GL[:,1])

    return GL, masks


def _process_sgls(tag, fs_a, dives, GL, sgl_dur):
    '''Split sub-glides and generate summary dataframe'''
    import numpy

    import pyotelem.glides

    # 7 Make 5sec sub-glides
    #--------------------------------------------------------------------------
    print('* Make sub-glides, duration {}\n'.format(sgl_dur))

    SGL, data_sgl_mask = pyotelem.glides.split_glides(len(tag),
                                                      sgl_dur,
                                                      fs_a,
                                                      GL)


    # 9 Generate summary information table for sub-glides
    #--------------------------------------------------------------------------
    pitch_lf_deg = numpy.rad2deg(tag['p_lf'].values)

    print('* Generate summary information table for sub-glides\n')
    sgls = pyotelem.glides.calc_glide_des_asc(tag['depth'].values,
                                              tag['p_lf'].values,
                                              tag['r_lf'].values,
                                              tag['h_lf'].values,
                                              tag['speed'].values,
                                              dives,
                                              SGL,
                                              pitch_lf_deg,
                                              tag['temperature'].values,
                                              tag['dsw'].values)
    return sgls, data_sgl_mask


def _now_str():
    '''Create POSIX formatted datetime string of current time'''
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')


def _save_config(cfg_add, file_cfg_yaml, name='parameters'):
    '''Load analysis configuration defaults'''
    from collections import OrderedDict
    import datetime

    import yamlord
    from .. import utils

    cfg = OrderedDict()

    # Record the last date modified & git version
    cfg['last_modified'] = _now_str()

    # Get git hash and versions of dependencies
    cfg['versions'] = utils.get_versions('smartmove')

    cfg[name] = cfg_add

    yamlord.write_yaml(cfg, file_cfg_yaml)

    return cfg
