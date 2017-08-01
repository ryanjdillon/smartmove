'''
Body density estimation. Japan workshop May 2016

Python implementation:  Ryan J. Dillon
Original Matlab Author: Lucia Martina Martin Lopez

Attributes
----------
sensors: pandas.DataFrame
    contains calibrated sensor data and data processed within glide analysis
exp_ind: numpy.ndarray
    indices of `sensors` data to be analyzed
dives: pandas.DataFrame
    start_idx
    stop_idx
    dive_dur
    depths_max
    depths_max_idx
    depths_mean
    compr_mean
cutoff_frq: float
    cutoff frequency for separating low and high frequency signals
stroke_frq: float
    frequency at which maximum power is seen in accelerometer PSD
J:
    frequency of stroke signal in accelerometer data (m/s2)
t_max:
    
GL: ndarray
    start/stop indices of glide events in `sensors` data
SGL:
    start/stop indices of subglide events in `sensors` data
sgls: pandas.DataFrame
  Contains subglide summary information of `sensors` data
glide_ratio: pandas.DataFrame
  Contains glide ratio summary information of `sensors` data
t_max: int
    maximum duration allowable for a fluke stroke in seconds, it can be set as
    1/`stroke_frq`
J:
    magnitude threshold for detecting a fluke stroke in [m/s2]
'''
# TODO glide identification performed on z-axis, change?

# TODO CLARIFY `stroke_frq` vs `fluke rate` low pass vs high-pass signals

# TODO add experiment info: # dives, # subglides asc/des in cfg_filter.yml
# TODO look into t_max / t_max yaml
# TODO use name convention in rest of repo: fname_, path_, etc.
# TODO GL and dives ind saved in masks? have routine that calcs dive info
# TODO move config glide, sgl, filter to default yaml files?
# TODO get rid of dives and glide_ratios
# TODO cleanup docstring(s)
# TODO sort out normalization and glide stats: get_stroke_freq()>automatic_freq()
# TODO put all masks into `acc_masks` dataframe for simplicity
# TODO move unessecary plotting to own routine for switching on/off

def run_all(cfg_project, debug=False, plots=True):
    import os
    import pyotelem
    import yamlord

    from .. import utils


    paths   = cfg_project['paths']
    fnames  = cfg_project['fnames']

    # Generate list of paths in acc data directory
    path_exps = list()
    for path_exp in os.listdir(os.path.join(paths['project'], paths['acc'])):

        # Only process directories
        if os.path.isdir(os.path.join(paths['project'], paths['acc'], path_exp)):

            path_exps.append(path_exp)

    # Get user selection of acc paths to process
    path_exps = sorted(path_exps)
    msg = 'Enter paths numbers to process:\n'
    process_ind = pyotelem.utils.get_dir_indices(msg, path_exps)

    # Process selected acc experiments
    for i in process_ind:
        path_exp = path_exps[i]

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
        cfg_glide, sensors, sgls, dives, glide_ratio = lleo_glide_analysis(cfg_project,
                                                                           path_exp,
                                                                           path_cal_acc,
                                                                           plots=plots,
                                                                           debug=debug)
    return cfg_glide, sensors, sgls, dives, glide_ratio



def lleo_glide_analysis(cfg_project, path_exp, path_cal_acc, plots=True, debug=False):
    '''Run glide analysis with little leonarda data'''
    from collections import OrderedDict
    import numpy
    import os
    import yamlord
    from pyotelem.plots import plotdynamics, plotglides
    import pyotelem.glides

    from .. import utils
    from . import utils_lleo

    paths = cfg_project['paths']
    fnames = cfg_project['fnames']

    # Input filenames
    fname_cal = fnames['tag']['cal']
    fname_cal_speed = fnames['tag']['cal_speed']

    # Input paths
    path_data_acc  = os.path.join(paths['project'], paths['acc'], path_exp)
    file_cal_acc   = os.path.join(paths['project'], paths['acc'], path_cal_acc, fname_cal)
    file_cal_speed = os.path.join(paths['project'], paths['acc'], fname_cal_speed)

    # Output filenames
    fname_cfg_exp = fnames['tag']['cfg_exp']
    file_cfg_exp  = os.path.join(paths['project'], paths['acc'], fname_cfg_exp)

    fname_cfg_glide = fnames['glide']['cfg_glide']
    fname_cfg_sgl   = fnames['glide']['cfg_sgl']
    fname_cfg_filt  = fnames['glide']['cfg_filt']

    fname_sensors             = fnames['tag']['data'].format(path_exp)
    fname_dives               = fnames['glide']['dives']
    fname_sgls                = fnames['glide']['sgls']
    fname_glide_ratio         = fnames['glide']['glide_ratio']
    fname_mask_sensors        = fnames['glide']['mask_sensors']
    fname_mask_sensors_glides = fnames['glide']['mask_sensors_glides']
    fname_mask_sensors_sgls   = fnames['glide']['mask_sensors_sgls']
    fname_mask_sensors_filt   = fnames['glide']['mask_sensors_filt']
    fname_mask_sgls           = fnames['glide']['mask_sgls']
    fname_mask_sgls_filt      = fnames['glide']['mask_sgls_filt']

    # Setup configuration files
    cfg_glide = _cfg_glide_params()
    cfg_sgl   = _cfg_sgl_params()
    cfg_filt  = _cfg_filt_params()

    # Output paths
    ignore = ['nperseg', 'peak_thresh', 'alpha', 'min_depth', 't_max, t_max']
    out_data  = os.path.join(paths['project'], paths['acc'], path_exp)
    os.makedirs(out_data, exist_ok=True)

    # LOAD DATA
    #----------
    # linearly interpolated sensors to accelerometer sensor
    sensors, dt_a, fs_a = utils_lleo.load_lleo(path_data_acc, file_cal_acc,
                                    file_cal_speed)

    # Plot speed if debug is on
    if debug:
        exp_ind = range(len(sensors))
        plotdynamics.plot_swim_speed(exp_ind, sensors['speed'].values)


    # Signal process data, calculate derived data and find stroke freqencies
    cfg_glide, sensors, dives, masks, exp_ind = _process_sensor_data(
                                                           cfg_project,
                                                           path_exp,
                                                           cfg_glide,
                                                           file_cfg_exp,
                                                           sensors,
                                                           fs_a,
                                                           Mw=None,
                                                           plots=plots,
                                                           debug=debug)
    # Save data
    sensors.to_pickle(os.path.join(out_data, fname_sensors))
    dives.to_pickle(os.path.join(out_data, fname_dives))
    masks.to_pickle(os.path.join(out_data, fname_mask_sensors))

    # Find Glides
    #------------
    GL, masks['glides'], glide_ratio = _process_glides(
                                                      cfg_glide,
                                                      sensors,
                                                      fs_a,
                                                      dives,
                                                      masks,
                                                      plots=plots,
                                                      debug=debug)

    # Save glide ratio dataframe
    dname_glide = utils.cat_keyvalues(cfg_glide, ignore)
    out_glide = os.path.join(paths['project'], paths['glide'], path_exp, dname_glide)
    os.makedirs(out_glide, exist_ok=True)
    glide_ratio.to_pickle(os.path.join(out_glide, fname_glide_ratio))

    # Save glide mask of sensor dataframe
    masks['glides'].to_pickle(os.path.join(out_glide, fname_mask_sensors_glides))

    # Save glide analysis configuration
    path_cfg_yaml = os.path.join(out_glide, fname_cfg_glide)
    _save_config(cfg_glide, path_cfg_yaml, 'glides')


    # SPLIT GLIDES
    #-------------
    # Split into subglides, generate summary tables
    sgls, masks['sgls'] = _process_sgls(cfg_sgl, sensors, fs_a, GL, dives)

    # Save subglide dataframe
    dname_sgl   = utils.cat_keyvalues(cfg_sgl, ignore)
    out_sgl   = os.path.join(out_glide, dname_sgl)
    os.makedirs(out_sgl, exist_ok=True)
    sgls.to_pickle(os.path.join(out_sgl, fname_sgls))

    # Save subglide mask of sensor dataframe
    masks['sgls'].to_pickle(os.path.join(out_sgl, fname_mask_sensors_sgls))

    # Save subglide analysis configuation
    _save_config(cfg_sgl, os.path.join(out_sgl, fname_cfg_sgl), 'sgls')


    # FILTER SUBGLIDES
    #-----------------
    # Include duration in filtering if splitting is fast enough?
    # Filter subglides
    exp_ind = numpy.where(masks['exp'])[0]
    masks['filt_sgls'], sgls['mask'] = utils.filter_sgls(len(sensors),
                                                   exp_ind,
                                                   sgls,
                                                   cfg_filt['pitch_thresh'],
                                                   cfg_filt['min_depth'],
                                                   cfg_filt['max_depth_delta'],
                                                   cfg_filt['min_speed'],
                                                   cfg_filt['max_speed'],
                                                   cfg_filt['max_speed_delta'])

    # Plot filtered data
    plotglides.plot_sgls(sensors['depth'].values, masks['filt_sgls'], sgls,
                         sgls['mask'], sensors['p_lf'].values,
                         sensors['r_lf'].values, sensors['h_lf'].values)

    # Save filtered subglide mask of sensor dataframe
    dname_filt  = utils.cat_keyvalues(cfg_filt, ignore)
    out_filt  = os.path.join(out_sgl, dname_filt)
    os.makedirs(out_filt, exist_ok=True)
    masks['filt_sgls'].to_pickle(os.path.join(out_filt, fname_mask_sensors_filt))

    # Save filtered subglide mask of sgl dataframe
    # TODO remove?
    sgls['mask'].to_pickle(os.path.join(out_filt, fname_mask_sgls_filt))

    # Save symlink to data and masks in filter directory
    utils.create_symlink(os.path.join(out_data, fname_sensors),
                         os.path.join(out_filt, fname_sensors))
    utils.create_symlink(os.path.join(out_data, fname_mask_sensors),
                         os.path.join(out_filt, fname_mask_sensors))
    utils.create_symlink(os.path.join(out_glide, fname_mask_sensors_glides),
                         os.path.join(out_filt, fname_mask_sensors_glides))
    utils.create_symlink(os.path.join(out_sgl, fname_mask_sensors_sgls),
                         os.path.join(out_filt, fname_mask_sensors_sgls))
    utils.create_symlink(os.path.join(out_sgl, fname_sgls),
                         os.path.join(out_filt, fname_sgls))

    # Save filter analysis configuation
    cfg_all           = OrderedDict()
    cfg_all['glide']  = cfg_glide
    cfg_all['sgl']    = cfg_sgl
    cfg_all['filter'] = cfg_filt
    _save_config(cfg_all, os.path.join(out_filt, fname_cfg_filt))

    return cfg_filt, sensors, sgls, dives, glide_ratio


def _process_sensor_data(cfg_project, path_exp, cfg_glide, file_cfg_exp, sensors, fs_a,
        Mw=None, plots=True, debug=False):
    '''Calculate body conditions summary statistics'''
    from collections import OrderedDict
    import numpy
    import os
    import pandas
    import pyotelem
    from pyotelem.plots import plotdives, plotdsp
    import yamlord

    from .. import utils
    from . import utils_ctd

    paths = cfg_project['paths']
    fnames = cfg_project['fnames']

    exp_idxs = [None, None]
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
        plotdives.plot_triaxial_depths_speed(sensors)

        # Get indices user input - mask
        exp_idxs[0] = pyotelem.utils.recursive_input('Analysis start index', int)
        exp_idxs[1] = pyotelem.utils.recursive_input('Analysis stop index', int)

        cfg_exp[path_exp] = OrderedDict()
        cfg_exp[path_exp]['start_idx'] = exp_idxs[0]
        cfg_exp[path_exp]['stop_idx']  = exp_idxs[1]
        yamlord.write_yaml(cfg_exp, file_cfg_exp)

    # Creat dataframe for storing masks for various views of the data
    masks = pandas.DataFrame(index=range(len(sensors)), dtype=bool)

    # Create mask of values to be considered part of the analysis
    masks['exp'] = False
    masks['exp'][exp_idxs[0]:exp_idxs[1]] = True

    # Create indices array `exp_ind` for analysis
    exp_ind = numpy.where(masks['exp'])[0]


    # 1.3 Calculate pitch, roll, and heading
    #--------------------------------------------------------------------------
    print('* Calculate pitch, roll, heading\n')
    sensors['p'], sensors['r'], sensors['h'] = pyotelem.dynamics.prh(sensors['Ax_g'].values,
                                                         sensors['Ay_g'].values,
                                                         sensors['Az_g'].values)

    # 2 Define dives
    #--------------------------------------------------------------------------
    # TODO use min_dive_depth and min_analysis_depth?
    print('* Define dives\n')
    dives, masks['dive'] = pyotelem.dives.finddives2(sensors['depth'].values,
                                                  cfg_glide['min_depth'])


    # 3.2.1 Determine `stroke_frq` fluking rate and cut-off frequency
    #--------------------------------------------------------------------------
    print('* Get stroke frequency\n')
    # calculate power spectrum of the accelerometer data at the whale frame
    Ax_g = sensors['Ax_g'][masks['exp']].values
    Az_g = sensors['Az_g'][masks['exp']].values

    # NOTE change `stroke_ratio` here to modify selectio method
    # TODO should perform initial lp/hp filter, then `stroke_f` comes from high-pass
    # should be OK other than t_max, these values are too high
    if debug is False:
        cutoff_frq, stroke_frq, stroke_ratio = pyotelem.glides.get_stroke_freq(Ax_g,
                                                       Az_g,
                                                       fs_a,
                                                       cfg_glide['nperseg'],
                                                       cfg_glide['peak_thresh'],
                                                       stroke_ratio=None)
        # Store user input cutoff and stroke frequencies
        cfg_glide['cutoff_frq']   = cutoff_frq
        cfg_glide['stroke_frq']   = stroke_frq
        cfg_glide['stroke_ratio'] = stroke_ratio

        # Calculate maximum duration of glides from stroke frequency
        cfg_glide['t_max']  = 1 /cfg_glide['stroke_frq']  # seconds
    else:
        cutoff_frq = 0.3
        cfg_glide['cutoff_frq'] = cutoff_frq


    # 3.2.2 Separate low and high frequency signals
    #--------------------------------------------------------------------------
    print('* Separate accelerometry to high and low-pass signals\n')
    order = 5
    cutoff_str = str(cfg_glide['cutoff_frq'])
    for btype, suffix in zip(['low', 'high'], ['lf', 'hf']):
        b, a, = pyotelem.dsp.butter_filter(cfg_glide['cutoff_frq'], fs_a, order=order,
                btype=btype)
        for param in ['Ax_g', 'Ay_g', 'Az_g']:
            key = '{}_{}_{}'.format(param, suffix, cutoff_str)
            sensors[key] = pyotelem.dsp.butter_apply(b, a, sensors[param].values)

    # Plot low and high freqency accelerometer signals
    if plots is True:
        plotdsp.plot_lf_hf(sensors['Ax_g'][masks['exp']],
                           sensors['Ax_g_lf_'+cutoff_str][masks['exp']],
                           sensors['Ax_g_hf_'+cutoff_str][masks['exp']],
                           title='x axis')

        plotdsp.plot_lf_hf(sensors['Ay_g'][masks['exp']],
                           sensors['Ay_g_lf_'+cutoff_str][masks['exp']],
                           sensors['Ay_g_hf_'+cutoff_str][masks['exp']],
                           title='y axis')

        plotdsp.plot_lf_hf(sensors['Az_g'][masks['exp']],
                           sensors['Az_g_lf_'+cutoff_str][masks['exp']],
                           sensors['Az_g_hf_'+cutoff_str][masks['exp']],
                           title='z axis')


    # 3.2.3 Calculate the smooth pitch from the low pass filter acceleration
    #       signal to avoid incorporating signals above the stroking periods
    #--------------------------------------------------------------------------
    print('* Calculate low-pass pitch, roll, heading\n')
    prh_lf = pyotelem.dynamics.prh(sensors['Ax_g_lf_'+cutoff_str].values,
                                   sensors['Ay_g_lf_'+cutoff_str].values,
                                   sensors['Az_g_lf_'+cutoff_str].values,)

    sensors['p_lf'], sensors['r_lf'], sensors['h_lf'] = prh_lf


    # 4 Define precise descent and ascent phases
    #--------------------------------------------------------------------------
    print('* Get precise indices of descents, ascents, phase and bottom\n')
    masks['des'], masks['asc'] = pyotelem.dives.get_des_asc2(sensors['depth'].values,
                                                  masks['dive'].values,
                                                  sensors['p_lf'].values,
                                                  cfg_glide['cutoff_frq'],
                                                  fs_a,
                                                  order=5)
    # Typecast des/asc columns to `bool`
    masks = masks.astype(bool)
    if plots is True:
        plotdives.plot_dives_pitch(sensors['depth'][masks['exp']],
                                   masks['dive'][masks['exp']],
                                   masks['des'][masks['exp']],
                                   masks['asc'][masks['exp']],
                                   sensors['p'][masks['exp']],
                                   sensors['p_lf'][masks['exp']])


    # 8 Estimate seawater density around the tagged animal
    #--------------------------------------------------------------------------
    print('* Estimate seawater density\n')

    # Study location and max depth to average salinities
    lon = cfg_project['experiment']['coords']['lon']
    lat = cfg_project['experiment']['coords']['lat']
    lat = cfg_project['experiment']['coords']['lat']
    max_depth = cfg_project['experiment']['net_depth']

    # Read data
    fname_ctd = fnames['glide']['ctd']
    file_ctd_mat = os.path.join(paths['project'], paths['ctd'], fname_ctd)

    t = sensors['temperature'].values

    sensors['dsw'] = utils_ctd.get_seawater_densities(file_ctd_mat, t, lon, lat,
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

    # TODO Need to set J (signal threshold) here, user input should be the
    # power, not the frequency. Just use a standard plot of acceleration here?

    if debug is False:
        # Plot PSD for J selection
        Ax_g_hf = sensors['Ax_g_hf_'+cutoff_str][masks['exp']].values
        Az_g_hf = sensors['Az_g_hf_'+cutoff_str][masks['exp']].values

        f_wx, Sx, Px, dpx = pyotelem.dsp.calc_PSD_welch(Ax_g_hf, fs_a, nperseg=512)
        f_wz, Sz, Pz, dpz = pyotelem.dsp.calc_PSD_welch(Az_g_hf, fs_a, nperseg=512)

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(f_wx, Sx, label='hf-x PSD')
        ax1.plot(f_wz, Sz, label='hf-z PSD')
        ax1.legend(loc='upper right')
        ax2.plot(sensors['datetimes'][masks['exp']], Ax_g_hf, label='hf-x')
        ax2.plot(sensors['datetimes'][masks['exp']], Az_g_hf, label='hf-z')
        ax2.legend(loc='upper right')

        fig.autofmt_xdate()
        plt.show()

        # Get user selection for J - select one for both axes
        cfg_glide['J'] = pyotelem.utils.recursive_input('J (fluke magnitude)', float)
    else:
        cfg_glide['J'] = 0.4

    return cfg_glide, sensors, dives, masks, exp_ind


def _process_glides(cfg_glide, sensors, fs_a, dives, masks, Mw=None, plots=True,
        debug= False):
    import numpy

    from .. import utils
    import pyotelem.glides

    cutoff_str = str(cfg_glide['cutoff_frq'])

    # TODO t_max * fs_a in routine below, 16.0 in cfg, check Kagari
    if Mw is None:
        # Get GL from dorso-ventral axis of the HPF acc signal
        GL = pyotelem.glides.get_stroke_glide_indices(sensors['Az_g_hf_'+cutoff_str].values,
                                                   fs_a,
                                                   cfg_glide['J'],
                                                   cfg_glide['t_max'])

    # TODO review once magnet_rot() routine finished
    elif Mw is not None:
        MagAcc, pry, Sa, GL, heading_lf, pitch_lf_deg = calc_mag_heading()

    # TODO
    # check glides duration and positive and negative zero crossings based
    # on selected J and t_max#


    # 10 Calculate glide ratio TODO keep?
    #--------------------------------------------------------------------------
    print('* Calculate glide ratio\n')
    glide_mask = utils.mask_from_noncontiguous_indices(len(sensors), GL[:,0], GL[:,1])

    glide_ratio = pyotelem.glides.calc_glide_ratios(dives,
                                                 masks['des'].values,
                                                 masks['asc'].values,
                                                 glide_mask,
                                                 sensors['depth'],
                                                 sensors['p_lf'])

    return GL, glide_mask, glide_ratio


def _process_sgls(cfg_sgl, sensors, fs_a, GL, dives):
    '''Split subglides and generate summary dataframe'''
    import numpy

    import pyotelem.glides

    # 7 Make 5sec sub-glides
    #--------------------------------------------------------------------------
    print('* Make sub-glides, duration {}\n'.format(cfg_sgl['dur']))

    SGL, data_sgl_mask = pyotelem.glides.split_glides(len(sensors),
                                                   cfg_sgl['dur'], fs_a, GL)


    # 9 Generate summary information table for subglides
    #--------------------------------------------------------------------------
    pitch_lf_deg = numpy.rad2deg(sensors['p_lf'].values)

    print('* Generate summary information table for subglides\n')
    sgls = pyotelem.glides.calc_glide_des_asc(sensors['depth'].values,
                                           sensors['p_lf'].values,
                                           sensors['r_lf'].values,
                                           sensors['h_lf'].values,
                                           sensors['speed'].values,
                                           dives,
                                           SGL,
                                           pitch_lf_deg,
                                           sensors['temperature'].values,
                                           sensors['dsw'].values)
    return sgls, data_sgl_mask


def _save_config(cfg_add, path_cfg_yaml, name='parameters'):
    '''Load analysis configuration defualts'''
    from collections import OrderedDict
    import datetime

    import yamlord
    from .. import utils

    cfg = OrderedDict()

    # Record the last date modified & git version
    fmt = '%Y-%m-%d_%H%M%S'
    cfg['last_modified'] = datetime.datetime.now().strftime(fmt)

    # Get git hash and versions of dependencies
    # TODO the versions added by this should only be logged in release, or
    # maybe check local installed vs requirements versions
    cfg['versions'] = utils.get_versions('smartmove')

    cfg[name] = cfg_add

    yamlord.write_yaml(cfg, path_cfg_yaml)

    return cfg


def _cfg_glide_params():
    '''Add fields for glide analysis to config dictionary'''
    from collections import OrderedDict
    import numpy

    cfg_glide = OrderedDict()

    # TODO not currently used, useful with magnetometer data
    ## Acceleromter/Magnotometer axis to analyze
    #cfg_glide['axis'] = 0

    # Number of samples per frequency segment in PSD calculation
    cfg_glide['nperseg'] = 256

    # Threshold above which to find peaks in PSD
    cfg_glide['peak_thresh'] = 0.10

    # High/low pass cutoff frequency, determined from PSD plot
    cfg_glide['cutoff_frq'] = None

    # Frequency of stroking, determinded from PSD plot
    cfg_glide['stroke_frq'] = 0.4 # Hz

    # fraction of `stroke_frq` to calculate cutoff frequency (Wn)
    cfg_glide['stroke_ratio'] = 0.4

    # Maximum length of stroke signal
    cfg_glide['t_max'] = 1 / cfg_glide['stroke_frq'] # seconds

    # Minimumn frequency for identifying strokes (3. Get stroke_frq)
    cfg_glide['J'] = '{:.4f}'.format(2 / 180 * numpy.pi) # 0.0349065 Hz

    # For magnetic pry routine
    cfg_glide['alpha'] = 25

    # TODO redundant for filt_params?
    # Minimum depth at which to recognize a dive (2. Define dives)
    cfg_glide['min_depth'] = 0.4

    return cfg_glide


def _cfg_sgl_params():
    '''Add fields for subglide analysis to config dictionary'''
    from collections import OrderedDict

    cfg_sgl = OrderedDict()

    # Duration of sub-glides (8. Split sub-glides, 10. Calc glide des/asc)
    cfg_sgl['dur'] = 2 # seconds

    ## TODO not used
    ## Minimum duration of sub-glides, `False` excludes sublides < dur seconds
    #cfg_sgl['min_dur'] = False # seconds

    return cfg_sgl


def _cfg_filt_params():
    '''Add fields for filtering of subglides to config dictionary'''
    from collections import OrderedDict

    cfg_filt = OrderedDict()

    # Pitch angle (degrees) to consider sgls
    cfg_filt['pitch_thresh'] = 30

    # Minimum depth at which to recognize a dive (2. Define dives)
    cfg_filt['min_depth'] = 0.4

    # Maximum cummulative change in depth over a glide
    cfg_filt['max_depth_delta'] = 8.0

    # Minimum mean speed of sublide
    cfg_filt['min_speed'] = 0.3

    # Maximum mean speed of sublide
    cfg_filt['max_speed'] = 10

    # Maximum cummulative change in speed over a glide
    cfg_filt['max_speed_delta'] = 1.0

    return cfg_filt


if __name__ == '__main__':

    run_all()
