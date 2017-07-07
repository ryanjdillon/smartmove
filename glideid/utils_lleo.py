
def speed_calibration_average(cal_fname, plot=False):
    '''Cacluate the coefficients for the mean fit of calibrations

    Notes
    -----
    `cal_fname` should contain three columns:
    date,est_speed,count_average
    2014-04-18,2.012,30
    '''
    import matplotlib.pyplot as plt
    import numpy
    import pandas

    # Read calibration data
    calibs = pandas.read_csv(cal_fname)

    # Get unique dates to process fits for
    dates = numpy.unique(calibs['date'])

    # Create x data for samples and output array for y
    n_samples = 1000
    x = numpy.arange(n_samples)
    fits = numpy.zeros((len(dates), n_samples), dtype=float)

    # Calculate fit coefficients then store `n_samples number of samples
    # Force intercept through zero (i.e. zero counts = zero speed)
    # http://stackoverflow.com/a/9994484/943773
    for i in range(len(dates)):
        cal = calibs[calibs['date']==dates[i]]
        xi = cal['count_average'].values[:, numpy.newaxis]
        yi = cal['est_speed'].values
        m, _, _, _ = numpy.linalg.lstsq(xi, yi)
        fits[i, :] = m*x
        # Add fit to plot if switch on
        if plot:
            plt.plot(x, fits[i,:], label='cal{}'.format(i))

    # Calculate average of calibration samples
    y_avg = numpy.mean(fits, axis=0)

    # Add average fit to plot and show if switch on
    if plot:
        plt.plot(x, y_avg, label='avg')
        plt.legend()
        plt.show()

    # Calculate fit coefficients for average samples
    x_avg = x[:, numpy.newaxis]
    m_avg, _, _, _ = numpy.linalg.lstsq(x_avg, y_avg)

    return m_avg


def load_lleo(path_data_acc, file_cal_acc, file_cal_speed, min_depth):
    '''Load lleo data for calculating body condition'''
    import numpy
    import os

    from pylleo.pylleo import lleoio, lleocal

    import utils_prh
    import utils_plot
    import yamlord

    # Parse tag model and id from directory/experiment name
    experiment_id = os.path.split(path_data_acc)[1].replace('-','')
    tag_model = experiment_id.split('_')[1]
    tag_id = int(experiment_id.split('_')[2])

    # Load calibrate data
    cal_dict = yamlord.read_yaml(file_cal_acc)

    # Verify sensor ID of data matches ID of CAL
    # TODO add tag_id to pylleo cal, must enter manually now
    if cal_dict['tag_id'] != tag_id:
        raise SystemError('Data `tag_id` does not match calibration `tag_id`')

    # Load meta data
    meta = lleoio.read_meta(path_data_acc, tag_model, tag_id)

    # Load data
    sample_f  = 1
    sensors = lleoio.read_data(meta, path_data_acc, sample_f, overwrite=False)

    # Apply calibration to data
    sensors['Ax_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_x')
    sensors['Ay_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_y')
    sensors['Az_g'] = lleocal.apply_poly(sensors, cal_dict, 'acceleration_z')

    # Calibrate propeller measurements to speed m/s^2
    m_speed = utils_prh.speed_calibration_average(file_cal_speed)
    sensors['speed'] = m_speed*sensors['propeller']

    # Linearly interpolate data
    sensors.interpolate('linear', inplace=True)

    # TODO remove, leave diagnostic plot in parent routine?
    exp_ind = range(len(sensors))
    utils_plot.plot_swim_speed(exp_ind, sensors['speed'].values)

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    return sensors, dt_a, fs_a#A_g, depths, speed, temperature,


