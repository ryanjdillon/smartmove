
def load_lleo(path_data_tag, file_cal_acc, file_cal_prop):
    '''Load lleo data for calculating body condition

    Args
    ----
    path_data_tag: str
        Path to datalogger data files
    file_cal_acc: str
        Path to accelerometer calibration file to use for calibrating
    file_cal_prop: str
        Path to propeller calibration file to use for calibrating

    Returns
    -------
    tag: pandas.DataFrame
        Dataframe with sensor data and calibrated sensor data
    dt_a: float
        Sampling rate of interpolated data
    fs_a: float
        Sampling frequency of interpolated data
    '''
    import numpy
    import os
    from pylleo import lleoio, lleocal
    import yamlord

    # Parse tag model and id from directory/experiment name
    experiment_id = os.path.split(path_data_tag)[1].replace('-','')
    tag_model = experiment_id.split('_')[1]
    tag_id = int(experiment_id.split('_')[2])

    # Load calibrate data
    cal_dict = yamlord.read_yaml(file_cal_acc)

    # Verify sensor ID of data matches ID of CAL
    if cal_dict['tag_id'] != str(tag_id):
        raise SystemError('Data `tag_id` ({}) does not match calibration '
                          '`tag_id` ({})'.format(tag_id, cal_dict['tag_id']))

    # Load meta data
    meta = lleoio.read_meta(path_data_tag, tag_model, tag_id)

    # Load data
    sample_f  = 1
    tag = lleoio.read_data(meta, path_data_tag, sample_f, overwrite=False)

    # Apply calibration to data
    tag = lleocal.calibrate_acc(tag, cal_dict)

    # Calibrate propeller measurements to speed m/s^2
    tag = lleocal.calibrate_propeller(tag, file_cal_prop)

    # Linearly interpolate data
    tag.interpolate('linear', inplace=True)

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    return tag, dt_a, fs_a
