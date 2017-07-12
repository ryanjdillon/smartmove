
def load_lleo(path_data_acc, file_cal_acc, file_cal_speed):
    '''Load lleo data for calculating body condition

    Args
    ----
    path_data_acc: str
        Path to datalogger data files
    file_cal_acc: str
        Path to accelerometer calibration file to use for calibrating
    file_cal_speed: str
        Path to propeller calibration file to use for calibrating

    Returns
    -------
    sensors: pandas.DataFrame
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
    experiment_id = os.path.split(path_data_acc)[1].replace('-','')
    tag_model = experiment_id.split('_')[1]
    tag_id = int(experiment_id.split('_')[2])

    # Load calibrate data
    cal_dict = yamlord.read_yaml(file_cal_acc)

    # Verify sensor ID of data matches ID of CAL
    if cal_dict['tag_id'] != tag_id:
        raise SystemError('Data `tag_id` does not match calibration `tag_id`')

    # Load meta data
    meta = lleoio.read_meta(path_data_acc, tag_model, tag_id)

    # Load data
    sample_f  = 1
    sensors = lleoio.read_data(meta, path_data_acc, sample_f, overwrite=False)

    # Apply calibration to data
    sensors = lleocal.calibrate_acc(sensors, cal_dict)

    # Calibrate propeller measurements to speed m/s^2
    sensors = lleocal.calibrate_propeller(sensors, fil_cal_speed)

    # Linearly interpolate data
    sensors.interpolate('linear', inplace=True)

    # Get original sampling rates of accelerometer and depth sensors
    dt_a = float(meta['parameters']['acceleration_x']['Interval(Sec)'])
    fs_a = 1/dt_a

    return sensors, dt_a, fs_a
