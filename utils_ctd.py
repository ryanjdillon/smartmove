def read_matlab(ctd_mat_file):
    from collections import OrderedDict
    import scipy.io

    mat = scipy.io.loadmat(ctd_mat_file)
    keys = sorted(list(mat.keys()))

    # Assign not meta fields to separate dicts for transects
    transects = OrderedDict()
    for k in keys:
        # Ignore meta fields in `mat` dict
        if not k.startswith('__'):
            k_id = k.split('_')[0]
            # Create station dict if not present
            if k_id not in transects.keys():
                transects[k_id] = OrderedDict()
            # Add field to station dict, remove k_id from k
            k_sub = '_'.join(k.split('_')[1:])
            transects[k_id][k_sub] = mat[k]

    # Calcuate depth from pressure, convert matlab time to datestamp
    for k in transects.keys():
        transects[k] = calc_depth_date(transects[k])

    return transects


def calc_depth_date(d):

    def matlab2datetime(matlab_datenum):
        '''Convert Matlab datenum to Python datetime'''
        import datetime

        day = datetime.datetime.fromordinal(int(matlab_datenum))
        dayfrac = datetime.timedelta(days=matlab_datenum%1) - \
                  datetime.timedelta(days=366)

        return day + dayfrac

    import gsw
    import numpy

    # Remove prefix from columns of station dataframe
    rm_prefix(d, 'KF')

    n_depths, n_stations = d['date_up'].shape
    d['depth'] = numpy.zeros((n_depths, n_stations), dtype=float)
    d['date_up'] = d['date_up'].astype(object)
    lats = numpy.meshgrid(d['lat'], numpy.ones(n_depths))[0]

    for i in range(n_depths):
        for j in range(n_stations):
            d['depth'][i][j] = gsw.z_from_p(d['press_up'][i][j], lats[i][j])
            try:
                d['date_up'][i][j] = matlab2datetime(d['date_up'][i][j])
            except:
                pass
    return d


def get_station_lonlats(transects):
    import pandas

    # Count number of lons/lats in all station dataframes
    n = 0
    for d in transects.values():
        n += len(d['lon'])

    # Create label, lon, lat dataframe
    data = pandas.DataFrame(index=range(n), columns=['id', 'lon', 'lat'])

    n0 = 0
    for l, d in transects.items():
        n1 = len(d['lon'])-1
        data.ix[n0:n0+n1, 'lon'] = d['lon'].T
        data.ix[n0:n0+n1, 'lat'] = d['lat'].T
        data.ix[n0:n0+n1, 'id'] = l
        n0 = n1

    return data


def plot_transect_data(transects):
    '''Plot CTD transects from transects dictionary'''
    import matplotlib.pyplot as plt

    from rjdtools.plot import maps

    # Read lon/lat data from transects dict
    data = get_station_lonlats(transects)

    # import transect data
    fig, ax = plt.subplots()

    scale = 2.0
    #map_props = maps.center_basemap(data['lon'], data['lat'], scale,
    #                                             return_dict=True)
    map_props = {'lon0': 18.55, 'lat0': 69.75, 'mapW':15000, 'mapH':15000}
    map_colors = maps.get_mapcolors()
    m = maps.draw_basemap(ax, map_props, map_colors, res='f')

    for label, c in zip(['KF2', 'KF4', 'KF5'], ['red','blue','green']):
        lons = data[data['id']==label]['lon'].values
        lats = data[data['id']==label]['lat'].values
        x, y = m(lons, lats)
        ax.scatter(x, y, color='black')
        ax.plot(x, y, color=c, label=label)

    ax.legend()
    plt.show()

    return None


def find_nearest_station(lon, lat, transects):
    from collections import OrderedDict
    import pyproj

    g = pyproj.Geod(ellps='WGS84')

    nearest_key = ''
    nearest_idx = ''
    min_dist = 9e12
    for k in transects.keys():
        lons = transects[k]['lon']
        lats = transects[k]['lat']

        for i in range(len(lons)):
            _, _, dist = g.inv(lon, lat, lons[i], lats[i])
            if dist[0] < min_dist:
                nearest_key = k
                nearest_idx = i
                min_dist = dist[0]

    return nearest_key, nearest_idx, min_dist


def calc_mean_salinity(transects, nearest_key, nearest_idx, max_depth):

    # Cacluate mean salinity above max_depth
    depths = transects[nearest_key]['depth'][:, nearest_idx]
    depth_mask = depths > -abs(max_depth)

    sals = transects[nearest_key]['sal_up'][:, nearest_idx]
    mean_sal = numpy.mean(sals[depth_mask])

    return mean_sal


if __name__ == '__main__':
    import os

    from rjdtools import yaml_tools

    # Study location and max depth
    # 69° 41′ 57.9″ North, 18° 39′ 4.5″ East
    lat = 69.69941666666666
    lon = 18.65125
    max_depth = 18 # meters

    ## Read data
    #cfg_path = './cfg_paths.yaml'
    #paths = yaml_tools.read_yaml(cfg_path)
    #paths_ctd = os.path.join(paths['root'], path['ctd'])
    ctd_mat_file = ('/home/ryan/Desktop/edu/01_PhD/projects/smartmove/data_coexist/'
                   'data_ctd/kaldfjorden2016_inner.mat')

    transects = read_matlab(ctd_mat_file)

    # Find nearest station
    nearest_key, nearest_idx, min_dist = find_nearest_station(lon, lat, transects)

    # Cacluate mean salinity above 18m
    mean_sal = calc_mean_salinity(transects, nearest_key, nearest_idx, max_depth):

    plot_transect_data(data)
