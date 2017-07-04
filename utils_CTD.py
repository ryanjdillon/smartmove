def matlab2datetime(matlab_datenum):
    '''Convert Matlab datenum to Python datetime'''
    import datetime

    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - \
              datetime.timedelta(days=366)

    return day + dayfrac

def zip_stations(names, stations):
    import pandas

    # Create label, lon, lat dataframe
    data = pandas.DataFrame(index=range(n), columns=['id', 'lon', 'lat'])

    n0 = 0
    for l, d in zip(names, stations):
        n1 = len(d['lon'])-1
        data.ix[n0:n0+n1, 'lon'] = d['lon'].T
        data.ix[n0:n0+n1, 'lat'] = d['lat'].T
        data.ix[n0:n0+n1, 'id'] = l
        n0 = n1

    return data


def plot_transect_data(data):
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
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import pickle

from rjdtools import yaml_tools
from rjdtools.plot import maps

# Read data
paths = yaml_tools.read_yaml('./cfg_paths.yaml')
paths_sal = os.path.join(paths['root'], 'data_salinity')

path_KF2 = os.path.join(paths_sal, '2016_KF2.p')
path_KF4 = os.path.join(paths_sal, '2016_KF4.p')
path_KF5 = os.path.join(paths_sal, '2016_KF5.p')

KF2 = pandas.read_pickle(path_KF2)
KF4 = pandas.read_pickle(path_KF4)
KF5 = pandas.read_pickle(path_KF5)

# Cound number of lons/lats, remove prefix from keys
n = 0
for d in [KF2, KF4, KF5]:
    keys = list(d.keys())
    for k in keys:
        if k.startswith('KF'):
            d[k[4:]] = d.pop(k)
    n += len(d['lon'])

import gsw
for d in [KF2, KF4, KF5]:
    n_depths, n_stations = d['date_up'].shape
    d['depth'] = numpy.zeros((n_depths, n_stations), dtype=float)
    d['date_up'] = d['date_up'].astype(object)
    lats = numpy.meshgrid(d['lat'], numpy.ones(n_depths))[0]

#    v_zfp = numpy.vectorize(gsw.z_from_p)
#    v_dt = numpy.vectorize(matlab2datetime)
#
#    d['depth'] = v_zfp(d['depth'], lats)
#    try:
#        d['date_up'] = v_zfp(d['date_up'])
#    except:
#        pass

    for i in range(n_depths):
        for j in range(n_stations):
            d['depth'][i][j] = gsw.z_from_p(d['press_up'][i][j], lats[i][j])
            try:
                d['date_up'][i][j] = matlab2datetime(d['date_up'][i][j])
            except:
                pass

## Save those to make things easier later
#pickle.dump(file=open(path_KF2, 'wb'), obj=KF2)
#pickle.dump(file=open(path_KF4, 'wb'), obj=KF4)
#pickle.dump(file=open(path_KF5, 'wb'), obj=KF5)

stations = ['KF2', 'KF4', 'KF5']
names = [KF2, KF4, KF5]
data = zip_stations(names, stations)

plot_transect_data(data)
