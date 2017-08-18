'''
`monitor_dict` was created manually by loading the best performing network, and
for each `frac` in `prj_NPLEA = pyproj.Proj('+init=EPSG:3574')numpy.arange(0, 1, 0.03)[1:]`, training the algorithm and
recording the results dataframe.

From this dataframe, the last value for `loss`, `err`, and `acc` were extracted
for each of the `monitors` results in the `results_dataset` dataframe.
'''
def lambert_yticks(ax, ticks):
    """Draw ticks on the left y-axis of a Lamber Conformal projection."""
    import numpy
    # http://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e

    def lc(t, n, b):
        '''line constructor'''
        la = numpy.linspace(b[0],b[1],n)
        ta = numpy.zeros(n) + t
        return numpy.vstack((la, ta)).T

    # Tick extractor
    te = lambda xy: xy[1]

    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)

    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ylabels = [ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels]
    ax.set_yticklabels(ylabels)

    return None

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    '''Get tick locations and labels for axis of a Lambert Conformal projection.
    '''
    import shapely.geometry as sgeom
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0],
                                                  xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])

    # Remove ticks that aren't visible:
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)

    return _ticks, ticklabels


def studyarea():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pyproj

    # TODO convert SOSI border data to shape and use instead of natural earth
    # https://github.com/torbjvi/sosi2shape.py/blob/master/sosi2shape.py

    # +proj=laea +lat_0=90 +lon_0=10 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84
    # +units=m +no_defs
    #prj_NPLAEA = pyproj.Proj('+init=EPSG:3575')

    g = ccrs.Globe(datum='WGS84', ellipse='WGS84')

    # North Pole Lambert Azmithul Equal Area - Europe
    nplaea = ccrs.LambertAzimuthalEqualArea(central_longitude=10.0,
                                            central_latitude=90.0,
                                            false_easting=0.0,
                                            false_northing=0.0,
                                            globe=g)

    ax = plt.axes(projection=nplaea)
    ax.set_extent([18, 19, 69.5, 70.0])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    minor_islands = cfeature.NaturalEarthFeature(category='physical',
                                               name='minor_islands_coastline',
                                               scale='10m',
                                               facecolor='none')

    lons = [18.65124,]
    lats = [69.69942,]

    ax.plot(lons, lats, color='blue', linewidth=2, marker='o',
            transform=ccrs.Geodetic())

    ax.add_feature(minor_islands, edgecolor='gray')
    ax.coastlines(resolution='10m')

    plt.show()

    ## Add lines for lon/lat
    #gl = ax.gridlines(crs=nplaea, draw_labels=True, linewidth=1, color='gray',
    #                  alpha=0.5, linestyle='--')

    #gl.xlabels_top = False
    #gl.ylabels_left = False

    ##gl.xlines = False
    #gl.xlocator = mticker.FixedLocator([18, 19])
    #gl.ylocator = mticker.FixedLocator([69, 70])
    #gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    #gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    #gl.xlabel_style = {'size': 15, 'color': 'gray'}
    ##gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.COASTLINE)
    #SOURCE = 'Natural Earth'
    #LICENSE = 'public domain'

    return None

def dataset_size(monitor_dict, fracs):
    import matplotlib.pyplot as plt

    plt.plot(fracs, m['acc']['train'], label='train accuracy')
    plt.plot(fracs, m['acc']['valid'], label='valid accuracy')
    plt.legend(loc='bottom right')
    plt.xlabel('Fraction of original dataset')
    plt.ylabel('Accuracy')

    plt.show()

if __name__ == '__main__':
    import numpy
    import pandas
    import yamlord

    paths = yamlord.read_yaml(os.path.join(path_code, 'cfg_paths.yaml'))

    path_root = paths['root']
    path_ann = paths['ann']
    path_model = os.path.join(path_root, path_ann, 'theanets_20170316_132225')

    # Plot dataset size accuracy
    fracs = numpy.arange(0, 1, 0.03)[1:]
    monitor_dict = pandas.read_pickle(os.path.join(path_model,
                                                   'dataset_monitors_end.p'))
    dataset_size(monitor_dict, fracs)
