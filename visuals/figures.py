'''
from pyotelem.plots.plotutils import add_alpha_labels
# Add alpha labels
xpos = [0.05, 0.9]
ypos = [0.95, 0.95]
axes = add_alpha_labels([ax1, ax2], xpos=xpos, ypos=ypos, color='black')

{'axes.axisbelow': True,
 'axes.edgecolor': '.8',
 'axes.facecolor': 'white',
 'axes.grid': True,
 'axes.labelcolor': '.15',
 'axes.linewidth': 1.0,
 'figure.facecolor': 'white',
 'font.family': [u'sans-serif'],
 'font.sans-serif': [u'Arial',
  u'DejaVu Sans',
  u'Liberation Sans',
  u'Bitstream Vera Sans',
  u'sans-serif'],
 'grid.color': '.8',
 'grid.linestyle': u'-',
 'image.cmap': u'rocket',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': u'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': u'out',
 'xtick.major.size': 0.0,
 'xtick.minor.size': 0.0,
 'ytick.color': '.15',
 'ytick.direction': u'out',
 'ytick.major.size': 0.0,
 'ytick.minor.size': 0.0}
'''
from os.path import join as _join
_linewidth = 0.5

def plot_sgls_tmbd(exps_all, path_plot=None, dpi=300):
    import numpy
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    import os
    import seaborn
    import scipy.optimize
    from pyotelem.plots import plotutils

    from . import latex

    # Coefficient of determination
    def r_squared(x, y):
        n = len(x)
        numr = ((n*sum(x*y)) - (sum(x)*sum(y)))**2
        denr = ((n*sum(x**2)) - (sum(x)**2)) * \
               ((n*sum(y**2)) - (sum(y)**2))
        return numr/denr

    # Quadratic function to fit data to
    def quadratic(x, a, b, c):
        return a*(x**2) + b*x + c

    # Linear function to fit data to
    def linear(x, m, c):
        return m*x + c

    # Unit normalize ndarray
    def unit_normalize(data):
        return (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))

    def annotate_coords(x, y, dist, theta=None):
        '''Get coords on a circle around x, y with radius `dist` at `theta`'''
        import numpy
        import itertools
        import copy

        n = 1
        if numpy.iterable(x): n = len(x)

        # Create cyclic generator of angles in radians
        deg = [120, 60, 90]
        theta_gen = itertools.cycle(numpy.deg2rad(deg))

        # Create array of theta's at specified angle
        theta = numpy.array([numpy.deg2rad(315)]*len(x))

        # Find neighboring points
        ind = set(range(len(x)))
        neighbors = list()
        for i in ind:
            others = ind - set([i,])
            for j in others:
                d = numpy.sqrt(abs(y[j] - y[i])**2 + abs(x[j]-x[i])**2)
                if d <= dist/2:
                    neighbors.append(j)

        # Set rotating angle of annotation position for neighboring points
        # Caclulate positions of neighbor annotations at `theta` and `dist`
        xa = copy.deepcopy(x)
        ya = copy.deepcopy(y)
        for i in neighbors:
            xa[i] = x[i] + numpy.sin(next(theta_gen))*dist
            ya[i] = y[i] + numpy.cos(next(theta_gen))*dist

        return xa, ya

    colors = seaborn.color_palette()
    seaborn.set_style('whitegrid', {'axes.linewidth':0.5,})

    # Plot colored, sized points
    fig, ax1 = plt.subplots(1, 1)#, figsize=(10,10))

    deltas = exps_all['rho_mod'] - exps_all['density_kgm3']
    udeltas = numpy.unique(deltas)
    animals = numpy.unique(exps_all['animal'])

    # Get all points for des/asc for generating linear fits
    x_all = exps_all['rho_mod'].values
    y_all = exps_all['perc_des'].values

    label_dist = 3
    xa_all, ya_all = annotate_coords(x_all, y_all, label_dist)

    c = 0
    arrowprops = dict(arrowstyle='wedge', facecolor='black')
    for a in animals:

        # Get density and percent des/asc values for animal
        amask = exps_all['animal'] == a
        labels = exps_all['id'][amask]

        x = exps_all['rho_mod'][amask].values
        y = exps_all['perc_des'][amask].values
        xa = xa_all[amask]
        ya = ya_all[amask]

        ind = numpy.where(amask)[0]
        for i in range(len(ind)):
            x = exps_all['rho_mod'][ind[i]]
            y = exps_all['perc_des'][ind[i]]
            exp_id = exps_all['id'][ind[i]]
            subdelta = x - exps_all['density_kgm3'][ind[i]]
            m = numpy.where(subdelta==udeltas)[0][0]

            text = ax1.annotate(exp_id, xy=(x,y), xytext=(xa[i], ya[i]),
                               color=colors[c],
                               horizontalalignment='center',
                               verticalalignment='center',
                               arrowprops=arrowprops)
        c += 1

    # Cacluate curve
    p_opt, p_cov = scipy.optimize.curve_fit(linear, x_all, y_all)
    xi = numpy.arange(x_all.min(), x_all.max(), 0.1)
    yi = linear(xi, *p_opt)
    r2 = r_squared(x_all, y_all)
    str_eqn = '{}x + {}'.format(round(p_opt[0], 4), round(p_opt[1], 2))
    str_r2  = r'$R^{2}$ '+str(round(r2,2))

    ax1.plot(xi, yi, label='fit', color='grey', linestyle='dashed')


    # Manually create legend using matplotlib artist proxies
    # dot size legend
    prox_handles = list()
    prox_labels = list()
    # color dot animal legend
    for i, a in enumerate(animals):
        prox_handles.append(mpatches.Patch(color=colors[i]))
        prox_labels.append(a)

    # Fit to points legend
    prox_handles.append((plt.plot([], [], color='grey', linestyle='dashed'))[0])
    prox_labels.append('Linear fit\n{}'.format(str_r2))

    # Plot regular artists
    leg = plt.legend(handles=prox_handles, labels=prox_labels,
                     bbox_to_anchor=(0, 1), loc='upper left',
                     facecolor='white')

    # Create delta rho_mod table
    ids_str = numpy.array(['']*len(udeltas), dtype=object)
    for i in range(len(udeltas)):
        ids = numpy.unique(exps_all['id'][deltas==udeltas[i]])
        ids_str[i] = ', '.join([str(f) for f in ids])

    table_title = r'$\Delta \, kg \cdot m^{-3}$'+'\n'
    # Use separate strings for each column for alignment
    table_id = ''
    table_rho = ''
    for i in range(len(udeltas)):
        table_id += '{} :\n'.format(ids_str[i])#, udeltas[i])
        table_rho += '{:6.1f}\n'.format(udeltas[i])

    props = dict(facecolor='white', edgecolor='none', alpha=1.0)
    ax1.annotate(table_title, xy=(0.82, 0.53), xycoords='axes fraction',
                 fontsize=12, bbox=props)
    ax1.annotate(table_id, xy=(0.89, 0.05), xycoords='axes fraction',
                 horizontalalignment='right', fontsize=10, bbox=props)
    ax1.annotate(table_rho, xy=(0.90, 0.05), xycoords='axes fraction',
                 fontsize=10, bbox=props)

    # Set x-axis attributes
    ax1.set_xlabel(r'$\rho_{mod} \; (kg \cdot m^{-3})$')

    # Set y-axis attributes ax1
    ax1.set_ylabel('Sub-glides during dive descents $(\%)$')
    ymax = plotutils.roundup(yi.max(), plotutils.magnitude(yi.max()))
    min_mag = plotutils.magnitude(yi.min())
    ymin = (yi.min() // 10**(min_mag)) * 10**(min_mag)
    ax1.set_ylim((ymin, ymax))

    # Create inverted y-axis duplicate, set y-axis attributes ax2
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sub-glides during dive ascents $(\%)$')
    yi_inv = 100 - yi
    ax2.plot(xi, yi_inv, color='none')
    ymax = plotutils.roundup(yi_inv.max(), plotutils.magnitude(yi_inv.max()))
    min_mag = plotutils.magnitude(yi_inv.min())
    ymin = (yi_inv.min() // 10**(min_mag)) * 10**(min_mag)
    ax2.set_ylim((ymax, ymin))
    ax2.grid(False)

    # Plot fit of all points
    fname = 'experiments_density_sgls-perc'
    ext = 'eps'
    if path_plot is not None:
        file_path = _join(path_plot, '{}.{}'.format(fname, ext))
        [ax.set_aspect('equal') for ax in [ax1, ax2]]
        plt.savefig(file_path, format=ext, bbox_inches='tight')

        latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                               dpi=dpi)
    plt.show()

    return None


def plot_learning_curves(m, path_plot=None):
    '''Plot learning curve of train and test accuracy/error

    Args
    ----
    m: dict()
        Dict of `train` and `valid` dicts with `err`, `loss`, and `acc`
    '''
    import matplotlib.pyplot as plt
    import numpy
    import scipy.optimize
    import seaborn
    from pyotelem.plots import plotutils

    from . import latex
    from . import utils

    seaborn.set_style('whitegrid')

    linear = lambda x, m, c: m*x + c

    fig, ax = plt.subplots()

    ax.plot(m['train']['err'], label='Train', linewidth=_linewidth)
    ax.plot(m['valid']['err'], label='Validation', linewidth=_linewidth)
    ax.set_ylabel('Cross-entropy error')
    ax.set_xlabel('No. of samples')

    # Fit a linear regression througn n_train to get ticks at regular locs
    x = m['n_train'].values.astype(float)
    y = numpy.arange(len(m['n_train']))
    popt, pcov = scipy.optimize.curve_fit(linear, x, y)

    # Create even tick labels given order of maximum value in `x`
    x_mag = plotutils.magnitude(x.max())
    x_max = plotutils.roundup(int(x.max()), x_mag)+10**x_mag
    new_labels = numpy.arange(0, x_max, 10**x_mag)
    new_ticks = linear(new_labels, *popt)
    ax.set_xlim((new_ticks[0], new_ticks[-1]))
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels.astype(int))

    ax.legend()

    # Save plot
    fname = 'learning_curve'
    ext = 'eps'
    if path_plot is not None:
        file_path = _join(path_plot, '{}.{}'.format(fname, ext))
        plt.savefig(file_path, format=ext, bbox_inches='tight')

    plt.show()

    return None


def sgl_density(exp_name, sgls, max_depth=20, textstr='', path_plot=None):
    '''Plot density of subglides over time for whole exp, des, and asc'''
    import seaborn
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    import numpy
    from pyotelem.plots import plotutils

    from . import utils

    # TODO add A/ B, bottom left
    # TODO move textbox bottom right
    # TODO set limits for density the same
    # TODO Update xaxis, units time elapsed
    # TODO save as svg

    # Make jointplots as subplots
    # http://stackoverflow.com/a/35044845/943773

    seaborn.set_style('white')

    fig = plt.figure()

    # time, mid between start and finish
    sgl_x = sgls['start_idx'] + ((sgls['stop_idx']-sgls['start_idx'])/2)

    # depth, calc avg over sgl time
    sgl_y = sgls['mean_depth']

    g = seaborn.jointplot(x=sgl_x, y=sgl_y, kind='hex', stat_func=None)

    # Set depth limit of plot and invert depth values
    g.fig.axes[0].set_ylim(0, max_depth)
    g.fig.axes[0].invert_yaxis()

    # Convert axes labels to experiment duration in hours/min
    g.set_axis_labels(xlabel='Experiment duration ($min \, sec$)',
                      ylabel='Depth ($m$)')
    mf2 = FuncFormatter(plotutils.nsamples_to_hourmin)
    g.fig.axes[0].xaxis.set_major_formatter(mf2)

    ## TODO add colorbar
    ## http://stackoverflow.com/a/29909033/943773
    #cax = g.fig.add_axes([1, 0.35, 0.01, 0.2])
    #plt.colorbar(cax=cax)

    # Add text annotation top left if `textstr` passed
    if textstr:
        props = dict(facecolor='none', edgecolor='none', alpha=0.1)
        g.fig.axes[0].text(0.02, 0.05, textstr, transform=g.fig.axes[0].transAxes,
                           fontsize=14, verticalalignment='top', bbox=props)

    if path_plot:
        ext = 'eps'
        fname_plot_sgl = 'heatmap_{}.{}'.format(exp_name, ext)
        file_plot_sgl = _join(path_plot, fname_plot_sgl)
        g.savefig(filename=file_plot_sgl)

    plt.close('all')

    return None


def filt_paths(path_project, cfg_ann):
    import numpy
    import os

    from ..config import paths, fnames
    from .. import utils

    # Get list of paths filtered by parameters in `cfg_ann['data']`
    path_tag = _join(path_project, paths['tag'])
    path_glide = _join(path_project, paths['glide'])
    data_paths = list()
    exp_names = list()
    for p in os.listdir(path_tag):
        path_exp = _join(path_tag, p)
        if os.path.isdir(path_exp):
            # Concatenate data path
            path_glide_data = _join(path_project, path_glide, p)
            path_subdir = utils.get_subdir(path_glide_data, cfg_ann['data'])
            data_paths.append(_join(path_glide_data, path_subdir))
            exp_names.append(p)

    sort_ind = numpy.argsort(data_paths)
    data_paths = numpy.array(data_paths)[sort_ind]
    exp_names = numpy.array(exp_names)[sort_ind]

    return data_paths, exp_names


def plot_sgl_histos(path_project, cfg_ann, path_plot):
    import matplotlib.pyplot as plt
    import numpy
    from os.path import join as _join
    import pandas
    from pyotelem.plots import plotutils
    from smartmove.config import paths, fnames
    from smartmove.visuals.figures import filt_paths
    import string
    import seaborn

    seaborn.reset_orig()
    seaborn.set_style('whitegrid', {'axes.linewidth':0.5,})

    fig, axes = plt.subplots(1, 6, figsize=(10,5), sharey=True, sharex=True)

    data_paths, exp_names = filt_paths(path_project, cfg_ann)

    net_depth = 18
    delta = 0.5
    bins = numpy.arange(0, net_depth+delta, delta)

    mod_dict = {'4weights':'4 Weights',
                '4floats':'4 Floats',
                '4neutralblocks':'4 Neutrals',
                '2neutral':'2 Neutrals'}

    exp_ind = [8, 13, 6, 12, 9, 10]
    textstr = ''
    for i in range(len(exp_ind)):
        exp_name = exp_names[exp_ind[i]]
        p = data_paths[exp_ind[i]]

        sgls = pandas.read_pickle(_join(p, fnames['glide']['sgls']))
        mask_sgls = pandas.read_pickle(_join(p, fnames['glide']['mask_sgls_filt']))
        sgls = sgls[mask_sgls]

        # time, mid between start and finish
        sgl_x = sgls['start_idx'] + ((sgls['stop_idx']-sgls['start_idx'])/2)

        # depth, calc avg over sgl time
        sgl_y = sgls['mean_depth']

        axes[i].hist(sgl_y, bins=bins, orientation='horizontal')

        # Generate title for subplot
        year   = exp_name[:4]
        month  = exp_name[4:6]
        day    = exp_name[6:8]
        animal = exp_name.split('_')[3].capitalize()
        mod    = ' '.join(exp_name.split('_')[4:])
        fmt = '{}, {}\n'
        textstr += fmt.format(year, mod_dict[mod.lower()])

    plotutils.add_alpha_labels(axes, xpos=0.70, ypos=0.07, color='black',
                               fontsize=12, facecolor='white',
                               edgecolor='white')

    labels = list(string.ascii_uppercase)[:len(exp_ind)]
    labels = ''.join(['{} :\n'.format(l) for l in labels])
    exp_ids = ''.join(['{:2d},\n'.format(i+1) for i in exp_ind])

    axes[-1].text(1.2, 1,  labels, va='top', ha='left', transform=axes[-1].transAxes)
    axes[-1].text(1.75, 1, exp_ids, va='top', ha='right', transform=axes[-1].transAxes)
    axes[-1].text(1.8, 1, textstr, va='top', ha='left', transform=axes[-1].transAxes)

    axes[0].set_yticks(range(0, net_depth+1))
    axes[0].set_yticklabels([str(i) if i%2==0 else '' for i in range(net_depth+1)])
    axes[0].set_xticks([0, 50, 100, 150])
    axes[0].set_xticklabels(['0', '', '100', ''])

    fig.text(0.4, 0.04, 'No. sub-glides', ha='center')
    axes[0].set_ylabel('Depth ($m$)')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(right=0.7, wspace=0.2, hspace=0.2)

    # Save plot
    fname = 'subglide_histograms'
    ext = 'eps'
    if path_plot is not None:
        file_path = _join(path_plot, '{}.{}'.format(fname, ext))
        plt.savefig(file_path, format=ext, bbox_inches='tight')

    plt.show()

    return None


def plot_sgl_highlight(path_project, cfg_ann, path_plot, clip_x=True):
    import numpy
    from pandas import read_pickle
    from pyotelem.plots import plotglides
    import seaborn

    from ..config import paths, fnames
    from .. import utils

    seaborn.set_context('paper')
    seaborn.set_style('white', {'axes.linewidth':0.1, 'xtick.color':'black',
        'xtick.major.size':4.0,'xtick.direction':'out'})

    data_paths, exp_names = filt_paths(path_project, cfg_ann)

    p = data_paths[0]
    e = exp_names[0]

    fname_tag = fnames['tag']['data'].format(e)
    tag = read_pickle(_join(p, fname_tag))
    mask_tag = read_pickle(_join(p, fnames['glide']['mask_tag']))
    mask_tag_filt = read_pickle(_join(p, fnames['glide']['mask_tag_filt']))


    sgls = read_pickle(_join(p, fnames['glide']['sgls']))
    mask_sgls_filt = read_pickle(_join(p, fnames['glide']['mask_sgls_filt']))

    # TODO hardcoded, would be insanely tricky to autofind a good example of
    # subglides, manually found and pass correct data indices for location
    mask_exp = mask_tag['exp']
    depths = tag['depth'].values
    pitch_lf = tag['p_lf'].values
    roll_lf = tag['r_lf'].values
    heading_lf = tag['h_lf'].values
    plotglides.plot_sgls(mask_exp, depths, mask_tag_filt, sgls, mask_sgls_filt,
                         pitch_lf, roll_lf, heading_lf,
                         idx_start=259000, idx_end=260800, path_plot=path_plot,
                         clip_x=clip_x)
    return None


def studyarea(path_plot):

    def plot_coords(ax, proj, lon, lat, marker='o', color='black'):
        import cartopy.crs as ccrs
        import numpy

        lon = numpy.asarray(lon)
        lat = numpy.asarray(lat)
        points = proj.transform_points(ccrs.Geodetic(), lon, lat)

        ax.scatter(points[:,0], points[:,1], marker='.' , color=color)

        return ax

    import cartopy.crs as ccrs
    from io import BytesIO
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy
    import os
    from owslib.wms import WebMapService
    import PIL
    import seaborn

    from . import maps

    # Plotting configuration
    rcParams.update({'figure.autolayout': True})
    seaborn.set_context('paper')
    sns_config = {'axes.linewidth':0.1,
                  'xtick.color':'black', 'xtick.major.size':4.0, 'xtick.direction':'out',
                  'ytick.color':'black', 'ytick.major.size':4.0, 'ytick.direction':'out'}
    seaborn.set_style('white', sns_config)

    # Bounding box
    lon0 = 18.565148 # BL
    lat0 = 69.618893 # BL
    lon1 = 19.071205 # TR
    lat1 = 69.768302 # TR

    layer = 'Vannflate'
    srs = 'EPSG:32634'

    # Create wms object for handling WMS calls
    url = 'http://openwms.statkart.no/skwms1/wms.topo3?'
    wms = WebMapService(url)

    # Project bounding box
    bbox = maps.project_bbox(srs, lon0, lat0, lon1, lat1)

    owslib_img = maps.get_wms_png(wms, bbox, layer, srs, width=1600, transparent=True)

    # Convert binary png data to PIL Image data

    color_land = (131, 196, 146, 255)#(230, 230, 230, 255) # Hex color #f2f2f2
    color_water = (237, 239, 242, 255) # Hex color #cccccc

    # https://stackoverflow.com/a/43514640/943773

    img = PIL.Image.open(BytesIO(owslib_img.read()))
    land = PIL.Image.new('RGB', img.size, color_land)
    water = PIL.Image.new('RGBA', img.size, color_water)
    land.paste(water, mask=img.split()[3])

    proj = ccrs.epsg(srs.split(':')[1])

    # Create project axis and set extent to bounding box
    ax = plt.axes(projection=proj)
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    ax.set_extent(extent, proj)

    # Plot projected image
    ax.imshow(land, origin='upper', extent=extent, transform=proj)

    # Project and plot Tromsø and study area positions
    wgs84 = ccrs.Geodetic()

    # Tromso
    plot_coords(ax, proj, 18.955364, 69.649197, marker='o', color='#262626')
    # Study area
    plot_coords(ax, proj, 18.65125, 69.69942, marker='^', color='#a35d61')
    # CTD
    plot_coords(ax, proj, 18.65362, 69.70214, marker='o', color='#42a4f4')

    # Create WGS84 axes labels with WGS84 axes positions
    ticks_lon, xlabels = maps.map_ticks(lon0, lon1, 4)
    ticks_lat, ylabels = maps.map_ticks(lat0, lat1, 4)

    # Create list of lons/lats at axis edge for projecting ticks_lon/lats
    lat_iter = numpy.array([lat0]*len(ticks_lon))

    # Project ticks_lon/lats and set to axis labels
    xpos = proj.transform_points(ccrs.Geodetic(), ticks_lon, lat_iter)[:,0]
    ypos = proj.transform_points(ccrs.Geodetic(), ticks_lon, ticks_lat)[:,1]

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, ha='right', rotation=45)
    ax.xaxis.tick_bottom()

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.yaxis.tick_right()

    # Turn off grid for seaborn styling
    ax.grid(False)

    # Write image data if `path_plot` passed
    if path_plot:
        import pickle
        # Save png of WMS data
        filename_png = os.path.join(path_plot, '{}.png'.format(layer))
        with open(filename_png, 'wb') as f:
            f.write(owslib_img.read())
        # Convert png to GEOTIFF data
        maps.png2geotiff(filename_png, srs, bbox)
        # Save pickle of `PIL` image data
        file_pickle_img = os.path.join(path_plot, '{}_img.p'.format(layer))
        pickle.dump(img, open(file_pickle_img, 'wb'))
        # Save Cartopy plot
        file_plot = os.path.join(path_plot, 'study_area.eps')
        plt.savefig(file_plot, dpi=600)

    plt.show()

    return None


def plot_ann_performance(cfg_ann, results_tune, path_plot):
    import matplotlib.pyplot as plt
    import seaborn

    import pandas

    colors = seaborn.color_palette()
    seaborn.set_context('notebook')

    # Convert results_tune configs + acc to pandas dataframe
    params = ['hidden_nodes','hidden_layers']
    df_cfgs = pandas.DataFrame(index=range(len(results_tune)),
            columns=params+['acc'])

    for i in range(len(results_tune)):
        for p in params:
            df_cfgs.iloc[i][p] = results_tune['config'][i][p]
        df_cfgs.iloc[i]['acc'] = results_tune.iloc[i]['accuracy']

    df_cfgs = df_cfgs.apply(pandas.to_numeric, errors='ignore')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

    sub_width = list()
    for ax, p in zip((ax1, ax2), params):
        df_cfgs.boxplot(ax=ax, column='acc', by=p)
        ax.set_xlabel(p.title().replace('_', ' '))
        ax.set_title('')
        sub_width = ax.get_xlim()[1]
        ax.grid(False)

    #ax6.axis('off')

    fig.texts = list()
    ax1.set_ylabel('Accuracy (%)')

    if path_plot:
        ext = 'eps'
        fname_plot_sgl = 'hyperparameter_accuracy.{}'.format(ext)
        file_plot_sgl = _join(path_plot, fname_plot_sgl)
        plt.savefig(filename=file_plot_sgl)

    plt.show()

    return None


def make_all(path_project, path_analysis):
    import os
    import pandas
    import yamlord

    from ..ann import pre
    from ..ann import utils_ann
    from ..config import paths, fnames
    from . import utils
    from . import latex

    # Create output path for plots
    path_plot = _join(path_project, 'paper/figures')
    os.makedirs(path_plot, exist_ok=True)

    # Define path to ANN analysis data
    path_output = _join(path_project, paths['ann'], path_analysis)

    # Load ANN configuration
    file_cfg_ann = _join(path_output, fnames['cfg']['ann'])
    cfg_ann = yamlord.read_yaml(file_cfg_ann)

    # Load experiment data
    file_field = _join(path_project, paths['csv'], fnames['csv']['field'])
    file_isotope = _join(path_project, paths['csv'], fnames['csv']['isotope'])
    field, isotope = pre.add_rhomod(file_field, file_isotope)

    # Compile experiments adding columns necessary for tables/figures
    exps_all = utils.compile_exp_data(path_project, field, cfg_ann)

    # Plot SGLs against `rho_mod`
    plot_sgls_tmbd(exps_all, path_plot=path_plot)

    # Plot confusion matrix
    # TODO CM key named validation in smartmove, should change with refactoring
    file_cms_data = _join(path_output, fnames['ann']['cms_tune'])
    cms_tune = pandas.read_pickle(file_cms_data)
    cm_test = cms_tune['validation']['cm']
    targets = cms_tune['targets']
    valid_targets = cms_tune['validation']['targets']
    target_ids = [i for i in range(len(targets)) if targets[i] in valid_targets]
    tick_labels = list()
    xlabel = r'Predicted $\rho_{mod}$ bin ($kg \cdot m^{-3}$)'
    ylabel = r'Observed $\rho_{mod}$ bin ($kg \cdot m^{-3})$'
    for i in range(len(valid_targets)):
        tick_labels.append('{}: {}<='.format(target_ids[i], valid_targets[i]))
    utils_ann.plot_confusion_matrix(cm_test, tick_labels, xlabel=xlabel,
                                    ylabel=ylabel, normalize=False, title='',
                                    cmap=None, xlabel_rotation=45,
                                    path_plot=path_plot)

    # Plot learning curve of ANN
    file_results_data = _join(path_output, fnames['ann']['dataset'])
    results_dataset = pandas.read_pickle(file_results_data)
    m = utils.last_monitors(results_dataset)
    plot_learning_curves(m, path_plot)

    # Plot study area plot
    studyarea(path_plot)

    # Plot hyperparameter accuracy performance
    file_results_tune = _join(path_output, fnames['ann']['tune'])
    results_tune = pandas.read_pickle(file_results_tune)
    plot_ann_performance(cfg_ann, results_tune, path_plot)

    # Plot subglide heatmaps
    plot_sgl_histos(path_project, cfg_ann, path_plot)

    # Plot example subglide plot
    plot_sgl_highlight(path_project, cfg_ann, path_plot, clip_x=True)

    # Convert all `.eps` images to `.png`
    for fig in os.listdir(path_plot):
        if fig.endswith('.eps'):
            # Convert eps to png
            fname = os.path.splitext(fig)[0]
            latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                                   dpi='300')
            utils.crop_png(_join(path_plot, fname+'.png'))

    return None
