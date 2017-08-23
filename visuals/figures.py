from os.path import join as _join

def plot_sgls_tmbd(exps_all, path_plot=None, dpi=300):
    import numpy
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    import os
    import seaborn
    import scipy.optimize

    from . import latex
    from pyotelem.plots.plotutils import add_alpha_labels

    seaborn.set(style="whitegrid", color_codes=True)
    colors = seaborn.color_palette()

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

    # Plot colored, sized points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))

    deltas = exps_all['rho_mod'] - exps_all['density_kgm3']
    udeltas = numpy.unique(deltas)

    axes = [ax1, ax2]
    keys = ['perc_des', 'perc_asc']
    animals = numpy.unique(exps_all['animal'])
    axes[0].set_ylabel('Percent of total sub-glides by dive phase $(\%)$')

    arrowprops = dict(arrowstyle='wedge', facecolor='black')
    label_dist = 3

    for ax, key in zip(axes, keys):
        c = 0
        ax.set_xlabel(r'$\rho_{mod} \; (kg \cdot m^{-3})$')

        # Get all points for des/asc for generating linear fits
        x_all = exps_all['rho_mod'].values
        y_all = exps_all[key].values

        xa_all, ya_all = annotate_coords(x_all, y_all, label_dist)

        for a in animals:

            # Get density and percent des/asc values for animal
            amask = exps_all['animal'] == a
            labels = exps_all['id'][amask]

            x = exps_all['rho_mod'][amask].values
            y = exps_all[key][amask].values
            xa = xa_all[amask]
            ya = ya_all[amask]

            ind = numpy.where(amask)[0]
            for i in range(len(ind)):
                x = exps_all['rho_mod'][ind[i]]
                y = exps_all[key][ind[i]]
                exp_id = exps_all['id'][ind[i]]
                subdelta = x - exps_all['density_kgm3'][ind[i]]
                m = numpy.where(subdelta==udeltas)[0][0]

                text = ax.annotate(exp_id, xy=(x,y), xytext=(xa[i], ya[i]),
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

        ax.plot(xi, yi, label='fit', color='grey', linestyle='dashed')

        # Expand ylimits to allow for annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin-5.0, ymax+5.0))

    # Add alpha labels
    xpos = [0.05, 0.9]
    ypos = [0.95, 0.95]
    axes = add_alpha_labels([ax1, ax2], xpos=xpos, ypos=ypos, color='black')

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
                     bbox_to_anchor=(1, 1), loc='upper left')

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

    ax2.annotate(table_title, xy=(1.06, 0.43), xycoords='axes fraction',
                 fontsize=12)
    ax2.annotate(table_id, xy=(1.15, 0.0), xycoords='axes fraction',
                 horizontalalignment='right', fontsize=10)
    ax2.annotate(table_rho, xy=(1.16, 0.0), xycoords='axes fraction',
                 fontsize=10)

    # Plot fit of all points
    fname = 'experiments_density_sgls-perc'
    ext = 'eps'
    if path_plot is not None:
        file_path = _join(path_plot, '{}.{}'.format(fname, ext))
        [ax.set_aspect('equal') for ax in [ax1, ax2]]
        plt.savefig(file_path, format=ext, bbox_inches='tight')

        # TODO perhaps move out of latex to image utils repo
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

    from . import latex
    from . import utils

    seaborn.set_style('whitegrid')

    linear = lambda x, m, c: m*x + c

    fig, ax = plt.subplots()

    ax.plot(m['train']['err'], label='Train')
    ax.plot(m['valid']['err'], label='Validation')
    ax.set_ylabel('Cross-entropy error')
    ax.set_xlabel('No. of samples')

    # Fit a linear regression througn n_train to get ticks at regular locs
    x = m['n_train'].values.astype(float)
    y = numpy.arange(len(m['n_train']))
    popt, pcov = scipy.optimize.curve_fit(linear, x, y)

    # Create even tick labels given order of maximum value in `x`
    x_mag = utils.magnitude(x.max())
    x_max = utils.roundup(int(x.max()), x_mag)+10**x_mag
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

        # TODO perhaps move out of latex to image utils repo
        latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                               dpi='600')
    plt.show()

    return None


def sgl_density(exp_name, sgls, max_depth=20, textstr='', path_plot=None):
    '''Plot density of subglides over time for whole exp, des, and asc'''
    import seaborn
    import matplotlib.pyplot as plt
    import numpy

    from . import utils

    # TODO add A/ B, bottom left
    # TODO move textbox bottom right
    # TODO set limits for density the same
    # TODO Update xaxis, units time elapsed
    # TODO save as svg

    # Make jointplots as subplots
    # http://stackoverflow.com/a/35044845/943773

    seaborn.set(style='white')

    fig = plt.figure()

    # time, mid between start and finish
    sgl_x = sgls['start_idx'] + ((sgls['stop_idx']-sgls['start_idx'])/2)

    # depth, calc avg over sgl time
    sgl_y = sgls['mean_depth']

    g = seaborn.jointplot(x=sgl_x, y=sgl_y, kind='hex', stat_func=None)

    g.fig.axes[0].set_ylim(0, max_depth)
    g.fig.axes[0].invert_yaxis()
    labels = g.fig.axes[0].get_xticks()
    labels = (numpy.array(labels)-labels[0])/16.0
    labels = [utils.hourmin(n) for n in labels]
    g.fig.axes[0].set_xticklabels(labels, rotation=45)
    g.set_axis_labels(xlabel='Experiment duration', ylabel='Depth (m)')

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


def plot_all_sgl_densities(path_project, cfg_ann, path_plot):
    import numpy
    import os
    import pandas

    from .. import utils
    from ..config import paths, fnames

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
    exp_id = 1
    for p, exp_name in zip(data_paths, exp_names):
        fname_tag = fnames['tag']['data'].format(exp_name)
        tag = pandas.read_pickle(_join(p, fname_tag))
        sgls = pandas.read_pickle(_join(p, fnames['glide']['sgls']))
        fname_mask_sgls_filt = fnames['glide']['mask_sgls_filt']
        mask_sgls = pandas.read_pickle(_join(p, fname_mask_sgls_filt))

        year   = exp_name[:4]
        month  = exp_name[4:6]
        day    = exp_name[6:8]
        animal = exp_name.split('_')[3].capitalize()
        mod    = ' '.join(exp_name.split('_')[4:])
        fmt = '{}. {}-{}-{} {} {}'
        textstr = fmt.format(exp_id, year, month, day, animal, mod)

        sgl_density(exp_name, sgls[mask_sgls], max_depth=20, textstr=textstr,
                    path_plot=path_plot)
        exp_id += 1

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
    targets = cms_tune['validation']['targets']
    tick_labels = list()
    xlabel = r'Predicted $\rho_{mod}$ bin ($kg \cdot m^{-3}$)'
    ylabel = r'Observed $\rho_{mod}$ bin ($kg \cdot m^{-3})$'
    for i in range(len(targets)):
        tick_labels.append('{}<='.format(targets[i]))
    utils_ann.plot_confusion_matrix(cm_test, tick_labels, xlabel=xlabel,
                                    ylabel=ylabel, normalize=False, title='',
                                    cmap=None, xlabel_rotation=45,
                                    path_plot=path_plot)

    # Plot learning curve of ANN
    file_results_data = _join(path_output, fnames['ann']['dataset'])
    results_dataset = pandas.read_pickle(file_results_data)
    m = utils.last_monitors(results_dataset)
    plot_learning_curves(m, path_plot)

    # Plot subglide heatmaps
    plot_all_sgl_densities(path_project, cfg_ann, path_plot)

    # Convert all `.eps` images to `.png`
    for fig in os.listdir(path_plot):
        if fig.endswith('.eps'):
            # Convert eps to png
            fname = os.path.splitext(fig)[0]
            latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                                   dpi='300')
    return None
