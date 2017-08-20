from os.path import join as _join

def plot_sgls_tmbd(exps_all, path_plot=None):
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    import os
    import seaborn
    import scipy.optimize

    from . import latex
    from pyotelem.plots.plotutils import add_alpha_labels

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
                if d <= dist:
                    neighbors.append(j)

        # Set rotating angle of annotation position for neighboring points
        for i in neighbors:
            theta[i] = next(theta_gen)

        # Caclulate positions of annotations at `theta` and `dist`
        xa = x + numpy.sin(theta)*dist
        ya = y + numpy.cos(theta)*dist

        return xa, ya

    # Plot colored, sized points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))

    delta = exps_all['rho_mod'] - exps_all['density_kgm3']
    delta_norm = unit_normalize(delta) + 1
    udeltas = numpy.unique(delta)
    sizes = numpy.unique(delta_norm)**2*30

    axes = [ax1, ax2]
    keys = ['perc_des', 'perc_asc']
    animals = numpy.unique(exps_all['animal'])
    axes[0].set_ylabel('Percent of total sub-glides by dive phase $(\%)$')

    from collections import OrderedDict
    markers = OrderedDict((
        #('.', 'point'),
        #(',', 'pixel'),
        ('o', 'circle'),
        ('v', 'triangle_down'),
        ('^', 'triangle_up'),
        ('<', 'triangle_left'),
        ('>', 'triangle_right'),
        #('1', 'tri_down'),
        #('2', 'tri_up'),
        #('3', 'tri_left'),
        #('4', 'tri_right'),
        #('8', 'octagon'),
        ('s', 'square'),
        ('p', 'pentagon'),
        #('*', 'star'),
        ('h', 'hexagon1'),
        #('H', 'hexagon2'),
        ('+', 'plus'),
        ('D', 'diamond'),
        ('d', 'thin_diamond'),
        #('|', 'vline'),
        #('_', 'hline')
        ))
    markers = list(markers.keys())
    arrowprops = dict(arrowstyle='-', facecolor='black')
    label_dist = 3

    for ax, key in zip(axes, keys):
        c = 0
        ax.set_xlabel(r'$\rho_{mod} \; (kg \, m^{-3})$')

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
                delta = x - exps_all['density_kgm3'][ind[i]]
                m = numpy.where(delta==udeltas)[0][0]

                # Plot positive density changed experiments
                ax.scatter(x, y, label=a, marker=markers[m], color=colors[c], s=60)

                text = ax.annotate(exp_id, xy=(x,y), xytext=(xa[i], ya[i]),
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

        ## Add text box with equation and r-squared
        #anchored_text = AnchoredText('{}\n{}'.format(str_eqn, str_r2), loc=3)
        #ax.add_artist(anchored_text)

    # Add alpha labels
    xpos = [0.05, 0.9]
    ypos = [0.95, 0.95]
    axes = add_alpha_labels([ax1, ax2], xpos=xpos, ypos=ypos, color='black')

    # Manually create legend using matplotlib artist proxies
    # dot size legend
    prox_handles = list()
    prox_labels = list()
    for i in range(len(udeltas)):
        sc = plt.scatter([], [], marker=markers[i], color='black', s=60)
        prox_handles.append(sc)
        prox_labels.append('{:5.2f}'.format(udeltas[i]))

    # space between size legend and marker legend
    prox_handles.append(plt.scatter([], [], s=0))
    prox_labels.append('')

    # color dot animal legend
    for i, a in enumerate(animals):
        prox_handles.append(plt.scatter([], [], color=colors[i], s=20))
        prox_labels.append(a)

    # fit to points legend
    prox_handles.append((plt.plot([], [], color='grey', linestyle='dashed'))[0])
    prox_labels.append('Linear fit\n{}'.format(str_r2))

    # Plot regular artists
    plt.legend(handles=prox_handles, labels=prox_labels,
               bbox_to_anchor=(1, 1), loc='upper left',
               title=r'$\Delta \, \rho_{mod}$'+'\n'+r'$(kg \, m^{-3})$')

    # Plot fit of all points
    fname = 'experiments_density_sgls-perc'
    ext = 'eps'
    if path_plot is not None:
        file_path = _join(path_plot, '{}.{}'.format(fname, ext))
        [ax.set_aspect('equal') for ax in [ax1, ax2]]
        plt.savefig(file_path, format=ext, bbox_inches='tight')

        # TODO perhaps move out of latex to image utils repo
        latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                               dpi='600')
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
    ax.set_xlabel('Number of samples')

    # Fit a linear regression througn n_train to get ticks at regular locs
    x = m['n_train'].values.astype(float)
    y = numpy.arange(len(m['n_train']))
    popt, pcov = scipy.optimize.curve_fit(linear, x, y)

    # Create even tick labels given order of maximum value in `x`
    x_mag = utils.magnitude(x.max())
    x_max = utils.roundup(int(x.max()), x_mag)+10**x_mag
    new_labels = numpy.arange(0, x_max, 10**x_mag)
    new_ticks = linear(new_labels, *popt)
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

    file_results_data = _join(path_output, fnames['ann']['dataset'])
    results_dataset = pandas.read_pickle(file_results_data)
    m = utils.last_monitors(results_dataset)
    plot_learning_curves(m, path_plot)

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

    for fig in os.listdir(path_plot):
        if fig.endswith('.eps'):
            # Convert eps to png
            fname = os.path.splitext(fig)[0]
            latex.utils.pdf_to_img(path_plot, fname, in_ext='eps', out_ext='png',
                                   dpi='300')
    return None
