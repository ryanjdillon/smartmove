
def compare_tags(path_project, cfg_experiments, param, singles=False):
    '''Compare data accross data files in acclerometer data folder
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy
    import os
    import pandas
    #import seaborn
    import yamlord

    from ..config import paths, fnames
    from . import utils

    cmap = matplotlib.cm.get_cmap('tab20b')

    path_tags = os.path.join(path_project, paths['tag'])

    c = 0
    # Ugly color palette for us color-blind people, could be improved
    dir_list = sorted(os.listdir(path_tags))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=(len(dir_list)-1))
    colors = [cmap(norm(i)) for i in range(len(dir_list))]

    min_val = 9e13
    max_val = 0
    min_exp = ''
    max_exp = ''
    idx_val = 9e13
    idx_exp = ''
    for d in dir_list:
        path_tag = os.path.join(path_tags, d)
        if os.path.isdir(path_tag):
            for fname in os.listdir(path_tag):
                if fname.startswith('pydata'):
                    print(c, fname)
                    fname = os.path.join(path_tag, fname)
                    data = pandas.read_pickle(fname)

                    start_idx = cfg_experiments[d]['start_idx']
                    stop_idx = cfg_experiments[d]['stop_idx']

                    mask_tag = numpy.zeros(len(data), dtype=bool)
                    mask_tag[start_idx:stop_idx] = 1
                    data = data[param].iloc[mask_tag]

                    if data.min() < min_val:
                        min_val = data.min()
                        min_exp = d
                    if data.max() > max_val:
                        max_val = data.max()
                        max_exp = d
                    if data.index[0] < idx_val:
                        idx_val = data.index[0]
                        idx_exp = d

                    plt.plot(data, label=d, color=colors[c])

                    if singles:
                        plt.legend()
                        plt.show()
            c += 1

    print('Min {}: {} {}'.format(param, min_exp, min_val))
    print('Max {}: {} {}'.format(param, max_exp, max_val))
    print('Min index: {} {}'.format(idx_exp, idx_val))

    if not singles:
        plt.legend()
        plt.show()

    return None
