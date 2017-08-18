
def compare_tags(path_project, param, singles=False):
    '''Compare data accross data files in acclerometer data folder
    '''
    import matplotlib.pyplot as plt
    import os
    import pandas
    import seaborn

    from ..config import paths, fnames

    path_tag = os.path.join(path_project, paths['tag'])

    c = 0
    # Ugly color palette for us color-blind people, could be improved
    dir_list = sorted(os.listdir(path_tag))
    colors = seaborn.color_palette("Paired", len(dir_list))
    for d in dir_list:
        if os.path.isdir(d):
            for f in os.listdir(os.path.join(path_tag, d)):
                if f.startswith('pydata'):
                    fname = os.path.join(path_tag, d, f)
                    data = pandas.read_pickle(fname)
                    plt.plot(data[param], label=f, color=colors[c])
                    c += 1
                    if singles:
                        plt.legend()
                        plt.show()
    if not singles:
        plt.legend()
        plt.show()

    return None
