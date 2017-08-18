
def make(path_project, path_analysis):
    from . import figures
    from . import tables

    figures.make_all(path_project, path_analysis)
    tables.make_all(path_project, path_analysis)

    return None
