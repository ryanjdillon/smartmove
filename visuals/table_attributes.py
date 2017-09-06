
def get_all():
    from collections import OrderedDict

    def add_attrs(attrs, name, title, caption, notes=None, adjustwidth=True):
        '''Add table attributes to dictionary'''

        attrs[name] = OrderedDict()

        attrs[name]['title']          = title
        attrs[name]['caption']        = caption
        attrs[name]['notes']          = notes
        attrs[name]['adjustwidth']    = adjustwidth
        attrs[name]['extrarowheight'] = 3

        return attrs

    attrs = OrderedDict()

    # table_ann_feature_descr 1
    title = '''ANN input features obtained from various sensors attached to seals in the field experiment.'''
    caption = ''
    attrs = add_attrs(attrs, 'table_ann_feature_descr', title, caption)


    # table_ann_target_descr 2
    title = r'''ANN target value bins with $\rho\textsubscript{mod}$ and lipid
percent ranges.'''
    caption = r'''Calculated $\rho\textsubscript{mod}$ values are assigned to a bin such that $bin\textsubscript{min} >= \rho\textsubscript{mod} > bin\textsubscript{max}$. Lipid percent ranges were calculated from the corresponding $\rho\textsubscript{mod}$ ranges using the equations presented in \cite{biuw_blubber_2003}. Bins prepended with an asterisk were those within which the ANN made classifications (see Table~\ref{table_ann_feature_stats}).'''
    attrs = add_attrs(attrs, 'table_ann_target_descr', title, caption,
            adjustwidth=False)


    # table_ann_params 3
    title = '''Neural network configuration attributes.'''
    caption = '''During the network tuning an ANN was trained and tested for all permutations of possible values for each attribute. Bold faced attributes produced the network yielding the highest accuracy. See Appendix 1 for a glossary of terms.'''
    attrs = add_attrs(attrs, 'table_ann_params', title, caption,
            adjustwidth=False)


    # table_exps 4
    title = 'Field experiments - dive and glide event summary with the added effect of body density blocks.'
    caption = r'''Glides were identified during descent and ascent dive phases and split into sub-glides (SGL). Body densities were derived from the isotope analyses presented in Table~\ref{table_isotope}, and total modified body densities ($\rho\textsubscript{mod}$) were calculated from those with the additional effect of the attached blocks.'''
    attrs = add_attrs(attrs, 'table_exps', title, caption)


    # table_isotope 5
    title = 'Body composition and density experiments - percent body composition summary, calculated using the tritiated water method.'
    caption = ''
    attrs = add_attrs(attrs, 'table_isotope', title, caption)


    # table_ann_feature_stats 6
    title = 'Input feature value ranges, mean, and standard deviation'
    caption = ''
    attrs = add_attrs(attrs, 'table_ann_feature_stats', title, caption,
            adjustwidth=False)


    # table_target_stats 7
    title = '''Number of sub-glide events per target value bin with percentage of total number of compiled target values.'''
    caption = '''There were a total of 9985 target values from all compiled sub-glides. The percentage of compiled data represents the percentage of all sub-glides in the compiled dataset classified into the associated bin number.'''
    attrs = add_attrs(attrs, 'table_ann_target_stats', title, caption,
            adjustwidth=False)

    return attrs
