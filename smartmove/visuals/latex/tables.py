def write_table(out_root, fname, data, cols, headers, adjustwidth=False,
        tiny=False, title='', caption='', centering=True, extrarowheight=0,
        label='', notes='', dpi=300):
    '''Create LaTeX table and write and compile to output directory

    Args
    ----
    out_root: str
        path where table files should be written
    fname: str
        name of files without extension
    data: pandas dataframe
        dataframe with containing columns in `write_cols`
    cols: OrderedDict
        key, value pairs of column names and format string
    headers: iterable
        list of lists containing string names of header columns. First should
        be names, second units, etc.
    adjustwidth: bool
        adjust table to use full width of the page
    tiny: bool
        use `tiny` LaTeX command
    title: str
        Bolded title of table
    caption: str
        Caption explaining table
    centering: bool
        Center table on page
    extrarowheight: int
        number of points to increase table row height by
    label: str
        label for linking to tabel in LaTeX
    notes: str
        Notes regarding table which appear below table

    Returns
    -------
    table: str
        Concatenated LaTeX table string
    '''
    import os

    from . import utils

    # Write table to file
    head   = __create_header(headers ,title, caption, adjustwidth,
                             tiny, centering, extrarowheight)
    body   = __create_body(data, cols)
    footer = __create_footer(label, notes, adjustwidth)

    # Concatenate elements
    table = head+body+footer

    # Concatenate output filename
    outfile = os.path.join(out_root, fname+'.tex')

    # Remove old table file if present
    try:
        os.remove(outfile)
    except:
        'no tex file.'

    # Write table to text .tex file
    f = open(outfile, 'a')
    f.write(table)
    f.close()

    # Generate pdf image of table in output directory
    utils.compile_latex(out_root, fname, dpi=dpi)

    return table



def __create_header(headers, title, caption, adjustwidth, tiny, centering,
        extrarowheight):
    '''create LaTeX multirow table header'''

    # Create table template
    n_cols = len(headers[0])
    head = r'\begin{table}[!ht]'+'\n'

    if adjustwidth:
        head += r'\begin{adjustwidth}{-2.25in}{0in}'+'\n'

    if tiny:
        head += r'\tiny'+'\n'

    if centering:
        head += r'\centering'+'\n'

    if title or caption:
        cap_str = r'\caption{'
        if title:
            cap_str += '{'+title+'}'
        if caption:
            cap_str += caption
        cap_str += '}'
        head += cap_str+'\n'

    if extrarowheight:
        if isinstance(extrarowheight, int):
            head += r'\setlength\extrarowheight{'+str(extrarowheight)+'pt}\n'
        else:
            raise TypeError('`extrarowheight` must be an integer value '
                            '(font point size)')

    head += r'\begin{tabular}{ '+('c '*n_cols)+'}'+'\n'
    head += r'\hline'+'\n'

    # Process each list of column names for table
    bold = True
    for cols in headers:
        col_str = ''
        for i in range(len(cols)):
            if bold == True:
                col_str += r' \textbf{'+cols[i]+'} '
            else:
                col_str += r' '+cols[i]+' '

            if i < len(cols)-1:
                col_str += '&'

        # Only first iteration/row will be bolded
        bold = False

        # Append header row to header
        head += col_str+r'\\'+'\n'

    # Add a horizontal line below header rows
    head += r'\hline'+'\n'

    return head


def __create_body(data, cols):
    '''create LaTeX multirow table body
    '''
    import datetime
    import numpy

    # Process each row of body data
    val_str = ''
    keys = list(cols.keys())
    for i in range(len(data)):
        row = data[keys].iloc[i]
        # Add values to row string
        for key in keys:
            # Handle datetime and str objects
            if isinstance(row[key], datetime.datetime):
                val = datetime.datetime.strftime(row[key], '%Y-%m-%d')
            elif data.dtypes[key]==object:
                val = str(row[key])
            # Handle numerics
            else:
                if numpy.isnan(row[key]):
                    val = '-'
                else:
                    val = (cols[key] % row[key])
            val_str = val_str + (val+' ')

            # If not the last column, add an `&`
            if key != keys[-1]:
                val_str = val_str+'& '

        # Add EOF chars to row line
        val_str = val_str+r'\\'+'\n'

    body = val_str+'\n'

    return body


def __create_footer(label, notes, adjustwidth):
    '''create LaTeX multirow table footer'''

    footer = r'\end{tabular}'+'\n'

    # Add table notes
    if notes:
        notes = (r'\begin{flushleft} '+notes+'\n'
                 r'\end{flushleft}'+'\n')
        footer += notes

    if label:
        footer += r'\label{'+label+'}\n'

    # Add end statement to adjustwidth
    if adjustwidth:
        footer += r'\end{adjustwidth}'+'\n'

    footer += r'\end{table}'+'\n'

    return footer


if __name__ == '__main__':

    from collections import OrderedDict
    import os
    import pandas

    # Create output filename for .tex table
    out_root = 'test/'
    os.makedirs(out_root, exist_ok=True)
    fname = 'test'

    # Create dictionary of columns names and associated value str format
    cols = OrderedDict()
    cols['date'] = '%str'
    cols['a']    = '%.2f'
    cols['b']    = '%.0f'

    # List of actual header names to write in table
    head_names = ['Date', 'A Col.', 'B Col.']

    # Units of header columns to write
    head_units = ['',
                  '(degrees)',
                  r'(m day\textsuperscript{-1})']

    headers = [head_names, head_units]

    data = pandas.DataFrame(index=range(5), columns=list(cols.keys()))
    data['a'] = range(5)
    data['b'] = range(5)

    table = write_table(out_root, fname, data, cols, headers)
