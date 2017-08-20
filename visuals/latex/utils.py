
def compile_latex(path, fname, debug, dpi=300):
    '''Compile LaTeX table to pdf image

    http://tex.stackexchange.com/a/209227/33219
    '''
    import os
    import subprocess

    # Call pdflatex from shell to compile .tex table to pdf document
    # pdflatex '\documentclass{article}\usepackage{array}\begin{document}\'
    #          'pagenumbering{gobble}\input{'+fname+'}\end{document}'

    cwd = os.getcwd()
    os.chdir(path)

    if debug: print('creating latex')

    cmd = ['pdflatex',
           (r'\documentclass{article}'
            r'\usepackage{changepage}'
            r'\usepackage{mathtools}'
            r'\usepackage{gensymb}'
            r'\usepackage{array}'
            r'\usepackage[usenames, dvipsnames]{color}'
            r'\begin{document}'
            r'\pagenumbering{gobble}'
            r'\input{'+fname+'}'
            r'\end{document}')
           ]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # Crop whitespace from table
    if debug: print('remove whitespace')

    pdfname = 'article.pdf'
    cmd = ['pdfcrop', pdfname, pdfname]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # Rename latex output to match filename
    if debug: print('rename pdf')
    os.renames('article.pdf', fname+'.pdf')

    # Remove compilation files by extension
    if debug: print('remove .aux .log')
    for f in os.listdir('./'):
        if any(f.endswith(ext) for ext in ['.aux', '.log']):
            os.remove(f)

    # Convert pdf to png image
    if debug: print('convert pdf to png')
    pdf_to_img(path, fname, in_ext='pdf', out_ext='png', dpi=300)

    os.chdir(cwd)

    return None


def pdf_to_img(path, fname, in_ext='pdf', out_ext='png', dpi=300):
    '''Convert pdf to image type with extension `ext`

    This utilizes the linux command `convert` from `imagemagik`
    convert -density 300 -trim exp_table.pdf -quality 100 test.png
    http://stackoverflow.com/a/6605085/943773
    '''
    import os
    import subprocess

    file_in  = os.path.join(path, '{}.{}'.format(fname, in_ext))
    file_out = os.path.join(path, '{}.{}'.format(fname, out_ext))
    cmd = ['convert', '-density', str(dpi), '-trim', file_in, '-quality', '100',
            file_out]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    return None


