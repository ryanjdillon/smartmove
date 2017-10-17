
def compile_latex(path, fname, out_ext=None, dpi=300):
    '''Compile LaTeX table to pdf image

    http://tex.stackexchange.com/a/209227/33219
    '''
    from contextlib import contextmanager
    from PIL import Image
    import os
    from subprocess import Popen, PIPE, CalledProcessError

    @contextmanager
    def cd(newdir):
        prevdir = os.getcwd()
        os.chdir(os.path.expanduser(newdir))
        try:
            yield
        finally:
            os.chdir(prevdir)

    # Call pdflatex from shell to compile .tex table to pdf document
    # pdflatex '\documentclass{article}\usepackage{array}\begin{document}\'
    #          'pagenumbering{gobble}\input{'+fname+'}\end{document}'

    with cd(path):

        cmd = ['pdflatex',
               (r'\documentclass{article}'
                r'\usepackage{changepage}'
                r'\usepackage[landscape]{geometry}'
                r'\usepackage{mathtools}'
                r'\usepackage{gensymb}'
                r'\usepackage{array}'
                r'\usepackage{nameref, hyperref}'
                r'\usepackage[usenames, dvipsnames]{color}'
                r'\begin{document}'
                r'\pagenumbering{gobble}'
                r'\input{'+fname+'}'
                r'\end{document}')
               ]

        with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='') # process line here

        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)

        # Rename latex output to match filename
        os.renames('article.pdf', fname+'.pdf')

        # Remove compilation files by extension
        for f in os.listdir('./'):
            if any(f.endswith(ext) for ext in ['.aux', '.log']):
                os.remove(f)

        # Convert pdf to png image
        if out_ext:
            in_ext = 'pdf'
            pdf_to_img(path, fname, in_ext=in_ext, out_ext=out_ext, dpi=dpi)

            # Crop 'png'
            file_im = os.path.join(path,'{}.{}'.format(fname, out_ext))
            im = Image.open(file_im, 'r')
            im2 = im.crop(im.getbbox())
            im2.save(file_im)

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


