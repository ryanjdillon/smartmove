Using the Analysis helper class
===============================

After setting up your project directory with `smartmove.create_project()`, a
template `YAML` files were copied to your project directory that configure the
information about the project, experiment indices, glide identification parameters, sub-glide filtering parameters, and parameters for the ANN. Read more about the configuration files here in :ref:`config`.

Once your project directory is configured, you can use the `Analysis` helper
class for running different parts of the analysis. The `Analysis` class object
keeps track of the configuration files and the ANN analysis from which to
generate tables and figures from. This allows you to easily run new ANN
configurations at a later time and inspect the results of all models that have
been run.

Create an analysis
------------------
Activate your virtual environment, and then lauch a python interpreter.

.. code::

    cd ~/opt/smartmove/
    source venv/bin/activate
    Ipython3

Then initiate an `Analysis` class object from which to run and inpect results
from the ANN. After initializing your analysis, you can execute the glide
identification function with the class method `run_glides()`, which will walk
you through the glide identification.

.. code:: python

    import smartmove

    path_project = './'

    a = smartmove.Analysis(path_project)

You can inspect the attributes of the object from within a Python interpreter,
such as `iPython`:

.. code:: python

    # Show the names of attributes for `a`
    vars(a).keys()

    # Show all attributes and class methods available from `a`
    dir(a)

    # Print the glide configuration dictionary
    a.cfg_glide


Glide identification
---------------------

.. code:: python

    # Run with `cfg_glide` and splitting sub-glides into 2s segments
    a.run_glides(sgl_dur=2)

See the :ref:`glide_identification` documentation for an overview of the
procedure.

Run ANN
-------

.. code:: python

    a.run_ann()

See the :ref:`ann` documentation for an overview of the
procedure.


Paper figures and tables
------------------------

Running the following routine will create a `paper/` subdirectory the project
directory passed when initializing your `Analysis` class object.

.. |select| image:: /images/ann/ann_analysis_select.png

Be sure to load the ANN analysis you wish to produce figures for before running
the commands.

|select|

.. code:: python

    # Generate figures
    a.make_tables()

    # Generate figures
    a.make_figures()
