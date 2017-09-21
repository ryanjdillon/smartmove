Project setup
==============

Creating the project directory
------------------------------

`smartmove` uses a series of configuration YAML files and a pre-defined
directory structure for input and output data files. All of these are located
in a project directory, which you define when you create your `smartmove`
project.

First create a directory to use as your `smartmove` project directory:

.. code::

    mkdir /home/<user>/smartmove_project

Open the `IPython` interpretor:

.. code::

    cd ~/opt
    IPython3

Then import `smartmove` and use the `smartmove.create_project()` method to create the necessary subdirectories and configuration files used in the smartmove analyses.

.. code:: python

    import smartmove

    path_project = '/home/<user>/smartmove_project'
    smartmove.create_project(path_project)

The configuration YAML files and necessary directories will then be created in
your project directory, and you should receive a message instructing you to
copy the necessary data files to their respective directories.

**Example project directory structure**

.. code::

    project directory
    ├── cfg_ann.yml
    ├── cfg_experiments.yml
    ├── cfg_glide.yml
    ├── cfg_project.yml
    ├── cfg_subglide-filter.yml
    ├── data_csv
    ├── data_ctd
    ├── data_glide
    ├── data_tag
    └── model_ann

Copying data to the project directory
-------------------------------------

.. note:: The `propeller_calibrations.csv` file is for calibration of the
    propeller sensor data from rotations to units of *m s^-2*, and the
    `cal.yml` files in the Little Leonardo data directories are used by
    `pylleo` for calibrating the accelerometer sensor data to units of gravity.

    See the `pylleo` `documentation
    <http://pylleo.readthedocs.io/en/latest/calibration.html#calibration>`_ for
    more information on performing these calibrations and the naming of data
    files.

The CSV files for field experiments, isotope experiments, and propeller
calibrations should be placed in the `data_csv` directory, the matlab file with
CTD measurements should be placed in the `data_ctd` directory, and the
directories containing Little Leonardo data should be placed in the `data_tag`
directory.

.. code::

    project directory
    ├── data_csv
    │   ├── field_experiments.csv
    │   ├── isotope_experiments.csv
    │   └── propeller_calibrations.csv
    ├── data_ctd
    │   └── kaldfjorden2016_inner.mat
    └── data_tag
        ├── 20150306_W190PD3GT_34839_Notag_Control
        ├── 20150310_W190PD3GT_34839_Notag_Control
        ├── 20150311_W190PD3GT_34839_Skinny_Control
        ├── 20150314_W190PD3GT_34839_Skinny_2neutralBlocks
        ├── 20150315_W190PD3GT_34839_Notag_Control
        ├── 20150316_W190PD3GT_34839_Skinny_4weighttubes_2Blocks
        ├── 20150317_W190PD3GT_34839_Skinny_4Floats
        ├── 20150318_W190PD3GT_34839_Notag_2neutrals
        ├── 20150320_W190PD3GT_34839_Skinny_4weights
        ├── 20150323_W190PD3GT_34839_Skinny_4NeutralBlocks
        ├── 20160418_W190PD3GT_34840_Skinny_2Neutral
        ├── 20160419_W190PD3GT_34840_Skinny_2Weighted
        ├── 20160422_W190PD3GT_34840_Skinny_4Floats
        └── 20160425_W190PD3GT_34840_Skinny_4Weights
