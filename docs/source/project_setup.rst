Pre-processing data
===================

You must first create a directory for your project with the data to be
processed. After the directories and data have been setup, you will configure
the code to use this project directory for your analysis.

.. codeblock:: bash
    project directory
    ├── data_accelerometer
    │   ├── 20150306_W190-PD3GT_34839_Notag_Control
    │   ├── 20150311_W190-PD3GT_34839_Skinny_Control
    │   ├── 20150314_W190PD3GT_34839_Skinny_2neutrals
    │   ├── 20150317_W190PD3GT_34839_Skinny_4Floats
    │   ├── 20150318_W190PD3GT_34839_Notag_2neutrals
    │   ├── 20150320_W190PD3GT_34839_Skinny_4weights
    │   └── speed_calibrations.csv
    ├── data_csv
    │   ├── field_experiments.csv
    │   └── isotope_experiments.csv
    └── data_ctd

The `data_accelerometer` should contain a data directory for each experiment
and its associated output files from the Little Leonardo dataloggers. 

These files should be named with the following naming convention::

    <date>_<tag model>_<tag serial>_<animal_name>_<modification>_suffix.TXT


The directory and file names are used by the program for processing, so it is
important that these are named correctly. Below is an example of how the
contents of a Little Leonardo data directory should look:

.. codeblock:: bash
    ./20160418_W190PD3GT_34840_Skinny_2Neutral
    ├── 20160418_W190PD3GT_34840_Skinny_2Neutral-Acceleration-X.TXT
    ├── 20160418_W190PD3GT_34840_Skinny_2Neutral-Acceleration-Y.TXT
    ├── 20160418_W190PD3GT_34840_Skinny_2Neutral-Acceleration-Z.TXT
    ├── 20160418_W190PD3GT_34840_Skinny_2Neutral-Depth.TXT
    ├── 20160418_W190PD3GT_34840_Skinny_2Neutral-Propeller.TXT
    └── 20160418_W190PD3GT_34840_Skinny_2Neutral-Temperature.TXT


Calibrate little leonardo data
------------------------------
Before running the glide identification program, the acceleration data must be calibrated to units of `g` (acceleration to gravity at Earths surface, i.e. 9.80665 m/s^2). There is a script in the `pylleo` which runs a `bokeh` web application to facilitate doing this.

For further information on how to use the calibration web application, please
see the calibration section of the `pylleo` documentation.

.. codeblock:: bash
    cd <project>/data_acceleration/20160418_W190PD3GT_34840_Skinny_2Neutral
    bokeh serve --show bokeh_calibration.py --args W190PD3GT 34839 1 ./


The following files will be created in the experiments data directory:

.. codeblock:: bash
    ./20160418_W190PD3GT_34840_Skinny_2Neutral
    ├── cal.yml
    ├── meta.yml
    └── pydata_20160418_W190PD3GT_34840_Skinny_2Neutral.p


Perform glide identification
----------------------------
From the glide directory

Configure neural network model
------------------------------

Run neural network model
------------------------
