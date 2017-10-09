smartmove
=========
An application for creating an Artificial Neural Network to classify body
density of marine mammals from datalogger data (primarily acceleration, depth,
and speed sensor data with the help of some salinity data from CTD
measurements).

While currently developed primarily for the use in a specific project, the code
is written as general as was possible during the project that will hopefully be
easily extended and applied in future projects.

Installation
------------

.. code:: bash

    pip install git@github.com:ryanjdillon/smartmove.git

Documentation
--------------
The full documentation can be found `here <http://smartmove.readthedocs.io/en/latest/index.html>`_ at Read the Docs.

Quickstart
----------

.. code:: python

    import smartmove
    smartmove.create_project('./')

    # Now copy Little Leo data to `data_tag`, csv data to `data_csv`, and CTD data to `data_ctd`
    # Then...

    # Setup your analysis helper
    a = smartmove.Analysis('./')

    # Run the glide identification
    a.run_glides()

    # Run the ANN training
    a.run_ann()

    # Make the figures and tables for the paper
    a.make_tables()
    a.make_figures()


Contributors
------------

  * **Ryan J. Dillon** - All neural network code (using the theanets_ python
    package), CTD, and plotting code. Python translations of glide identification code.

  * **Lucia Martina Martin Lopez** - Original glide identification code written
    in Matlab

  * **Kagari Aoki** - Calculations of modified seal body density

  * **Ippei Suzuki** - Propeller speed calibration calculations

  * Code taken or adapted from `Stackoverflow <https://stackoverflow.com/>`_ is in the public domain, and
    the respective posts where it was found should be linked to in the document
    string of the routines in which it was used.

Acknowledgements
----------------
This development of this code was partially financed by  ARCEx funded by the
Research Council of Norway (Project No. 228107) and as a component of the
COEXIST (Condition and energy expenditure estimates from free-ranging marine
mammals project), which was financed by the Research Council of Norway (Project
No. 234411/E40).

The `theanets`_ package was used for implementing the artificial neural network
and is great work. Thanks `Leif Johnson`__!

.. _theanets: https://github.com/lmjohns3/theanets
.. _leif: https://github.com/lmjohns3
__ leif_

Thank you to Mark Johnson and René Swift at Sea Mammal Research Unit (St.
Andrews, Scotland)the for your help and advice along the way on the processing
of tag data.
