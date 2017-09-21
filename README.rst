smartmove
=========
An application for creating an Artifical Neural Network to classify body
density of marine mammals from datalogger data (primarily acceleration, depth,
and speed sensor data with the help of some salinity data from CTD
measurements).

While currently developed primarily for the use in a specific project, the code
is written as general as was possible during the project that will hopefully be
easily extended and applied in future projects.

Installation
============

.. code:: bash

    pip install git@github.com:ryanjdillon/smartmove.git

Documentation
=============
Documentation can be found at Read the Docs

Contributers
=================

  * **Ryan J. Dillon** - All neural network code (using the theanets_ python
    package), CTD, and plotting code. Python translations of glide identification code.

  * **Lucia Martina Martin Lopez** - Original glide identification code written
    in Matlab
    
  * **Kagari Aoki** - Calculations of modified seal body density

  * **Ippei Suzuki** - Propeller speed calibration calculations

  * Code taken or adapted from `Stackoverflow <https://stackoverflow.com/>`_ is in the public domain, and
    the respective posts where it was found should be linked to in the document
    string of the routines in which it was used.

Thanks
======
The `theanets`_ package was used for implementing the aritifical neural network
and is great work. Thanks `Leif Johnson`__!

.. _theanets: https://github.com/lmjohns3/theanets
.. _leif: https://github.com/lmjohns3
__ leif_
