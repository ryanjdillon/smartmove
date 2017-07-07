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

* Currently under development (alpha release soon), so may not currently
  function

.. codeblock:: bash
    pip install git@bitbucket.org:ryanjdillon/smartmove.git

Documentation
=============

* Coming shortly

Code contributers
=================

  * **Ryan J. Dillon** - All neural network code (using the `Theanets`
    package), CTD, and plotting code. The `glideid.py` code for glide
    identification is adapted from code written by Lucia Martina Martin Lopez
    (see below). That script uses the library
    [`pyotelem`](https://bitbucket.org/ryanjdillon/pyotelem) with respective
    authorship listed in that repository.

  * **Lucia Martina Martin Lopez** - Original glide identification code has
    written in Matlab

  * Code taken or adapted from [`Stackoverflow`](https://stackoverflow.com/) is
    in the public domain, and the respective posts where it was found should be
    linked to in the document string of the routines in which it was used.

Other contributers
==================

  * **Kagari Aoki** - Calculations of modified seal body density

  * **Ippei Suzuki** - Propeller speed calibration calculations


Thanks
======
The `theanets` package was used for implementing the aritifical neural network
and is great work. Thanks Leif Johnson](https://github.com/lmjohns3)!
