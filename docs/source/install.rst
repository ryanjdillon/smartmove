Installation
============

Download Smartmove
------------------
First you will want to install `Python` version 3.5+. There are loads of
tutorials on doing this, and the commands provided for installation and running
the code *should* prevent any issues from occuring if you have multiple versions
of `Python` installed.

You can get the `smartmove` code by either using `git clone` (shown below), or
by downloading an archive file and extracting it to the location of your
choice. Note, if the location of the code is not in you `PYTHONPATH`, you will
only be able to import the code as a `Python` module from its parent directory
(`more on that here
<https://docs.python.org/3.5/tutorial/modules.html#the-module-search-path>`_).

The publication release version can be found here: <git link>

The latest version of the code can be found here: <git link>

.. code::

    mkdir ~/opt
    cd ~/opt
    git clone <git link>

Virtual environment
-------------------

Using `virtualenv` will ensure that you have the correct versions of the
dependencies installed, but it is possible to just install directly in your
native `Python` environment (in which case, skip to :ref:`dependencies`).

With `virtualenv` already installed, create a virtual environment to ensure
that you are you using the correct dependency versions for `smartmove`:

.. code::

    cd ~/opt/smartmove
    virtualenv --python=python3 venv

Then activate the virtual environment before installing the dependencies using the
`requirements.txt` file.

.. code::

    source venv/bin/activate

.. note:: Anytime you run `smartmove`, either in a `Python` interpreter or
    running a script from a commandline, you'll need to activate the virtual
    environment first.

.. _dependencies:

Installing dependencies
-----------------------

After you have installed and activated your virtual environment, or if you are
skipping that step and installing everything to your local `Python`
installation, install the files using the `requirements.txt` file from
`smartmove` directory:

.. code::

    cd ~/opt/smartmove
    source venv/bin/activate
    pip3 install -r requirements.txt
