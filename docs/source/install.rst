Installation
------------

Using `virtualenv` will ensure that you have the correct versions of the
dependencies installed, but it is possible to just install directly in your
native `Python` environment (in which case, skip to `Installing with pip`_).

Installing with pip
-------------------

.. codeblock:: bash

    cd ~/opt
    source venv/bin/activate
    pip install gitlink

After installing `smartmove`, you can install the dependencies using the
requirements file.

.. codeblock:: bash

    pip install -r requirements
