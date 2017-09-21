.. _ann:

Artifical Neural Network Analysis
=================================
.. |start1|  image:: /images/ann/ann_start1.png
.. |start2|  image:: /images/ann/ann_start2.png
.. |run|    image:: /images/ann/ann_running.png
.. |test|   image:: /images/ann/ann_samplesize_test.png
.. |end|    image:: /images/ann/ann_end.png

Input data selection
--------------------
When first running the ANN setup and training, A summary of glide
identification features to be used are displayed

|start1|

All experiments who have glide identification processing results that match those
features will be listed for selection to be compiled into a dataset for the ANN.

You call selecte all experiments by typing `all`, or you can type the
experiment's ids individually in a list separated by commas.

|start2|

After typing your selection and hitting enter, you will see the `Theanets`
trainer intialize and begin the training process.

|run|

|test|

|end|
