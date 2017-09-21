.. _glide_identification:

Glide identification
====================



Plot tools
----------

Throughout the glide identification process, plots will be generated from which
you can inspect the data. These are plot instances created using the
`matplotlib` library, which have some default tools that are used by the user
to determine values to be manually entered by the user, so we'll cover those
tools now.

.. |home| image:: /images/matplotlib/button_home.png
.. |back| image:: /images/matplotlib/button_back.png
.. |fwd|  image:: /images/matplotlib/button_fwd.png
.. |pan| image:: /images/matplotlib/button_pan.png
.. |zoom| image:: /images/matplotlib/button_zoom.png
.. |cfg| image:: /images/matplotlib/button_config.png
.. |save| image:: /images/matplotlib/button_save.png

+----------+-------------------------------------------+
| **Icon** | **Bokeh documentation description**       |
+----------+-------------------------------------------+
| |home|   | Reset the original view of the plot       |
+----------+-------------------------------------------+
| |back|   | Back to previous plot view                |
+----------+-------------------------------------------+
| |fwd|    | Forward to next plot view                 |
+----------+-------------------------------------------+
| |pan|    | Pan axes with left mouse, zoom with right |
+----------+-------------------------------------------+
| |zoom|   | Zoom to the selected rectangle            |
+----------+-------------------------------------------+
| |cfg|    | Configure subplot attributes              |
+----------+-------------------------------------------+
| |save|   | Save the figure                           |
+----------+-------------------------------------------+

.. |term_select| image:: /images/glide/glide_tag_select.png

.. |plot_exp1|   image:: /images/glide/glide_exp-bound_zoom1.png
.. |plot_exp2|   image:: /images/glide/glide_exp-bound_zoom2.png
.. |plot_exp3|   image:: /images/glide/glide_exp-bound_zoom3.png
.. |plot_exp4|   image:: /images/glide/glide_exp-bound_zoom4.png
.. |term_exp1|   image:: /images/glide/glide_exp-bound_calc.png
.. |term_exp2|   image:: /images/glide/glide_exp-bound_select.png

.. |plot_psd1|   image:: /images/glide/glide_psd_main.png
.. |plot_psd2|   image:: /images/glide/glide_psd_zoom.png
.. |term_cutoff| image:: /images/glide/glide_cutoff_select.png

.. |plot_acc1|   image:: /images/glide/glide_acc-plot_main.png
.. |plot_acc2|   image:: /images/glide/glide_acc-plot_zoom.png

.. |plot_dive1|  image:: /images/glide/glide_dive-PRH-plot_main.png
.. |plot_dive2|  image:: /images/glide/glide_dive-PRH-plot_zoom.png

.. |plot_J1|     image:: /images/glide/glide_fluke-plot_main.png
.. |plot_J2|     image:: /images/glide/glide_fluke-plot_zoom1.png
.. |plot_J3|     image:: /images/glide/glide_fluke-plot_zoom2.png
.. |term_J|      image:: /images/glide/glide_fluke_select.png

.. |plot_sgl1|   image:: /images/glide/glide_sgl-plot_main.png
.. |plot_sgl2|   image:: /images/glide/glide_sgl-plot_zoom.png


Select tag data to process
--------------------------
At the start of the glide processing you are prompted to select the tag data
directories which should be processed. You can type `all` for processing all
tag directories or type a list of ID numbers for those you wish to process
separated by commas (e.g. `0,4,6`).

|term_select|

If the data has not previously been loaded by `pylleo` you will see then see output pertaining to the loading of the data, and a binary `pandas` pickle file will be saved in the data directory for subsequent loading.


Selecting a cutoff frequency
----------------------------
The Power Spectral Density plot will be then be shown, which is used for
determining the cutoff frequency to split the filter the accelerometry data.

|plot_psd1|

Using the |zoom| tool select the area to the left including the peak and
the area of the curve up to the point at which it flattens out. The frequency
(x-axis values) used for the smartmove paper was selected to be the point past
the falling inflection point which was roughly half the distance between the
maximum and the falling inflection point (pictured below).

.. note:: User input required

|plot_psd2|

The frequency you determine from looking at these plots can then be entered in
the terminal.

|term_cutoff|

Review processed data
---------------------
The accelerometer data will then be low and high-pass filtered, and plots of
the split accelerometry data will be shown with the original signal, low-pass
filtered signal, and high-pass filtered signal.

|plot_acc1|

You can use the |zoom| tool to get a better idea of how the signals have been
split at higher resolutions.

|plot_acc2|

Plots of the identified dives are then shown with descent phases labeled in blue and
ascent phases labeled in green. The subplot beneath the dives shows the pitch
angle of the animal calculated from the accelerometer data, with the low
filtered signal (red) plotted on top of the original signal (green).

|plot_dive1|

Zoomed in used |zoom|

|plot_dive2|


Selecting a glide threshold
----------------------------
A diagnostic plot for determining the threshold for determing what portions
of the accelerometer signal are considered to be active stroking vs. gliding
will be displayed. will then be displayed showing the PSD plot of the high
frequency signals for the x ans z axes, along with a plot of these signals over
time. In the PSD plot, the peak in power (y-axis) should occur roughly at the
frequency (x-axis) the characterizes stroking movements. Zooming into greater
detail in the acceleration subplot using the |zoom| tool, You can then look at
areas which appear to have relatively steady activity below this frequency
(y-axis).

|plot_J1|
|plot_J2|
|plot_J3|

After determining the threshold, enter it when prompted in the terminal

|term_J|

Glide events will then be identified as as the areas below this threshold, and
sub-glides will be split from the glides using the `sgl_dur` value passed to
(e.g. `a.run_glides(sgl_dur=2)`).


Reviewing split sub-glides
--------------------------
A plot of the depth data and high frequency acceleration of the z-axis will be
shown with each sub-glide highlighted and the sequential labels of the dive number in
which it occurred and the number of the sub-glide.

|plot_sgl1|

Zooming in with |zoom| will give you better view of things.

|plot_sgl2|
