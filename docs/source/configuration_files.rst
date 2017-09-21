.. _config:

Configuration files
===================

The configuration files copied to the project directory from
`smartmove/_templates` are used for configuration of ___


Project configuration
---------------------

.. code:: yaml

    # Datalogger calibration directories
    cal:
      # Contains acceleration calibration YAMLs for associated tag ID
      # Sensors should be calibrated per month--closest calibration used for analysis
      w190pd3gt:
        34839:
          2015:
            03: 20150306_W190PD3GT_34839_Notag_Control

        34840:
          2016:
            04: 20160418_W190PD3GT_34840_Skinny_2Neutral

    # Parameters for field experiment
    experiment:
      # 69° 41′ 57.9″ North, 18° 39′ 4.5″ East
      coords:
        lon: 18.65125
        lat: 69.69942
      net_depth: 18 #meters
      fname_ctd: 'kaldfjorden2016_inner.mat'


Glide analysis
--------------

.. code:: yaml

    # Number of samples per frequency segment in PSD calculation
    nperseg: 256

    # Threshold above which to find peaks in PSD
    peak_thresh: 0.10

    # High/low pass cutoff frequency, determined from PSD plot
    cutoff_frq: None

    # Frequency of stroking, determinded from PSD plot
    stroke_frq: 0.4 # Hz

    # fraction of `stroke_frq` to calculate cutoff frequency (Wn)
    stroke_ratio: 0.4

    # Maximum length of stroke signal
    # 1/stroke_freq
    t_max: 2.5 # seconds

    # Minimumn frequency for identifying strokes
    # 2 / (180*numpy.pi) (Hz)
    J: 0.0349

    # For magnetic pry routine
    alpha: 25

    # Minimum depth at which to recognize a dive
    min_depth: 0.4


Sub-glide filtering
-------------------

.. code:: yaml

    # Pitch angle (degrees) to consider sgls
    pitch_thresh: 30

    # Minimum depth at which to recognize a dive (2. Define dives)
    min_depth: 0.4

    # Maximum cummulative change in depth over a glide
    max_depth_delta: 8.0

    # Minimum mean speed of sublide
    min_speed: 0.3

    # Maximum mean speed of sublide
    max_speed: 10

    # Maximum cummulative change in speed over a glide
    max_speed_delta: 1.0


Artificial Neural network
-------------------------

.. code:: yaml


    # Parameters for compiling data
    data:
        sgl_cols:
            - 'exp_id'
        glides:
          cutoff_frq: 0.3
          J: 0.05
        sgls:
          dur: 2
        filter:
          pitch_thresh: 30
          max_depth_delta: 8.0
          min_speed: 0.3
          max_speed: 10
          max_speed_delta: 1.0

    # Data and network config common to all structures
    net_all:
        features:
            - 'abs_depth_change'
            - 'dive_phase_int'
            - 'mean_a'
            - 'mean_depth'
            - 'mean_pitch'
            - 'mean_speed'
            - 'mean_swdensity'
            - 'total_depth_change'
            - 'total_speed_change'
        target: 'rho_mod'
        valid_frac: 0.6
        n_targets:  10

    # Network tuning parameters, all permutations of these will be trained/validated
    net_tuning:
        # Number of nodes in each hidden layer
        hidden_nodes:
            - 10
            - 20
            - 40
            - 60
            - 100
            - 500
        # Number of hidden layers
        hidden_layers:
            - 1
            - 2
            - 3
        # Trainers (optimizers)
        # https://theanets.readthedocs.io/en/stable/api/trainers.html
        # http://sebastianruder.com/optimizing-gradient-descent/
        algorithm:
            - adadelta
            - rmsprop
        hidden_l1:
            - 0.1
            - 0.001
            - 0.0001
        weight_l2:
            - 0.1
            - 0.001
            - 0.0001
        momentum:
            - 0.9
        patience:
            - 10
        min_improvement:
            - 0.999
        validate_every:
            - 10
        learning_rate:
            - 0.0001
