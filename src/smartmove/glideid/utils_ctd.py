from collections import OrderedDict
import datetime

import gsw
import numpy
import pandas
import pyproj
import scipy.io


def read_matlab(file_ctd_mat):
    """Read IMR matlab CTD transect file to python dictionary

    Args
    ----
    file_ctd_mat: str
        full path and name of IMR CTD Matlab file

    Returns
    -------
    transects: OrderedDict
        Dictionary of transects, each with corresponding CTD data
    """

    mat = scipy.io.loadmat(file_ctd_mat)
    keys = sorted(list(mat.keys()))

    # Assign not meta fields to separate dicts for transects
    transects = OrderedDict()
    for k in keys:
        # Ignore meta fields in `mat` dict
        if not k.startswith("__"):
            k_id = k.split("_")[0]
            # Create station dict if not present
            if k_id not in transects.keys():
                transects[k_id] = OrderedDict()
            # Add field to station dict, remove k_id from k
            k_sub = "_".join(k.split("_")[1:])
            transects[k_id][k_sub] = mat[k]

    # Calcuate depth from pressure, convert matlab time to datestamp
    for k in transects.keys():
        transects[k] = calc_depth_date(transects[k])

    return transects


def calc_depth_date(transect):
    """Cacluate the depth from pressure and convert matlab date to python

    Args
    ----
    transect: dict
        Transect dictionary with corresponding CTD data

    Returns
    -------
    transect: dict
        Input transect dictionary with depths and updated timestamps
    """

    def matlab2datetime(matlab_datenum):
        """Convert Matlab datenum to Python datetime"""

        day = datetime.datetime.fromordinal(int(matlab_datenum))
        dayfrac = datetime.timedelta(days=matlab_datenum % 1) - datetime.timedelta(
            days=366
        )

        return day + dayfrac

    n_depths, n_stations = transect["date_up"].shape
    transect["depth"] = numpy.zeros((n_depths, n_stations), dtype=float)
    transect["date_up"] = transect["date_up"].astype(object)
    lats = numpy.meshgrid(transect["lat"], numpy.ones(n_depths))[0]

    for i in range(n_depths):
        for j in range(n_stations):
            transect["depth"][i][j] = gsw.z_from_p(
                transect["press_up"][i][j], lats[i][j]
            )
            try:
                transect["date_up"][i][j] = matlab2datetime(transect["date_up"][i][j])
            except Exception:
                pass

    return transect


def get_station_lonlats(transects):
    """Get latitude and longitude for each station in each transect

    Args
    ----
    transects: OrderedDict
        Dictionary of transects, each with corresponding CTD data

    Returns
    -------
    data: pandas.DataFrame
        Dataframe with transect name, latitude and longitude of each station
    """

    # Count number of lons/lats in all station dataframes
    n = 0
    for transect in transects.values():
        n += len(transect["lon"])

    # Create label, lon, lat dataframe
    data = pandas.DataFrame(index=range(n), columns=["id", "lon", "lat"])

    n0 = 0
    for key, transect in transects.items():
        n1 = len(transect["lon"]) - 1
        data.ix[n0 : n0 + n1, "lon"] = transect["lon"].T
        data.ix[n0 : n0 + n1, "lat"] = transect["lat"].T
        data.ix[n0 : n0 + n1, "id"] = key
        n0 = n1

    return data


def find_nearest_station(lon, lat, transects):
    """Find the transect and lon/lat index of the station nearest to lon, lat

    Args
    ----
    lon: float
        Longitude from which the distance to the station locations are measured
    lat: float
        Latitude from which the distance to the station locations are measured
    transects: OrderedDict
        Dictionary of transects, each with corresponding CTD data

    Returns
    -------
    transect_key: str
        Key name of transect in `transects` with the station nearest to lon/lat
    station_idx: int
        Index of the transect data array for the station nearest to lon/lat
    min_dist: float
        Distance in meters betweeen lon/lat and the nearest station
    """

    g = pyproj.Geod(ellps="WGS84")

    transect_key = ""
    station_idx = ""
    min_dist = 9e12
    for k in transects.keys():
        lons = transects[k]["lon"]
        lats = transects[k]["lat"]

        for i in range(len(lons)):
            _, _, dist = g.inv(lon, lat, lons[i], lats[i])
            if dist[0] < min_dist:
                transect_key = k
                station_idx = i
                min_dist = dist[0]

    return transect_key, station_idx, min_dist


def calc_mean_salinity(transects, transect_key, station_idx, max_depth):
    """Calculate the mean salinity above a maximum depth at a station in
    transects

    Args
    ----
    transects: OrderedDict
        Dictionary of transects, each with corresponding CTD data
    transect_key: str
        Key name of transect in `transects` with the station nearest to lon/lat
    station_idx: int
        Index of the transect data array for the station nearest to lon/lat

    Returns
    -------
    mean_sal: float
        Mean salinity above `max_depth` at specified station
    """

    # Cacluate mean salinity above max_depth
    depths = transects[transect_key]["depth"][:, station_idx]
    depth_mask = depths > -abs(max_depth)

    sals = transects[transect_key]["sal_up"][:, station_idx]
    mean_sal = numpy.mean(sals[depth_mask])

    return mean_sal


def get_seawater_densities(file_ctd_mat, t, lon, lat, max_depth):

    transects = read_matlab(file_ctd_mat)

    # Find nearest station
    nearest_key, nearest_idx, min_dist = find_nearest_station(lon, lat, transects)

    # Cacluate mean salinity above 18m
    mean_sal = calc_mean_salinity(transects, nearest_key, nearest_idx, max_depth)

    SA = numpy.asarray([mean_sal] * len(t))
    p = numpy.zeros(len(t))

    return gsw.rho(SA, t, p)
