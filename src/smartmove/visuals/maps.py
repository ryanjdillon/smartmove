"""
This module contains functions for retrieving Kartverket map data for specified areas.

Notes
-----
The retrieval of map data uses `owslib`. Read more in the `owslib image
tutorial
<https://geopython.github.io/OWSLib/index.html?highlight=webmapservice>`_.

Karverket data is retrieved from the `Kartverket WMS
<http://kartverket.no/data/API-og-WMS/>`_.

Karverket WMS example:

.. code:: bash

    http://openwms.statkart.no/skwms1/wms.topo3?version=1.1.1&styles=&service=wms&REQUEST=map&SRS=EPSG:32633&BBOX=210924.955,6668620.35,255289.776,6688292.32&LAYERS=topo3_WMS&WIDTH=1650&HEIGHT=1100&FORMAT=image/png&BGCOLOR=0xFFFFFF&TRANSPARENT=TRUE

"""


def get_wms_dict(xml):
    """An almost useful routine from creating a dict from a capabilities XML

    Args
    ----
    xml: str
        Capabilities XML in string format

    Returns
    -------
    d: OrderedDict
        Capabilities XML key/values in dict format
    """
    from collections import OrderedDict
    from bs4 import BeautifulSoup

    def get_attrs(layer, key):
        return layer.find(key).attrs

    soup = BeautifulSoup(xml, "lxml")

    layers = soup.findAll("layer")[1:]

    d = OrderedDict()
    for l in layers:
        title = l.find("title").text
        d[title] = OrderedDict()

        boundingboxes = l.findAll("boundingbox")
        for srs in sorted([srs.text for srs in l.findAll("srs")]):
            for bb in boundingboxes:
                if bb["srs"] == srs:
                    d[title][srs] = OrderedDict()
                    for k in sorted(bb.attrs.keys()):
                        if k != "srs":
                            d[title][srs][k] = bb.attrs[k]
    return d


def project_bbox(srs, lon0, lat0, lon1, lat1):
    """Project the bounding box for map extent coords from WGS84 to `srs`

    Args
    ----
    srs: str
        Spatial Reference System for map output
    lon0: float
        Minimum longitude for map extent
    lat0: float
        Minimum latitude for map extent
    lon1: float
        Maximum longitude for map extent
    lat1: float
        Maximum latitude for map extent

    Returns
    -------
    bbox: float tuple
        Bounding box for map extent. Value is `minx, miny, maxx, maxy` in units
        of the SRS
    """

    import pyproj

    wgs84 = pyproj.Proj(init="EPSG:4326")
    proj = pyproj.Proj("+init={}".format(srs))

    minx, miny = pyproj.transform(wgs84, proj, lon0, lat0)
    maxx, maxy = pyproj.transform(wgs84, proj, lon1, lat1)

    return (minx, miny, maxx, maxy)


def get_size(bbox, width):
    """Generate adjusted width and height from bounds and given width

    Args
    ----
    bbox: float tuple
        Bounding box for map extent. Value is `minx, miny, maxx, maxy` in units
        of the SRS
    width: int
        Pixel width for Karverket WMS GetMap() query

    Return
    ------
    width: int
        Adjusted pixel width for Karverket WMS GetMap() query
    height: int
        Adjusted pixel height for Karverket WMS GetMap() query
    """
    import pyproj

    # Maximum WIDTH/HEIGHT dimension for Kartveket WMS GetMap call
    maxdim = 4096

    # Make width equal `maxdim` if too large
    width = min(width, maxdim)

    # Get ratio between projected dimensions
    xdiff = bbox[2] - bbox[0]
    ydiff = bbox[3] - bbox[1]
    yx_ratio = ydiff / xdiff

    # Calcuate height from projected dimension
    height = round(width * yx_ratio)

    # Adjust values if height too large
    if height > maxdim:
        height = maxdim
        width = round(height / yx_ratio)

    return width, height


def get_wms_png(wms, bbox, layer, srs, width=1600, transparent=True):
    """Get map data via WMS GetMap method for given bounding box, and width

    Args
    ----
    wms: owslib.wms
        WMS object with getmap() class method to call
    bbox: float tuple
        Bounding box for map extent. Value is `minx, miny, maxx, maxy` in units
        of the SRS
    layer: str
        Name of WMS layer to retrieve
    srs: str
        Spatial reference system
    width: int
        Pixel width for Karverket WMS GetMap() query
    transparent: bool
        Switch to make background color transparent (Default: True)

    Returns
    -------
    oswslib_img: owslib.image
        Image object with retrieved image data
    """

    img_fmt = "image/png"

    # Generate size parameters from `bbox` and desired pixel width
    size = get_size(bbox, width)

    # Retrieve map data using WMS GetMap() call
    owslib_img = wms.getmap(
        layers=[layer],
        srs=srs,
        bbox=bbox,
        size=size,
        format=img_fmt,
        transparent=transparent,
    )

    return owslib_img


def png2geotiff(filename_png, srs, bbox):
    """Read and convert png file to GEOTIFF file with GDAL

    Args
    ----
    filename_png: str
        Path and filename of png file to be output
    srs: str
        Spatial reference system string (e.g. EPSG:4326)
    bbox: float tuple
        Bounding box for map extent. Value is `minx, miny, maxx, maxy` in units
        of the SRS
    """
    import os
    import subprocess

    filename_tif = "{}.tif".format(os.path.splitext(filename_png)[0])
    params = (srs, bbox[0], bbox[1], bbox[2], bbox[3], filename_png, filename_tif)
    call = "gdal_translate -a_srs {} -a_ullr {} {} {} {} {} {}".format(*params)
    subprocess.check_call(call.split(" "))

    return None


def map_ticks(pos0, pos1, n, nsew=False):
    """Generate n tick positions and labels from given start and end position

    Args
    ----
    pos0: float
        Lon or Lat starting point
    pos1: float
        Lon or Lat end point
    n: int
        Number of tick positions and labels to generate
    nsew: bool
        Switch to append N, S, E, or W to tick labels (Default: False)

    Returns
    -------
    ticks: list of float
        Projected tick positions
    labels: list of str
        Labels in DPS for generated tick positions
    """
    import numpy

    def parse_degminsec(dec_degs, method=None, round_secs=False):
        """Parse decimal degrees to degrees, minutes and seconds"""
        degs = numpy.floor(dec_degs)
        dec_mins = numpy.abs((dec_degs - degs) * 60)
        mins = numpy.floor(dec_mins)
        secs = numpy.abs((dec_mins - mins) * 60)

        if method == "lon":
            if degs < 0:
                nsew = "W"
            elif degs > 0:
                nsew = "E"
            else:
                nsew = ""
        elif method == "lat":
            if degs < 0:
                nsew = "S"
            elif degs > 0:
                nsew = "N"
            else:
                nsew = ""
        else:
            nsew = ""

        if round_secs:
            secs = numpy.round(secs)

        return degs, mins, secs, nsew

    ticks = numpy.linspace(pos0, pos1, n)
    print("lon lat", pos0, pos1)

    fmt = "{:.0f}$\degree$ {:.0f}$'$ {:.0f}$''$"

    degs, mins, secs, nsews = parse_degminsec(ticks, round_secs=True)
    if nsew:
        fmt += " {}"
        values = zip(degs, mins, secs, nsews)
        labels = [fmt.format(d, m, s, ns) for d, m, s in values]
    else:
        values = zip(degs, mins, secs)
        labels = [fmt.format(d, m, s) for d, m, s in values]

    return ticks, labels
