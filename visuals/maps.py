'''
A Package for retrieving Kartverket image data for specified areas.

Notes
-----
Web Cache (tiles) GetCapabilities():
http://opencache.statkart.no/gatekeeper/gk/gk.open?Version=1.0.0&service=wms&request=getcapabilities

owslib image tutorial
https://geopython.github.io/OWSLib/index.html?highlight=webmapservice

Kartverket WMS
http://kartverket.no/data/API-og-WMS/

Karverket example
http://openwms.statkart.no/skwms1/wms.topo3?version=1.1.1&styles=&service=wms&REQUEST=map&SRS=EPSG:32633&BBOX=210924.955,6668620.35,255289.776,6688292.32&LAYERS=topo3_WMS&WIDTH=1650&HEIGHT=1100&FORMAT=image/png&BGCOLOR=0xFFFFFF&TRANSPARENT=TRUE
'''

def get_wms_dict(xml):
    '''An almost useful routine from creating a dict from a capabilities XML'''
    from collections import OrderedDict
    from bs4 import BeautifulSoup

    def get_attrs(layer, key):
        return layer.find(key).attrs

    soup = BeautifulSoup(xml, 'lxml')

    layers = soup.findAll('layer')[1:]

    d = OrderedDict()
    for l in layers:
        title = l.find('title').text
        d[title] = OrderedDict()

        boundingboxes = l.findAll('boundingbox')
        for srs in sorted([srs.text for srs in l.findAll('srs')]):
            for bb in boundingboxes:
                if bb['srs'] == srs:
                    d[title][srs] = OrderedDict()
                    for k in sorted(bb.attrs.keys()):
                        if k != 'srs':
                            d[title][srs][k] = bb.attrs[k]
    return d


def project_bbox(srs, lon0, lat0, lon1, lat1):
    '''Project the bounding box for map extent coords from WGS84 to `srs`

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
    '''

    import pyproj

    wgs84 = pyproj.Proj(init='EPSG:4326')
    proj = pyproj.Proj('+init={}'.format(srs))

    minx, miny = pyproj.transform(wgs84, proj, lon0, lat0)
    maxx, maxy = pyproj.transform(wgs84, proj, lon1, lat1)

    return (minx, miny, maxx, maxy)


def get_size(bbox, width):
    '''Generate adjusted width and height from bounds and given width'''
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
    '''
    Args
    ----
    srs: str
        Spatial reference system

    '''

    img_fmt = 'image/png'

    # Generate size parameters from `bbox` and desired pixel width
    size = get_size(bbox, width)

    # Retrieve map data using WMS GetMap() call
    owslib_img = wms.getmap(layers=[layer], srs=srs, bbox=bbox, size=size,
                            format=img_fmt, transparent=transparent)

    return owslib_img


def png2geotiff(filename_png, srs, bbox):
    '''Read and convert png file to GEOTIFF file with GDAL'''
    import os
    import subprocess

    filename_tif = '{}.tif'.format(os.path.splitext(filename_png)[0])
    params = (srs, bbox[0], bbox[1], bbox[2], bbox[3], filename_png, filename_tif)
    call = 'gdal_translate -a_srs {} -a_ullr {} {} {} {} {} {}'.format(*params)
    subprocess.check_call(call.split(' '))

    return None


def transform_points(x, y, x0, y0, x1, y1, img_size):
    x_ratio = img_size[0]/(x1-x0)
    y_ratio = img_size[1]/(y1-y0)

    new_x = (x - x0) * x_ratio
    new_y = (y - y0) * y_ratio
    return new_x, new_y


def map_ticks(pos0, pos1, n, nsew=False):
    import numpy

    def parse_degminsec(dec_degs, method=None, round_secs=False):
        '''Parse decimal degrees to degrees, minutes and seconds'''
        degs = numpy.floor(dec_degs)
        dec_mins = numpy.abs((dec_degs - degs) * 60)
        mins = numpy.floor(dec_mins)
        secs = numpy.abs((dec_mins - mins) * 60)

        if method == 'lon':
            if degs < 0:
                nsew = 'W'
            elif degs > 0:
                nsew = 'E'
            else:
                nsew = ''
        elif method == 'lat':
            if degs < 0:
                nsew = 'S'
            elif degs > 0:
                nsew = 'N'
            else:
                nsew = ''
        else:
            nsew = ''

        if round_secs:
            secs = numpy.round(secs)

        return degs, mins, secs, nsew

    ticks = numpy.linspace(pos0, pos1, n)
    print('lon lat', pos0, pos1)

    fmt = "{:.0f}$\degree$ {:.0f}$'$ {:.0f}$''$"

    degs, mins, secs, nsews = parse_degminsec(ticks, round_secs=True)
    if nsew:
        fmt += ' {}'
        values = zip(degs, mins, secs, nsews)
        labels = [fmt.format(d, m, s, ns) for d, m, s in values]
    else:
        values = zip(degs, mins, secs)
        labels = [fmt.format(d, m, s) for d, m, s in values]

    return ticks, labels
