
def lung_capacity(mass):
    '''Caclulate lung capacity Kooyman and Sinnett (1981)'''
    return 0.135*(mass**0.92)


def calc_CdAm(farea, mass):
    '''Calculate drag term

    Args
    ----
    farea: float
        frontal area of animal
    mass: float
        total animal mass

    Returns
    -------
    CdAm: float
        friction term of hydrodynamic equation

    Miller et al. 2004: Cd set to 1.06, prolate spheroid, fineness ratio 5.0
    '''
    Cd = 1.06
    return Cd*area/mass


def bodycomp(mass, tbw, method='reilly', simulate=False, n_rand=1000):
    '''Create dataframe with derived body composition values

    Args
    ----
    mass: ndarray
        dry weight of the seal (kg)
    tbw: ndarray
        wet weight of the seal (kg)
    method: str
        name of method used to derive composition values
    simulate: bool
        switch for generating values with random noise
    n_rand: int
        number of density values to simulate

    Returns
    -------
    field: pandas.Dataframe
        dataframe containing columns for each body composition value
    '''
    import numpy
    import pandas

    if len(mass) != len(tbw):
        raise SystemError('`mass` and `tbw` arrays must be the same length')

    bc = pandas.DataFrame(index=range(len(mass)))

    rnorm = lambda n, mu, sigma: numpy.random.normal(mu, sigma, n)

    if method == 'reilly':
        if simulate is True:
            bc['ptbw'] = 100 * (tbw / mass)
            bc['ptbf'] = 105.1 - (1.47 * bc['ptbw']) + rnorm(n_rand, 0, 1.1)
            bc['ptbp'] = (0.42 * bc['ptbw']) - 4.75 + rnorm(n_rand, 0, 0.8)
            bc['tbf'] = mass * (bc['ptbf'] / 100)
            bc['tbp'] = mass * (bc['ptbp'] / 100)
            bc['tba'] = 0.1 - (0.008 * mass) + \
                               (0.05 * tbw) + rnorm(0, 0.3, n_rand)
            bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - \
                          0.4 + rnorm(0, 17.2, n_rand)
        else:
            bc['ptbw'] = 100 * (tbw / mass)
            bc['ptbf'] = 105.1 - (1.47 * bc['ptbw'])
            bc['ptbp'] = (0.42 * bc['ptbw']) - 4.75
            bc['tbf'] = mass * (bc['ptbf'] / 100)
            bc['tbp'] = mass * (bc['ptbp'] / 100)
            bc['tba'] = 0.1 - (0.008 * mass) + (0.05 * tbw)
            bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - 0.4
    else:
        bc['ptbw'] = 100 * (tbw / mass)
        bc['tbf'] = mass - (1.37 * tbw)
        bc['tbp'] = 0.27 * (mass - bc['tbf'])
        bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - 0.4
        bc['ptbf'] = 100 * (bc['tbf'] / mass)
        bc['ptbp'] = 100 * (bc['tbp'] / mass)

    return bc


def perc_bc_from_lipid(p_lipid):
    '''Calculate body composition component percentages based on % lipid

    Args
    ----
    p_lipid: ndarray
        array of percent lipid values from which to calculate body composition

    Returns
    -------
    perc_comps: pandas.Dataframe
        dataframe of percent composition values from percent lipids
    '''
    import pandas

    p_comps = pandas.DataFrame(index=range(len(p_lipid)))

    p_comps['perc_lipid']   = p_lipid
    p_comps['perc_water']   = 71.4966 - (0.6802721 * p_lipid)
    p_comps['perc_protien'] = (0.42 * p_comps['perc_water']) - 4.75
    p_comps['perc_ash']     = 100 - (p_lipid + p_comps['perc_water'] + \
                                     p_comps['perc_protien'])

    return p_comps


def water_from_lipid_protien(lipid, protein):
    '''Calculate total body water from total lipid and protein

    Parameters from solving original Fedak & Reilly eqs.
    '''
    return -4.408148e-16+(2.828348*protein) + (1.278273e-01*lipid)


def lip2dens(p_lipid, lipid_dens=0.9007, prot_dens=1.34, water_dens=0.994, a_dens=2.3):
    '''Derive tissue density from lipids'''

    p_comps = perc_bc_from_lipid(p_lipid)

    p_comps['density'] = (lipid_dens * (0.01 * p_comps['perc_lipid'])) + \
                         (prot_dens  * (0.01 * p_comps['perc_protien'])) + \
                         (water_dens * (0.01 * p_comps['perc_water'])) + \
                         (a_dens     * (0.01 * p_comps['perc_ash']))
    return p_comps


def dens2lip(seal_dens, lipid_dens=0.9007, prot_dens=1.34, water_dens=0.994, a_dens=2.3):
    '''density to lipid

    Args
    ----
    seal_dens: ndarray
        An array of seal densities (g/cm^3), must be broadcastable array. The
        calculations only yield valid percents with densities between
        0.888-1.123 wit other parameters left as defaults.
    '''

    ad_numerat =  -3.2248 * a_dens
    pd_numerat = -25.2786 * prot_dens
    wd_numerat = -71.4966 * water_dens

    ad_denom = -0.034  * a_dens
    pd_denom = -0.2857 * prot_dens
    wd_denom = -0.6803 * water_dens

    p_lipid = ((100 * seal_dens) + ad_numerat + pd_numerat + wd_numerat) / \
              (lipid_dens + ad_denom + pd_denom + wd_denom)

    p_all = lip2dens(p_lipid)
    p_all = p_all[['perc_water', 'perc_protien', 'perc_ash']]

    p_all['density'] = seal_dens
    p_all['perc_lipid'] = p_lipid

    return p_all


def buoyant_force(dens, vol, sw_dens=1.028):
    return (1000 * (sw_dens - dens)) * (1000 * vol) * 0.00981


def total_buoyant_force(dens_kgm3, vol_m3, sw_dens=1028.0):
    '''Cacluate buoyant force of object in flued

    Density (fluid displaced) * volume (displaced) * gravity
    '''
    g  = 9.80665                  # m/s^2
    bf = sw_dens * vol_m3 * g     # kg/m^3 * m^3 * m/s^2 = kg*m/s^2 (N)
    w  = -((dens_kgm3 * vol_m3) * g)

    return bf + w


def diff_speed(sw_dens=1.028, seal_dens=1.053, seal_length=300, seal_girth=200,
        CD=0.09):
    import numpy

    surf, vol = surf_vol(seal_length, seal_girth)

    BF = buoyant_force(seal_dens, vol, sw_dens)

    V = 2 * (BF/(CD * sw_dens * (surf*1000)))

    if V >= 0:
        V = numpy.sqrt(V)
    else:
        V = -numpy.sqrt(-V)

    return V


def lip2en(BM, perc_lipid):
    '''lipids to percent total body weight'''
    PTBW = 71.4966 - (0.6802721*perc_lipid)
    return (40.8*BM) - (48.5*(0.01*PTBW*BM)) - 0.4


def surf_vol(length, girth):
    import numpy

    ar   = 0.01 * girth / (2 * numpy.pi)
    stll = 0.01 * length
    cr   = stll / 2
    e    = numpy.sqrt(1-(ar**2/cr**2))

    surf = ((2*numpy.pi * ar**2) + \
           (2 * numpy.pi * ((ar * cr)/e)) * 1/(numpy.sin(e)))

    vol  = (((4/3) * numpy.pi)*(ar**2) * cr)

    return surf, vol


def calc_seal_volume(mass_kg, dens_kgm3, length=None, girth=None):
    '''Cacluate seal volume from mass and either density or length and girth'''
    if (length is not None) and (girth is not None):
        _, seal_vol = surf_vol(length, girth)
    else:
        seal_vol = mass_kg / dens_kgm3

    return seal_vol
