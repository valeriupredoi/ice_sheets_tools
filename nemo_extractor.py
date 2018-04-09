import iris
import matplotlib.pyplot as plt
import numpy as np
import sys

use_str="""Usage: python nemo_extractor.py --savecubes --verbose --data=<grid_T datafile> --mesh=<grid_T meshfile> --mask=<grid_T maskfile>"""

l_verbose = any(arg == '--verbose' for arg in sys.argv)
l_savecubes = any(arg == '--savecubes' for arg in sys.argv)
datafile = None
maskfile = None
meshfile = None
for arg in sys.argv:
    if arg.split('=')[0]=='--data':
        datafile=arg.split('=')[1]
    if arg.split('=')[0]=='--mask':
        maskfile=arg.split('=')[1]
    if arg.split('=')[0]=='--mesh':
        meshfile=arg.split('=')[1]

if datafile == None: 
    print(use_str)
    exit()

# Initial data load
# load nemo data cubelist
datas = iris.load(datafile)
print('Data cube')
print('---------------')
print(datas)

if maskfile is None:
  maskfile = 'nemo_antarctic_regions_eORCA025.nc'
  print('WARNING: Using default maskfile: %s' % maskfile)
  antarctic_mask = iris.load(maskfile)
else:
  antarctic_mask = iris.load(maskfile)

print('\nAntarctic Mask cube')
print('-----------------------')
print(antarctic_mask)

# load dimensions cube
if meshfile is None:
  meshfile = 'mesh_mask_745_normal-NEW.nc'
  print('WARNING: Using default meshfile: %s' % meshfile)
  c=iris.load(meshfile)
else:
  c=iris.load(meshfile)

print('Running with arguments:')
print("l_verbose", l_verbose)
print("l_savecubes", l_savecubes)
print("Data file", datafile)
print("Mask file", maskfile)
print("Mesh file", meshfile)
##############################################
################## FUNCTIONS #################
##############################################
def x_label(var_name):
    """Set ylabels corresponding to vars"""
    if var_name == 'Temperature':
        x_l = 'average T[C]'
    elif var_name == 'Salinity':
        x_l = 'average Salinity per 1000'

    return x_l


def _make_3d(two_dim_array, z_dim):
    """Make a 3d array from a 2d array with z:z_dim"""
    three_dim_shape = (1, z_dim, two_dim_array.shape[1], two_dim_array.shape[2])
    three_dim_array = np.empty(three_dim_shape)
    for i in range(z_dim):
        three_dim_array[:, i, :] = two_dim_array
    return three_dim_array


def _get_nemo_mask(nm_mask):
    """Apply a 1-0 NEMO 3d mask"""
    inmask = np.zeros_like(nm_mask, bool)
    inmask[nm_mask == 0] = True
    inmask[nm_mask == 1] = False
    return inmask


def _apply_mask(mask, nemo_mask, var_data):
    """Apply a 1-0 mask"""
    imask = np.zeros_like(mask.data, bool)
    imask[mask.data == 0.] = True
    imask[mask.data == 1.0] = False
    var_mask = np.zeros(var_data.shape, bool)
    for i in range(var_data.shape[1]):
        var_mask[:,i] = imask
    nemo_mask |= var_mask
    for m in range(var_mask.shape[1]):
        if np.all(var_mask[:, m]):
            print('WARNING: all values are \
                   masked at depth level %i' % m)
    var_data_masked = np.ma.array(var_data,
                                  mask=var_mask,
                                  fill_value=1e+20)
    return var_data_masked


def _save_depth_cubes(var_data_masked,
                      orig_cube,
                      mask,
                      cube_name,
                      coeff):
    """Save cubes per examined region coeff"""
    times = orig_cube.coords('time')[0]
    plev = orig_cube.coords('depth')[0]
    lats = mask.coord('latitude')
    lons = mask.coord('longitude')
    cspec = [(times, 0), (plev, 1), (lats, 2), (lons, 3)]
    var_masked_cube = iris.cube.Cube(var_data_masked,
                                     standard_name=orig_cube.standard_name,
                                     long_name=cube_name,
                                     dim_coords_and_dims=cspec)
    save_name = cube_name + '_Region_' + _region_name(coeff) + '.nc'
    iris.save(var_masked_cube, save_name)


def _compute_average(var_avg_dict, var_data_masked, wgts, coeff):
    """Compute weighted average"""
    region_name = 'Region_' + _region_name(coeff)
    shape = var_data_masked.shape
    # average over 3dim with 3dim weights array
    var_avg = np.ma.average(var_data_masked, axis=(2,3), weights = wgts)
    var_avg_dict[region_name] = var_avg[0, :]
    return var_avg_dict


def _plot_var(var_avg_dict, depth_points, var_name):
    """Plot variables"""
    for el in var_avg_dict.keys():
        plt.plot(var_avg_dict[el], depth_points, label=el)
        plt.grid()
        plt.ylim(0., 2500.)
        plt.ylabel('Depth [m]')
        xl = x_label(var_name)
        plt.xlabel(xl)
        plt.title(el)
        plt.legend()
        plot_name = var_name + '_' + el + '.png'
        plt.savefig(plot_name)
        plt.close()


def _region_name(indx):
    """Put names on idx-ed regions"""
    region_dict = {0: 'East_Antarctica-open',
                   1: 'Bellingshausen-open',
                   2: 'East_Antarctica-shelf',
                   3: 'Amundsen-open',
                   4: 'Weddell-shelf',
                   5: 'Amundsen-shelf',
                   6: 'Ross-shelf',
                   7: 'Weddell-open',
                   8: 'Ross-open',
                   9: 'Bellingshausen-shelf'}

    return region_dict[indx]

############### DATA OPEARTIONS #################
# dimensions
# dim1
glamt = c.extract('glamt')[0].data
# dim2
gphit = c.extract('gphit')[0].data
# dx
e1t = c.extract('e1t')[0].data
# dy
e2t = c.extract('e2t')[0].data
# dz
e3t0 = c.extract('e3t_0')[0].data
# z
gdept0 = c.extract('gdept_0')[0].data
# nemo mask
nemo_mask = c.extract('tmask')[0].data
# make e1t and e2t 3-dim arrays with z: e3t0.shape[1]
e1t_3d = _make_3d(e1t, e3t0.shape[1])
e2t_3d = _make_3d(e2t, e3t0.shape[1])

temp_cube=datas.extract('sea_water_potential_temperature')[0]
sal_cube=datas.extract('sea_water_salinity')[0]

# data
t_w = temp_cube.data
s_w = sal_cube.data
#t_w_cube = iris.cube.Cube(t_w)
#s_w_cube = iris.cube.Cube(s_w)
#if l_savecubes: iris.save(t_w_cube, 'temperature.nc')
#if l_savecubes: iris.save(s_w_cube, 'salinity.nc')
depth_points = temp_cube.coords('depth')[0].points
weights = e1t_3d * e2t_3d * e3t0
#want to do averages only below 200m
#e3t0[np.where(gdept0 < 200] = 0.)

# assemble user dict
var_dict = {'Temperature': t_w,
            'Salinity': s_w}
cube_dict = {'Temperature': temp_cube,
            'Salinity': sal_cube}

# operate on variables
for var in var_dict.keys():
    if l_verbose:
        print('\nLooking at variable %s' % var)
        print('-----------------------------------')
    dict_v_avg = {}
    nemo_msk = _get_nemo_mask(nemo_mask)
    for j in range(len(antarctic_mask)):
        var_masked = _apply_mask(antarctic_mask[j], nemo_mask, var_dict[var])
        if l_verbose:
            print('\nLooking at region %s' % _region_name(j))
            print('-------------------------')
            print('Regional Mean: %.3f' % np.ma.mean(var_masked))
        dict_v_avg = _compute_average(dict_v_avg, var_masked,
                                      weights, j)
        if l_savecubes:
            _save_depth_cubes(var_masked,
                              cube_dict[var],
                              antarctic_mask[j], var, j)
    _plot_var(dict_v_avg, depth_points, var)
