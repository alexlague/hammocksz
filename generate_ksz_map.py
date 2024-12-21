# Generate kSZ map and template

import numpy as np
from scipy.stats import binned_statistic
import camb
from camb import model, initialpower
from pixell import enmap, utils, enplot, curvedsky
import density_field_library as DFL

do_plot = True

## Compute linear Pk ##

redshift = 1.0
ang_size = 5 # deg
ang_size *= np.pi / 180. # rad

pars =  camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[redshift,], kmax=10.0)

pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)
dA = results.angular_diameter_distance(redshift) * 0.675 # check h units

# Non-linear spectrum for electrons (for now until Pge model)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

## Velocity prefactor ##

f = (results.get_Omega("cdm", redshift) + results.get_Omega("baryon", redshift))**0.55
a = 1. / (1+redshift)
H = results.h_of_z(redshift) * 3e5
faH = f * a * H

## Parameters to simulation generate box ##

grid              = 64    #grid size
BoxSize           = dA * ang_size #Mpc/h
seed              = 1      #value of the initial random seed
Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
threads           = 1      #number of openmp threads
verbose           = True   #whether to print some information

print('Boxsize: ', BoxSize)

# read power spectrum; k and Pk have to be floats, not doubles
k, Pk = kh, pk[0]
k, Pk = k.astype(np.float32), Pk.astype(np.float32)

# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
                              BoxSize, threads, verbose)

#k, Pk = kh, pk_nonlin[0]
#k, Pk = k.astype(np.float32), Pk.astype(np.float32)

#df_3D_electrons = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
#                              BoxSize, threads, verbose)

def get_biased_field(delta, b1, bd2, b2, bG2):
    
    knorm = get_knorm()
    kx, ky, kz = get_ks()
    Phi_k = np.fft.fftn(delta) / np.where(knorm!=0, knorm**2, np.inf)
    delta_k = np.fft.fftn(delta)
    delta2_k = np.fft.fftn(delta**2.)
    G2_k = ((kx * ky)**2 + (kx * kz)**2 + (ky * kz)**2) * Phi_k**2
    
    out_k = b1 * delta_k + b2 * delta2_k + bd2 * knorm**2 * delta_k + bG2 * G2_k
    out = np.fft.ifftn(out_k).real
    
    return out

def get_ks():
    freqs = np.fft.fftfreq(grid)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs)

    kNyq = np.max(kx)/(np.pi*grid/BoxSize)
    kx /= kNyq
    ky /= kNyq
    kz /= kNyq

    return kx, ky, kz

def get_knorm():

    kx, ky, kz = get_ks()
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)

    return knorm

def get_pk(field1, field2):
    fourier1 = np.fft.fftn(field1) * ((BoxSize/grid**2)**3)**0.5
    fourier2 = np.fft.fftn(field2) * ((BoxSize/grid**2)**3)**0.5
    pk3d = fourier1 * np.conjugate(fourier2)

    knorm = get_knorm()

    pk_binned = binned_statistic(knorm.ravel(), np.abs(pk3d.ravel()), bins=np.logspace(-3, 0, 101), statistic='mean')[0]
    #pk_binned = jax_binned_pk(knorm.ravel(), np.abs(pk3d.ravel()), np.logspace(-3, 0, 101))
    
    return np.logspace(-3, 0, 100), pk_binned

def get_q(delta, delta_lin):
    '''
    Compute line-of-sight momentum field of density field delta
    The velocities are approximated from the linear field
    '''
    knorm = get_knorm()
    kz = get_ks()[-1]
    Phi = np.fft.fftn(delta_lin) / np.where(knorm!=0, knorm**2, np.inf)
    #Phi = np.nan_to_num(Phi)
    
    vel_z = np.fft.ifftn(-1j*kz * Phi).real *np.pi**0.5 ## check if new scaling needed for inverse FFT

    return (1.+delta) * vel_z * faH, vel_z * faH


def spherical_harm_flat_map(flat_map, lonra, latra, lmax):
    '''
    Convert a 2D numpy array to a full sky map
    '''

    # Define area of map using numpy
    # pixell wants the box in the following format:
    # [[dec_from, RA_from], [dec_to, RA_to]]
    # Note RA goes "from" left "to" right!
    box = np.array([[lonra[0], latra[1]], [lonra[1], latra[0]]]) * utils.degree
    
    # Define a map geometry
    # the width and height of each pixel will be .5 arcmin
    shape, wcs = enmap.geometry(pos=box, shape=flat_map.shape, proj='car')
    
    # Create an empty ndmap
    ndmap = enmap.zeros(shape, wcs=wcs)
    
    ndmap[:] = flat_map

    alms = curvedsky.map2alm(ndmap, lmax=lmax)

    cl = curvedsky.alm2cl(alms)
    
    return cl

## Project momentum field to 2D map ##
b1, bd2, b2, bG2 = 1., -0.2, 0.3, 1e-3
df_3D_electrons = get_biased_field(df_3D, b1, bd2, b2, bG2)

q_template, vel = get_q(df_3D, df_3D)
template_map = np.sum(q_template, axis=0) / 3e3

q_ksz, _ = get_q(df_3D_electrons, df_3D)
ksz_map = np.sum(q_ksz, axis=0) / 3e3 # TODO: change this to actual normalization!!!

if do_plot:
    import matplotlib.pyplot as plt
    plt.imshow(template_map)
    plt.colorbar()
    plt.show()
    
    
    plt.imshow(ksz_map)
    plt.colorbar()
    plt.show()

# also plot 3d pk vs expectation...


lonra = [0, ang_size]
latra = [0, ang_size]

cls = spherical_harm_flat_map(ksz_map, lonra, latra, 4000)

if do_plot:
    ls = np.arange(len(cls))
    plt.semilogy(ls, ls*(ls+1)*cls/2/np.pi)
    plt.show()


# Save fields
np.save('fields/template_delta_field.npy', df_3D)
np.save('fields/template_vel_field.npy', vel) 
np.save('fields/bd2_field.npy', get_biased_field(df_3D, 0., 1., 0., 0.))
np.save('fields/b2_field.npy', get_biased_field(df_3D, 0., 0., 1., 0.))
np.save('fields/bG2_field.npy', get_biased_field(df_3D, 0., 0., 0., 1.))

# Save maps for inference
np.savetxt('maps/template_map.dat', template_map)
np.savetxt('maps/ksz_map.dat', ksz_map)
