# Generate CMB map

import numpy as np
import healpy as hp
import camb
from camb import model
import matplotlib.pyplot as plt

do_plot = True
grid = 64
ang_size = 5 # deg

# Noise properties (assuming ACT)
#noise_level = 15. # uK arcmin
#cmb_beam = 1. # arcmin

# Noise properties (assuming goal SO)
noise_level = 6.
cmb_beam = 1. # need to double check this

noise_level *=  2.9e-4 # arcmin to rad
cmb_beam *= 2.9e-4 # arcmin to rad


# Compute CMB Cls
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=6000)

results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
cls=powers['total'][:,0]
ls = np.arange(len(cls))
cls[0] = 0.
cls[1] = 0.
cls[2:] /= (ls[2:] * (ls[2:]+1))
cls[2:] *= 2 * np.pi

# Compute CMB Nls (Knox formula) CHECK
nls = noise_level**2. * np.exp(ls * (ls+1) * cmb_beam**2.)

if do_plot:
    plt.semilogy(ls, ls * (ls+1)*cls/2/np.pi)
    plt.semilogy(ls, ls * (ls+1)*nls/2/np.pi)
    plt.show()

    
alms = hp.sphtfunc.synalm(cls+nls)
cmb = hp.sphtfunc.alm2map(alms, nside=1024)

cmb_square = hp.visufunc.cartview(cmb, xsize=grid, ysize=grid, lonra=[25,25+ang_size], latra=[10,10+ang_size], return_projected_map=True)

if do_plot:
    plt.imshow(cmb_square)
    plt.colorbar()
    plt.show()

# Save map for inference
np.savetxt('maps/cmb_map_with_noise.dat', cmb_square)

compute_variances = True

from tqdm import tqdm

if compute_variances:
    nsamples = 150
    maps = np.zeros((grid, grid//2+1, nsamples))
    #maps_imag = np.zeros((grid, grid, nsamples))
    
    for i in tqdm(range(nsamples)):
        alms = hp.sphtfunc.synalm(cls+nls)
        cmb = hp.sphtfunc.alm2map(alms, nside=1024)
        cmb_square = hp.visufunc.cartview(cmb, xsize=grid, ysize=grid, lonra=[25,25+ang_size], latra=[10,10+ang_size], return_projected_map=True)
        
        cmb_square_fourier = np.fft.rfftn(cmb_square)
        maps[:, :, i] = np.abs(cmb_square_fourier)
        #maps_imag[:, :, i] = cmb_square_fourier.imag 
        plt.close()
        
    maps_mean = np.mean(maps, axis=2)
    maps_std = np.std(maps, axis=2)

    np.savetxt('maps/cmb_maps_fourier_means.dat', maps_mean)
    np.savetxt('maps/cmb_maps_fourier_stds.dat', maps_std)
