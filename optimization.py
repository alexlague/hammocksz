# Run optimization instead of HMC sampling

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import optax
import tqdm
from scipy.stats import binned_statistic

cmb_map = np.loadtxt('maps/cmb_map_with_noise.dat') * 1.0
ksz_map = np.loadtxt('maps/ksz_map.dat') #np.loadtxt('maps/ksz_map.dat')
temp_map = np.loadtxt('maps/template_map.dat') #np.loadtxt('maps/template_map.dat')

N = len(cmb_map)

#ksz_map[np.isnan(np.log10(ksz_map/temp_map))] = temp_map[np.isnan(np.log10(ksz_map/temp_map))]
#plt.imshow(np.log10(ksz_map/temp_map))
#plt.colorbar()
#plt.show()

#import sys
#sys.exit()

cmb_map = jnp.array(cmb_map + ksz_map)
temp_map = jnp.array(temp_map)
#temp_fourier =  jnp.fft.fftn(temp_map)
#temp2_fourier = jnp.fft.fftn(temp_map**2)

# error on the mean fourier ceoffs
means = jnp.ravel(np.loadtxt('maps/cmb_maps_fourier_means.dat'))
stds = jnp.ravel(np.loadtxt('maps/cmb_maps_fourier_stds.dat'))

# Load fields
delta_field = jnp.array(np.load('fields/template_delta_field.npy'))
v_field = jnp.array(np.load('fields/template_vel_field.npy'))

freqs = jnp.fft.fftfreq(N)
kx2d, ky2d = jnp.meshgrid(freqs, freqs)
k2d = (kx2d**2. + ky2d**2.)**0.5

bd2_field = jnp.array(np.load('fields/bd2_field.npy'))
b2_field = jnp.array(np.load('fields/b2_field.npy'))
bG2_field = jnp.array(np.load('fields/bG2_field.npy'))

#kx, ky, kz = jnp.meshgrid(freqs, freqs, freqs)
#kNyq = np.max(kx)/(np.pi*grid/BoxSize)
#kx /= kNyq
#ky /= kNyq
#kz /= kNyq
#knorm = (kx**2. + ky**2. + kz**2.)**0.5

# Precompute field variables
#Phi_k = jnp.fft.fftn(delta_field) / jnp.where(knorm!=0, knorm**2, np.inf)
delta_k = jnp.fft.fftn(delta_field)
bd2_field_k =  jnp.fft.fftn(bd2_field)
b2_field_k =  jnp.fft.fftn(b2_field)
bG2_field_k = jnp.fft.fftn(bG2_field)
#delta2_k = jnp.fft.fftn(delta_field**2.)
#G2_k = ((kx * ky)**2 + (kx * kz)**2 + (ky * kz)**2) * Phi_k**2


#G2_temp_fourier = (kx * ky * temp_fourier) **2 - jnp.fft.fftn(jnp.fft.ifftn(k2d**2 * temp_fourier)**2)  # check signs; from 2205.06270 Eq 27

@jax.jit
def ksz_forward_model(b1, bd2, b2, bG2):
    
    #Phi_k = jnp.fft.fftn(delta_field) / jnp.where(knorm!=0, knorm**2, np.inf)
    #delta_k = jnp.fft.fftn(delta_field)
    #delta2_k = jnp.fft.fftn(delta_field**2.)
    #G2_k = ((kx * ky)**2 + (kx * kz)**2 + (ky * kz)**2) * Phi_k**2
    
    #delta_e_k = b1 * delta_k + b2 * delta2_k + bd2 * knorm**2 * delta_k + bG2 * G2_k
    delta_e_k = b1 * delta_k + b2 * b2_field_k + bd2 * bd2_field_k + bG2 * bG2_field_k
    delta_e = jnp.fft.ifftn(delta_e_k).real

    updated_template = jnp.sum((1+delta_e) * v_field, axis=0)
    
    return (updated_template  - 0*jnp.mean(updated_template))  / 3e3 #* jnp.std(temp_map) / jnp.std(updated_template)

def boxcar(x, a, b):
    return (jnp.heaviside(x-a, 1.) - jnp.heaviside(x-b, 1.)) / (b-a)


@jax.jit
def compute_loss(pixels_and_biases):
    #amplitudes = pixels_and_biases[:N*(N//2+1)]
    #phases = pixels_and_biases[N*(N//2+1):2*N*(N//2+1)]
    pixels = pixels_and_biases[:N**2]
    pixels_map = jnp.reshape(pixels, (N, N))
    pixels_fourier = jnp.ravel(jnp.fft.rfftn(pixels_map))
    amplitudes = jnp.abs(pixels_fourier)
    b1, bd2, b2, bG2 = pixels_and_biases[N**2:]
    #amplitudes_map = jnp.reshape(amplitudes, (N, (N//2+1)))
    #phases_map = jnp.reshape(phases, (N, (N//2+1)))
    #pixels_fourier = amplitudes * jnp.exp(phases*1j)
    
    # implement the model
    #kSZ_model_fourier = b1 * temp_fourier + b2 * k2d**2 * temp_fourier + b3 * temp2_fourier + b4 * G2_temp_fourier
    kSZ_model = ksz_forward_model(b1, bd2, b2, bG2)
    kSZ_model_fourier = jnp.ravel(jnp.fft.rfftn(kSZ_model)) + pixels_fourier

    norm =  1. #jnp.ravel(k2d+0.1)**2
    
    # compute FFT coeffs
    cmb_fourier = jnp.ravel(jnp.fft.rfftn(cmb_map))
    #kSZ_model_fourier = jnp.fft.fftn(kSZ_model)
    #temp_fourier =  jnp.fft.fftn(temp_map)

    '''
    auto_template = jnp.ravel((kSZ_model_fourier * jnp.conjugate(kSZ_model_fourier))) / 1e9
    cross_template = jnp.ravel((cmb_fourier * jnp.conjugate(kSZ_model_fourier))) /1e9

    diff_real = auto_template.real-cross_template.real
    diff_imag = auto_template.imag-cross_template.imag

    loss1 = jnp.sum((diff_real**2. + diff_imag**2) / norm)
    
    auto_template = jnp.ravel((kSZ_model_fourier * jnp.conjugate(temp_fourier))) / 1e9
    cross_template = jnp.ravel((cmb_fourier * jnp.conjugate(temp_fourier))) /1e9
    
    diff_real = auto_template.real-cross_template.real
    diff_imag = auto_template.imag-cross_template.imag

    loss2 = jnp.sum((diff_real**2. + diff_imag**2) / norm) #- 0.001*jnp.sum(jnp.abs(auto_template))

    return (loss2 + loss1)
    '''

    prior = jnp.sum((amplitudes-means)**2/2/stds**2) #+ boxcar(phases, 0, 2*np.pi)
    loss = jnp.sum(jnp.abs(cmb_fourier-kSZ_model_fourier)**2/2/stds**2)

    #debug test loss
    #loss = jnp.sum((jnp.array([b1, bd2, b2, bG2]) - jnp.array([1., -0.2, 0.3, 1e-3]))**2) + jnp.sum(pixels**2)
    
    return loss #+ prior 
    
#print(compute_loss(np.concatenate((ksz_map / temp_map), np.array([1, 0, 0]))))

def compute_cross_corr(map1, map2):
    '''
    '''
    freqs = np.fft.fftfreq(N)
    kx2d, ky2d = np.meshgrid(freqs, freqs)
    k2d = (kx2d**2. + ky2d**2.)**0.5

    bin_edges = np.linspace(0, 0.7, 26)
    bin_cen = np.linspace(0, 0.7, 25)

    map1_fft = np.fft.fftn(map1)
    map2_fft = np.fft.fftn(map2)

    cross = np.ravel(map1_fft * np.conjugate(map2_fft))
    auto1 = np.ravel(map1_fft * np.conjugate(map1_fft))
    auto2 = np.ravel(map2_fft * np.conjugate(map2_fft))

    r = binned_statistic(np.ravel(k2d), cross.real / np.sqrt(np.abs(auto1)*np.abs(auto2)), bins=bin_edges)[0]
    
    return bin_cen, r


# Test to make sure likelihood is working

b1_true, bd2_true, b2_true, bG2_true = 1., -0.2, 0.3, 1e-3 #true values
params = jnp.zeros(N**2+4)
params = params.at[:N**2].set(jnp.array(np.loadtxt('maps/cmb_map_with_noise.dat').ravel()))
params = params.at[N**2].set(b1_true)
params = params.at[N**2+1].set(bd2_true)
params = params.at[N**2+2].set(b2_true)
params = params.at[N**2+3].set(bG2_true)

#print(params[-4:])
#print(ksz_map.mean())
#print(ksz_forward_model(b1_true, bd2_true, b2_true, bG2_true).mean())
#print(np.abs(ksz_map-ksz_forward_model(b1_true, bd2_true, b2_true, bG2_true))/ksz_map)
print("Optimal loss: ", compute_loss(params)) # should be ~0 if cmb and noise are 0

#plt.imshow(ksz_map)
#plt.colorbar()
#plt.show()
#plt.imshow(ksz_forward_model(b1_true, bd2_true, b2_true, bG2_true))
#plt.colorbar()
#plt.show()


key = jax.random.PRNGKey(42)
start_learning_rate = 5e-4

#optimizer = optax.adam(start_learning_rate)
#optimimzer = optax.lbfgs(start_learning_rate)
#optimimzer = optax.radam(start_learning_rate)
optimizer = optax.adam(start_learning_rate)
#optimizer = optax.adagrad(start_learning_rate)

# Initialize parameters of the model + optimizer.
params = jnp.zeros(N**2+4)
#initial_amp = jnp.array(np.random.normal(means, stds, size=means.shape))
#initial_phases = jnp.array(np.random.uniform(0, 2*np.pi, size=means.shape))

#params = jnp.zeros(2*N*(N//2+1) + 4)
#params = params.at[:N*(N//2+1)].set(initial_amp)
#params = params.at[N*(N//2+1):2*N*(N//2+1)].set(initial_phases)
#params = params.at[2*N*(N//2+1)].set(1.) # bG2 needs to be low
params = params.at[:N**2].set(jnp.ravel(cmb_map) + jnp.array(np.random.normal(0., cmb_map.std()/100, size=N**2)))
params = params.at[N**2].set(1.)
print(params[N**2:])
loss_ini = compute_loss(params)
print(loss_ini)
opt_state = optimizer.init(params)

# A simple update loop.
N_epochs_adam = 100_000
N_epochs_bfgs = 10_000
N_epochs = N_epochs_adam + N_epochs_bfgs
losses = np.zeros(N_epochs)
for i in tqdm.tqdm(range(N_epochs_adam)):
    losses[i], grads = jax.value_and_grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if losses[i] == np.min(losses[losses!=0.]):
        opt_params = params

optimimzer = optax.lbfgs(1e-3)

for i in tqdm.tqdm(range(N_epochs_bfgs)):
    losses[i+N_epochs_adam], grads = jax.value_and_grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if losses[i] == np.min(losses[losses!=0.]):
        opt_params = params

'''
plt.imshow(ksz_map)
plt.colorbar()
plt.show()

plt.imshow(temp_map * opt_params.reshape(N, N))
plt.colorbar()
plt.show()
'''

#b1, bd2, b2, bG2 = opt_params[2*N*(N//2+1):] #opt_params[N**2:]
b1, bd2, b2, bG2 = opt_params[N**2:] 
print(b1, bd2, b2, bG2)
#kSZ_model = b1 * temp_fourier + b2 * k2d**2 * temp_fourier + b3 * temp2_fourier + b4 * G2_temp_fourier
#kSZ_model =jnp.fft.ifftn(kSZ_model).real
kSZ_model = ksz_forward_model(b1, bd2, b2, bG2)

opt_output = kSZ_model #temp_map * opt_params.reshape(N, N)

plt.imshow(abs(ksz_map - temp_map))
plt.colorbar()
plt.show()

plt.imshow(abs(ksz_map - opt_output))
plt.colorbar()
plt.show()

kbins, corr_in = compute_cross_corr(ksz_map, temp_map)
kbins, corr_out = compute_cross_corr(ksz_map, opt_output)

plt.plot(kbins, corr_in, label='Template')
plt.plot(kbins, corr_out, label='Optimization')
plt.legend()
plt.xlabel('$k$')
plt.ylabel('$r(k)$')
plt.savefig('2D_correlation_output.png')
plt.show()

plt.plot(np.arange(N_epochs), losses) 
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss function')
plt.savefig('loss.png')
plt.show()
