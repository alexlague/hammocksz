# Run map-level inference

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

do_plot = False
run_hmc = True
## LOAD CMB MAP & TEMPLATE ##


cmb_map = np.loadtxt('maps/cmb_map_with_noise.dat')
ksz_map = np.loadtxt('maps/ksz_map.dat') #np.loadtxt('maps/ksz_map.dat')
temp_map = np.loadtxt('maps/template_map.dat') #np.loadtxt('maps/template_map.dat')

cmb_map = jnp.array(cmb_map + ksz_map)
temp_map = jnp.array(temp_map)

cmb_fourier = jnp.fft.rfftn(cmb_map)

N = len(cmb_map)

# error on the mean fourier coeffs
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

# Precompute field variables
delta_k = jnp.fft.fftn(delta_field)
bd2_field_k =  jnp.fft.fftn(bd2_field)
b2_field_k =  jnp.fft.fftn(b2_field)
bG2_field_k = jnp.fft.fftn(bG2_field)

## DEFINE MODEL ##

def ksz_forward_model(b1, bd2, b2, bG2):
    
    delta_e_k = b1 * delta_k + b2 * b2_field_k + bd2 * bd2_field_k + bG2 * bG2_field_k
    delta_e = jnp.fft.ifftn(delta_e_k).real

    updated_template = jnp.sum((1+delta_e) * v_field, axis=0)
    
    return updated_template  / 3e3 # TODO fix normalization




def model():

    # define parameters (incl. prior ranges)
    with numpyro.plate('Npix', len(jnp.ravel(cmb_fourier))):
        #pix = numpyro.sample('pix', dist.LogUniform(1.e-3, 1.e3))
        pixels_fourier = numpyro.sample('pix', dist.Normal(jnp.ravel(means), jnp.ravel(stds)))
        phases_fourier = numpyro.sample('phi', dist.Uniform(-jnp.pi, jnp.pi))
        
    with numpyro.plate('B', 4):
        biases = numpyro.sample('biases', dist.Uniform(-2, 2))

    #pixels_map = jnp.reshape(pix, (N, N))
    
    # implement the model
    kSZ_model = ksz_forward_model(*biases)
    # compute FFT coeffs
    #cmb_fourier = jnp.fft.rfftn(cmb)
    kSZ_model_fourier = jnp.ravel(jnp.fft.rfftn(kSZ_model)) + pixels_fourier * jnp.exp(1j * phases_fourier)

    
    # notice that we clamp the outcome of this sampling to the observation y
    numpyro.sample('obs', dist.Normal(jnp.abs(jnp.ravel(cmb_fourier)-kSZ_model_fourier), stds), obs=0.)

    #numpyro.sample('obs', dist.Gamma(diff_real+diff_imag, rate=4.), obs=0.00001)


if run_hmc:
    
    ## INITIALIZE LIKELIHOOD ##
    # need to split the key for jax's random implementation
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)
    ini_noise = jnp.array(np.random.normal(cmb_map.mean(), cmb_map.std()/2.5, size=cmb_map.shape))
    ini_noise_fourier = jnp.fft.rfft(ini_noise)
    ini_fourier = jnp.ravel(cmb_fourier + ini_noise_fourier)  
    inisample = {"pix":jnp.abs(ini_fourier), "phi":jnp.angle(ini_fourier), "biases":jnp.array([1.0, 0., 0., 0.])}

    ## RUN & SAVE CHAINS ##
    # run HMC with NUTS
    #ernel = HMC(model, target_accept_prob=0.8,  init_strategy = numpyro.infer.util.init_to_value(values=inisample))
    kernel = NUTS(model, target_accept_prob=0.7, init_strategy = numpyro.infer.util.init_to_value(values=inisample))
    mcmc = MCMC(kernel, num_warmup=200, num_samples=2000)
    mcmc.run(rng_key_, )#cmb=cmb_map, temp=temp_map, err=err) #x=x, y=y, y_err=y_err)
    mcmc.print_summary()
    
    #pixels_out = np.mean(10**mcmc.get_samples()["pix"], axis=0)

    #np.savetxt('maps/hmc_out_map.dat', pixels_out.reshape(N, N))

    
    
if do_plot:
    import matplotlib.pyplot as plt
    plt.imshow(ksz_map-temp_map)
    plt.colorbar()
    plt.show()
    
    plt.imshow(ksz_map- (pixels_out.reshape(N, N) * temp_map))
    plt.colorbar()
    plt.show()

    plt.imshow(pixels_out.reshape(N, N))
    plt.colorbar()
    plt.show()

#chain = np.array([mcmc.get_samples()[x] for x in labels]).T
#print(np.mean(chain, axis=0)[::100])

do_test = False
if do_test:
    kSZ_model = temp_map * pixels_out.reshape(N, N)

    cmb_fourier = jnp.fft.fftn(cmb_map)
    kSZ_model_fourier = jnp.fft.fftn(kSZ_model)
    temp_fourier =  jnp.fft.fftn(temp_map)
    kSZ_truth_fourier = jnp.fft.fftn(ksz_map)

    auto_template = jnp.ravel((kSZ_model_fourier * jnp.conjugate(kSZ_model_fourier))) / 1e9
    cross_template = jnp.ravel((cmb_fourier * jnp.conjugate(kSZ_model_fourier))) /1e9
    true_cross = jnp.ravel((cmb_fourier * jnp.conjugate(kSZ_truth_fourier))) /1e9
    testing = jnp.ravel((temp_fourier * jnp.conjugate(temp_fourier))) / 1e9
    true_ksz = jnp.ravel((kSZ_truth_fourier * jnp.conjugate(kSZ_truth_fourier))) / 1e9

    model_truth = jnp.ravel((kSZ_truth_fourier * jnp.conjugate(kSZ_model_fourier))) / 1e9
    template_truth = jnp.ravel((kSZ_truth_fourier * jnp.conjugate(temp_fourier))) / 1e9
    
    #plt.hist(np.log10(abs(testing-cross_template)), bins=100, label='Reference')
    #plt.hist(np.log10(abs(auto_template-cross_template)), bins=100, label='Optimized', alpha=0.5)
    #plt.scatter(k2d, cross_template-testing, label='Initial Template', marker='.')
    plt.scatter(k2d, (cross_template-auto_template).real + (cross_template-auto_template).imag, label='HMC Result')
    plt.scatter(k2d, (true_cross-true_ksz).real + (true_cross-true_ksz).imag, label='Optimal', marker='.')
    #plt.scatter(k2d, err, marker='x')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    from scipy.stats import binned_statistic
    bin_edges = np.linspace(0, 0.7, 26)
    bin_cen = np.linspace(0, 0.7, 25)
    r_template = binned_statistic(np.ravel(k2d), (template_truth).real / np.sqrt(np.abs(true_ksz)*np.abs(testing)), bins=bin_edges)[0]
    r_hmc = binned_statistic(np.ravel(k2d), (model_truth).real / np.sqrt(np.abs(true_ksz)*np.abs(auto_template)), bins=bin_edges)[0]
    
    #plt.scatter(k2d, (model_truth).real / np.sqrt(np.abs(true_ksz)*np.abs(auto_template)), label='HMC Result')
    #plt.scatter(k2d, (template_truth).real / np.sqrt(np.abs(true_ksz)*np.abs(testing)), label='Original template')
    #plt.ylim(-1.1, 1.1)
    plt.plot(bin_cen, r_template, label='Template')
    plt.plot(bin_cen, r_hmc, label='HMC')
    plt.legend()
    plt.show()
    
