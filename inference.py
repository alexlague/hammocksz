# Run map-level inference

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

do_plot = True

## LOAD CMB MAP & TEMPLATE ##


#cmb_map = np.loadtxt('maps/simple_cmb_map.dat')#[::16, ::16]
#temp_map = np.loadtxt('maps/simple_template_map.dat')#[::16, ::16]

cmb_map = np.loadtxt('maps/cmb_map_with_noise.dat') *0.1
ksz_map = np.loadtxt('maps/ksz_map.dat')
temp_map = np.loadtxt('maps/template_map.dat') 
#temp_map = ksz_map + 10.**np.random.normal(0, 0.25, ksz_map.shape)

cmb_map = jnp.array(cmb_map + ksz_map)
temp_map = jnp.array(temp_map)

N = len(cmb_map)

freqs = np.fft.fftfreq(N)
kx, ky = np.meshgrid(freqs, freqs)
k2d = (kx**2. + ky**2.)**0.5
#err = 1. / (np.ravel(k2d)+1.0e-3) / 3000.
#err = jnp.array(err)
err = 1e-4

## DEFINE MODEL ##
def model(cmb, temp, err):

    # define parameters (incl. prior ranges)
    with numpyro.plate('N2', N*N):
        #pix = numpyro.sample('pix', dist.LogUniform(1.e-3, 1.e3))
        pix = numpyro.sample('pix', dist.Uniform(-1.5, 1.5))

    pixels_map = jnp.reshape(10.**pix, (N, N))
    
    # implement the model
    kSZ_model = temp * pixels_map

    # compute FFT coeffs
    cmb_fourier = jnp.fft.fftn(cmb)
    kSZ_model_fourier = jnp.fft.fftn(kSZ_model)
    temp_fourier =  jnp.fft.fftn(temp)
    
    #auto_template = jnp.ravel((kSZ_model_fourier * jnp.conjugate(kSZ_model_fourier))) / 1e9 
    #cross_template = jnp.ravel((cmb_fourier * jnp.conjugate(kSZ_model_fourier))) /1e9

    auto_template = jnp.ravel((kSZ_model_fourier * jnp.conjugate(temp_fourier))) / 1e9
    cross_template = jnp.ravel((cmb_fourier * jnp.conjugate(temp_fourier))) /1e9
    
    diff_real = auto_template.real-cross_template.real
    diff_imag = auto_template.imag-cross_template.imag
    
    # notice that we clamp the outcome of this sampling to the observation y
    numpyro.sample('obs', dist.Normal(diff_real+diff_imag, err), obs=0.)
    #numpyro.sample('obs', dist.Gamma(diff_real+diff_imag, rate=4.), obs=0.00001)


if run_hmc:
    
    ## INITIALIZE LIKELIHOOD ##
    # need to split the key for jax's random implementation
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)
    inisample = {"pix":jnp.ones(N**2)}#jnp.array(np.random.uniform(-0.7, 0.7, size=N**2))}

    ## RUN & SAVE CHAINS ##
    # run HMC with NUTS
    #ernel = HMC(model, target_accept_prob=0.8,  init_strategy = numpyro.infer.util.init_to_value(values=inisample))
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy = numpyro.infer.util.init_to_value(values=inisample))
    mcmc = MCMC(kernel, num_warmup=200, num_samples=2000)
    mcmc.run(rng_key_, cmb=cmb_map, temp=temp_map, err=err) #x=x, y=y, y_err=y_err)
    mcmc.print_summary()
    
    pixels_out = np.mean(10**mcmc.get_samples()["pix"], axis=0)

    np.savetxt('maps/hmc_out_map.dat', pixels_out.reshape(N, N))

    
    
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

do_test = True
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
    
