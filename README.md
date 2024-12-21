# hammocksz

## generate_cmb_map

This script creates a set of CMB maps including primary CMB (lensed) and noise.

## generate_ksz_map

This script creates a density field with velocities and then generates a "true" ksz map and a template ksz map (the two are correlated on large scales but differ on small scales). The "true" ksz map is constructed using an EFT approach from the template map assuming fixed biases up to second order.

## optimization

This script finds the best fit point (pixel value and ksz biases) using gradient descent (ADAM).

## inference

This script is the main inference algorithm using the NUTS HMC smapler in numpyro. (In development).