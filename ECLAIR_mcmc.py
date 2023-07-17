import os
import sys
import ECLAIR_parser
import numpy as np
from scipy.stats import truncnorm


### Parse input ini file
ini_fname = sys.argv[1]
ini = ECLAIR_parser.parse_ini_file(ini_fname)


### If a new chain is started, create copy of the ini file in output folder
if not ini["continue_chain"] and not ini["debug_mode"]:
    ECLAIR_parser.copy_ini_file(ini_fname, ini)


### Print output path root
print("#"*(25 + len(ini["output_root"])))
print(f"### Starting MCMC in {ini['output_root']} ###")
print("#"*(25 + len(ini["output_root"])))


### Import requested variant of class python wrapper
which_class = ini["which_class"]
exec(f"import {which_class} as classy")


### Import requested MCMC sampler
which_sampler = ini["which_sampler"]
if which_sampler == "emcee":
    import emcee as MCMCsampler
elif which_sampler == "zeus":
    import zeus as MCMCsampler


### Import and store in list all requested log(likelihoods)
lkl = []
for like_name in ini["likelihoods"]:
    exec(f"import likelihoods.{like_name}")
    exec(f"lkl.append(likelihoods.{like_name}.get_loglike)")


### Various useful quantities
# List of all variable names
var_names = [p[1] for p in ini["var_par"]]

# List of uniform priors ([lower, upper] bounds)
uni_pri = np.array([[p[3], p[4]] for p in ini["var_par"]])

# List of Gaussian priors ([mean, stddev])
gauss_pri = np.array([[p[1], p[2]] for p in ini["gauss_priors"]])

# List of indices of variables with Gaussian priors
ix_gauss_pri = np.array([var_names.index(p[0]) for p in ini["gauss_priors"]])

# List of names of derived parameters with Gaussian priors
drv_gauss_pri = [n[0] for n in ini["drv_gauss_priors"]]

# List of names of derived parameters with uniform priors
drv_uni_pri = [n[0] for n in ini["drv_uni_priors"]]

# "Bad result" to be returned by lnlike() if evaluation fails in any way
bad_res = tuple([-np.inf] * (2 + len(lkl) + len(ini["derivs"])))


### Actual loglike function
def lnlike(p):

    # Deal with uniform priors on MCMC parameters
    if not np.all((uni_pri[:, 0] <= p) & (p <= uni_pri[:, 1])):
        return bad_res

    # Deal with Gaussian priors on MCMC parameters
    lnp = 0.
    if len(ini["gauss_priors"]) > 0:
        lnp = np.sum(
            -0.5 * (p[ix_gauss_pri] - gauss_pri[:, 0])**2.
            / gauss_pri[:, 1]**2.)

    # Create parameters dictionnary for class and likelihoods
    class_input = ini["base_par_class"].copy()
    lkl_input = ini["base_par_lkl"].copy()

    # Loop over MCMC parameters
    for i, par in enumerate(ini["var_par"]):
        if par[0] == "var_class":
            class_input[par[1]] = p[i]
        else:
            lkl_input[par[1]] = p[i]

    # Deal with constraints
    for cst in ini["constraints"]:
        exec(cst)

    # Deal with parameter arrays
    final_class_input = class_input.copy()
    for n in ini["array_var"].keys():
        all_val = []
        for i in range(ini["array_var"][n]):
            all_val.append(str(final_class_input[f"{n}_val_{i}"]))
            final_class_input.pop(f"{n}_val_{i}")
        final_class_input[n] = ",".join(all_val)

    # Run class
    class_run = classy.Class()
    class_run.set(final_class_input)
    try:
        class_run.compute()
    except Exception as e:
        if ini["debug_mode"]:
            print(e)
        class_run.struct_cleanup()
        class_run.empty()
        return bad_res

    # Compute likelihoods
    lnls = [0.]*len(lkl)
    for i, like in enumerate(lkl):
        try:
            lnls[i] = float(like(class_input, lkl_input, class_run))
        except Exception as e:
            if ini["debug_mode"]:
                print(e)
            class_run.struct_cleanup()
            class_run.empty()
            return bad_res
        if not np.isfinite(lnls[i]):
            if ini["debug_mode"]:
                print(f"Likelihood '{ini['likelihoods'][i]}' is not finite.")
            class_run.struct_cleanup()
            class_run.empty()
            return bad_res

    # Computed derived parameters if requested
    derivs = []
    if len(ini["derivs"]) > 0:
        # Collect background and thermodynamics structures
        bg = class_run.get_background()
        th = class_run.get_thermodynamics()
        for deriv in ini["derivs"]:
            exec(f"derivs.append({deriv[1]})")
            # Computed prior on derived parameter if requested
            if deriv[0] in drv_gauss_pri:
                ix = drv_gauss_pri.index(deriv[0])
                pri = ini["drv_gauss_priors"][ix]
                lnp += -0.5 * (derivs[-1] - pri[1])**2. / pri[2]**2.
            if deriv[0] in drv_uni_pri:
                ix = drv_uni_pri.index(deriv[0])
                pri = ini["drv_uni_priors"][ix]
                test_uni_pri = pri[1] < derivs[-1] < pri[2]
                if not test_uni_pri:
                    class_run.struct_cleanup()
                    class_run.empty()
                    return bad_res

    # Clean up after CLASS
    class_run.struct_cleanup()
    class_run.empty()

    # Return log(lkl*prior)/T, log(prior), log(lkl), derivs
    res = [(sum(lnls) + lnp) / ini["temperature"], lnp] + lnls + derivs
    return tuple(res)


### Import additional modules for parallel computing if requested
pool = None
# Override: no parallelisation if in debug mode
if ini["debug_mode"]:
    pass
# Multithreading parallel computing via python-native multiprocessing module
elif ini["parallel"][0] == "multiprocessing":
    from multiprocessing import Pool
    n_threads = int(ini["parallel"][1]) # number of threads chosen by user
    pool = Pool(n_threads)
# MPI parallel computing via external schwimmbad module
elif ini["parallel"][0] == "MPI":
    from schwimmbad import MPIPool
    pool = MPIPool()
    if not pool.is_master(): # Necessary bit for MPI
        pool.wait()
        sys.exit(0)


### MCMC settings variables (for convenience of use)
n_dim = len(ini["var_par"])
n_steps = ini["n_steps"]
thin_by = ini["thin_by"]
n_walkers = ini["n_walkers"]


### Randomize initial walkers using "start" & "width" columns in ini file
p0_start = [par[2] for par in ini["var_par"]]
std_start = [par[5] for par in ini["var_par"]]
p_start = np.zeros((n_walkers, n_dim))
for i in range(n_dim):
    # Use truncnorm pdf just in case the prior "eats" too much of the Gaussian
    p_start[:, i] = truncnorm.rvs(
        (uni_pri[i, 0] - p0_start[i]) / std_start[i],
        (uni_pri[i, 1] - p0_start[i]) / std_start[i],
        loc=p0_start[i],
        scale=std_start[i],
        size=n_walkers,
    )


### Read input file if provided and modify initial positions accordingly
if ini["input_fname"] is not None:
    # Get parameters names and number of walkers from chain (from .ini file)
    in_ini = ECLAIR_parser.parse_ini_file(
        f"{ini['input_fname'][:-4]}.ini",
        silent_mode=True)
    in_names = [par[1] for par in in_ini["var_par"]]
    in_nw = in_ini["n_walkers"]
    # Get requested sample from chain
    input_p = np.loadtxt(ini['input_fname'])[:, 2:len(in_names)+2]
    input_p = input_p.reshape(-1, in_nw, len(in_names))[ini["ch_start"], :, :]
    # Find which current chain parameters are in provided input file
    ix_in_names = []
    for name in var_names:
        if name not in in_names:
            print(f"'{name}' parameter not found in input chain.")
            print("Will be randomized.")
            ix_in_names.append(-1)
        else:
            ix_in_names.append(in_names.index(name))
    # Fill up the intial walkers positions using p_start as a basis
    for nw in range(n_walkers):
        for i, ix in enumerate(ix_in_names):
            # In p_start, replace only the parameters present in input file
            if ix != -1:
                p_start[nw, i] = input_p[nw % in_nw, ix]


### Prepare some inputs for the MCMC
blobs_dtype = [("lnprior", float)]
blobs_dtype += [(f"lnlike_{name}", float) for name in ini["likelihoods"]]
blobs_dtype += [(deriv[0], float) for deriv in ini["derivs"]]
names = "  ".join(var_names)
blobs_names = "  ".join([b[0] for b in blobs_dtype])


### Initialize output file
if not ini["continue_chain"] and not ini["debug_mode"]:
    var_header = ["%s:%s" % (i + 2, n) for i, n in enumerate(var_names)]
    blobs_header = ["%s(d):%s" % (i + len(var_names) + 2, b[0])
                    for i, b in enumerate(blobs_dtype)]
    with open(f"{ini['output_root']}.txt", "w") as output_file:
        output_file.write("# 0:walker_id  1:lnprob  "
                          + "  ".join(var_header)
                          + "  "
                          + "  ".join(blobs_header)
                          + "\n")


### Do the actual MCMC
if (__name__ == "__main__") & (not ini["debug_mode"]):
    sampler = MCMCsampler.EnsembleSampler(
        n_walkers,
        n_dim,
        lnlike,
        pool=pool,
        blobs_dtype=blobs_dtype,
        **{k:v for k, v in ini['sampler_kwargs']},
    )
    ct = 0
    for result in sampler.sample(p_start, iterations=n_steps, thin_by=thin_by, progress=False):
        # Collect quantities depending on sampler
        if which_sampler == "emcee":
            result_coords = result.coords
            result_log_prob = result.log_prob
            result_blobs = result.blobs
        elif which_sampler == "zeus":
            result_coords = result[0]
            result_log_prob = result[1]
            result_blobs = result[2]
        # One-time check for infinities
        if ct == 0:
            n_finite = np.isfinite(result_log_prob).sum()
            if n_finite < 2:
                raise ValueError(
                    "Your chain cannot progress: "
                    "less than 2 of your walkers are starting at a finite "
                    "value of the posterior. Please check if your starting "
                    "positions are correct, and/or use debug mode to check "
                    "your likelihoods."
                )
            elif n_finite < (0.5 * n_walkers):
                print(
                    "Warning, your chain will take time to converge: "
                    f"only {n_finite * 100. / n_walkers}% of your walkers are "
                    "starting at a finite value of the posterior. Please check "
                    "if your starting positions are correct, and/or use debug "
                    "mode to check your likelihoods."
                )
        # Save current state in output file
        with open(f"{ini['output_root']}.txt", "a") as output_file:
            np.savetxt(
                output_file,
                np.hstack((
                    np.arange(n_walkers)[:, None],
                    result_log_prob[:, None],
                    result_coords,
                    result_blobs.view(dtype=np.float64).reshape(n_walkers, -1),
                ))
            )
        # Print MCMC progress
        ct += 1
        print(f"Current step : {ct} of {n_steps}")
    if ini["parallel"][0] == "MPI":
        pool.close()
        sys.exit()
