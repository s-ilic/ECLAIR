import sys
import ECLAIR_tools
import numpy as np
from scipy.stats import truncnorm


### Parse input ini file
ini_fname = sys.argv[1]
ini = ECLAIR_tools.parse_ini_file(ini_fname)


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
lkls = []
for like_name in ini["likelihoods"]:
    exec(f"import likelihoods.{like_name}")
    exec(f"lkls.append(likelihoods.{like_name}.get_loglike)")
    # exec(f"lkls.append(likelihoods.{like_name}.likelihood(ini['base_par_lkl']))")


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
bad_res = tuple([-np.inf] * (2 + len(lkls) + len(ini["derivs"])))


### Actual loglike function
def lnlike(p, counter):

    # Deal with uniform priors on MCMC parameters
    if not np.all((uni_pri[:, 0] <= p) & (p <= uni_pri[:, 1])):
        return bad_res

    # Deal with Gaussian priors on MCMC parameters
    lnp = 0.
    if len(ini["gauss_priors"]) > 0:
        lnp = np.sum(-0.5 * (p[ix_gauss_pri] - gauss_pri[:, 0])**2.
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
            all_val.append(str(final_class_input[f"{n}({i})"]))
            final_class_input.pop(f"{n}({i})")
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
    lnls = [0.]*len(lkls)
    for i, lkl in enumerate(lkls):
        try:
            lnls[i] = float(like(class_input, lkl_input, class_run))
            # lnls[i] = lkl.get_loglike(class_input, lkl_input, class_run)
        except Exception as e:
            if ini["debug_mode"]:
                print(f"The likelihood '{ini['likelihoods'][i]}' "
                      "raised the following error:")
                print(e)
            class_run.struct_cleanup()
            class_run.empty()
            return bad_res
        if not np.isfinite(lnls[i]):
            if ini["debug_mode"]:
                print(f"The likelihood '{ini['likelihoods'][i]}' is not finite.")
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

    # Grab current temperature (or inverse temperature)
    if ini["temperature_is_notinv"]:
        fact_T = 1. / ini["temperature"][counter.return_count()][0]
    else:
        fact_T = ini["temperature"][counter.return_count()][0]

    # Return log(lkl*prior)/T, log(prior), log(lkl), derivs
    res = [(sum(lnls) + lnp) * fact_T, lnp] + lnls + derivs
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
    in_ini = ECLAIR_tools.parse_ini_file(
        f"{ini['input_fname'][:-4]}.ini",
        silent_mode=True)
    in_names = (['lnprob']
                + [par[1] for par in in_ini["var_par"]]
                + ["lnprior"]
                + [f"lnlike_{name}" for name in in_ini["likelihoods"]]
                + [deriv[0] for deriv in in_ini["derivs"]])
    in_nw = in_ini["n_walkers"]
    # Read previous chain, reshape, trim, reverse and reshape again
    input_p = np.loadtxt(ini['input_fname'])
    in_nd = input_p.shape[1]
    input_p = input_p.reshape(-1, in_nw, in_nd)
    in_ns = input_p.shape[0]
    max_ix = ini["ch_start"] if ini["ch_start"] >= 0 else in_ns+ini["ch_start"]
    input_p = input_p[:max_ix+1, :, 1:][::-1, :, :].reshape(-1, in_nd-1)
    # Grab all the unique samples, in order
    keep_ix = np.sort(np.unique(input_p, axis=0, return_index=True)[1])
    input_p = input_p[keep_ix, :]
    # Trim that sample if requested
    for k in ini['keep_input']:
        if k[0] in in_names:
            par_ix = in_names.index(k[0])
            if k[1] == ">":
                g = input_p[:, par_ix] > float(k[2])
            else:
                g = input_p[:, par_ix] < float(k[2])
            input_p = input_p[g, :]
    if input_p.shape[0] < n_walkers:
        raise ValueError("Not enough unique samples in your input chain fulfil "
                         "your 'keep_input' requirements!")
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
                p_start[nw, i] = input_p[nw, ix]


### Prepare some inputs for the MCMC
blobs_dtype = ([("lnprior", float)]
               + [(f"lnlike_{name}", float) for name in ini["likelihoods"]]
               + [(deriv[0], float) for deriv in ini["derivs"]])


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


### Prepare a step counter for the MCMC (used when varying temperature)
step_counter = ECLAIR_tools.counter()


### Required steps if profiles are requested
# Adjust walker positions
for p in ini["profiles"]:
    ix = var_names.index(p[0])
    p_start[:, ix] = p[1]
# Prepare extra arguments for emcee special profiling move
if len(ini["profiles"]) > 0:
    extra_args = {}
    extra_args['indexes_to_keep'] = [var_names.index(p[0]) for p in ini["profiles"]]
    if which_sampler == "emcee":
        if ini["temperature_is_notinv"]:
            extra_args['temperature_array'] = [T[0] for T in ini["temperature"]]
        else:
            extra_args['temperature_array'] = [1./T[0] for T in ini["temperature"]]
        extra_args['counter'] = step_counter


### In debug mode, compute a single likelihood to print potential error messages
if ini["debug_mode"]:
    test_lnl = lnlike(p_start[0], step_counter)
    print(f"lnpost={test_lnl[0]}")
    print(f"lnprior={test_lnl[1]}")
    for i, n in enumerate(ini["likelihoods"]):
        print(f"lnlike({n})={test_lnl[2+i]}")


### Create copy of the ini file in output folder
if not ini["debug_mode"]:
    if ini["parallel"][0] == "MPI":
        if pool.is_master():
            ECLAIR_tools.copy_ini_file(ini_fname, ini)
    else:
        ECLAIR_tools.copy_ini_file(ini_fname, ini)


### Do the actual MCMC
if (__name__ == "__main__") & (not ini["debug_mode"]):

    print(f"### Starting MCMC in {ini['output_root']} ###")

    # Prepare sampler extra keywords
    sampler_kwargs = {}
    for k, v in ini['sampler_kwargs'].items():
        exec(f"sampler_kwargs['{k}'] = {v}")

    # Create sampler and run
    sampler = MCMCsampler.EnsembleSampler(
        n_walkers,
        n_dim,
        lnlike,
        args=(step_counter,),
        pool=pool,
        blobs_dtype=blobs_dtype,
        **sampler_kwargs,
    )
    start, log_prob0, blobs0, ct = p_start, None, None, 0
    for ix, T_list in enumerate(ini['temperature']):
        for result in sampler.sample(start, log_prob0=log_prob0, blobs0=blobs0,
                                     iterations=T_list[1],
                                     thin_by=thin_by, progress=False):
            # Collect quantities
            if which_sampler == "emcee":
                coords = result.coords
                log_prob = result.log_prob
                blobs = result.blobs
            elif which_sampler == "zeus":
                coords = result[0]
                log_prob = result[1]
                blobs = result[2]
            # One-time check for infinities
            if ct == 0:
                n_finite = np.isfinite(log_prob).sum()
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
                        f"only {n_finite * 100. / n_walkers}% of your walkers "
                        "are starting at a finite value of the posterior. "
                        "Please check if your starting positions are correct, "
                        "and/or use debug_mode to check your likelihoods."
                    )
            # Save current state in output file
            if ct % thin_by == 0:
                with open(f"{ini['output_root']}.txt", "a") as output_file:
                    np.savetxt(
                        output_file,
                        np.hstack((
                            np.arange(n_walkers)[:, None],
                            log_prob[:, None],
                            coords,
                            blobs.view(dtype=np.float64).reshape(n_walkers, -1),
                        ))
                    )
            # Print then increase MCMC counter
            print(f"Current step: {ct+1} of {n_steps}")
            ct += 1
        # Rescale loglikes to account for temperature change
        if ix < (len(ini['temperature'])-1):
            old_T = ini["temperature"][ix][0]
            new_T = ini["temperature"][ix+1][0]
            start = coords.copy()
            log_prob0 = log_prob.copy() * old_T / new_T
            blobs0 = blobs.copy()
        # Increase temperature counter
        step_counter.increase_count()
    # Some MPI-related cleanup
    if ini["parallel"][0] == "MPI":
        pool.close()
        sys.exit()
