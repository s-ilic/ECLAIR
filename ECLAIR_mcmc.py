import os
import sys
import ECLAIR_parser
import numpy as np
from scipy.stats import truncnorm


### Parse input ini file
ini_fname = sys.argv[1]
ini = ECLAIR_parser.parse_ini_file(ini_fname)


### If a new chain is started, create copy of the ini file in output folder
if not ini["continue_chain"]:
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
likes = []
for like_name in ini["likelihoods"]:
    exec(f"import likelihoods.{like_name}")
    exec(f"likes.append(likelihoods.{like_name}.get_loglike)")


### Misc useful quantities
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
# "Bad result" to be returned by lnlike if evaluation fails in any way
bad_res = tuple([-np.inf] * (2 + len(likes) + len(ini["derivs"])))


### Actual loglike function
def lnlike(p):

    # Deal with uniform priors
    if not np.all((uni_pri[:, 0] <= p) & (p <= uni_pri[:, 1])):
        return bad_res

    # Deal with Gaussian priors
    lnp = 0.
    if len(ini["gauss_priors"]) > 0:
        lnp = np.sum(
            -0.5 * (p[ix_gauss_pri] - gauss_pri[:, 0])**2.
            / gauss_pri[:, 1]**2.)

    # Create parameters dictionnary for class and likelihoods
    class_input = ini["base_par_class"].copy()
    likes_input = ini["base_par_likes"].copy()

    # Loop over parameters
    for i, par in enumerate(ini["var_par"]):
        if par[0] == "var_class":
            class_input[par[1]] = p[i]
        else:
            likes_input[par[1]] = p[i]

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
    lnls = [0.]*len(likes)
    for i, like in enumerate(likes):
        try:
            lnls[i] = float(like(class_input, likes_input, class_run))
        except Exception as e:
            if ini["debug_mode"]:
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
        bg = class_run.get_background() # in case it is needed
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

    # Return log(likes*prior)/T, log(prior), log(likes), derivs
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
    pool = Pool(int(ini["parallel"][1])) # number of threads chosen by user
# MPI parallel computing via external schwimmbad module
elif ini["parallel"][0] == "MPI":
    from schwimmbad import MPIPool
    pool = MPIPool()
    if not pool.is_master(): # Necessary bit for MPI
        pool.wait()
        sys.exit(0)


### Conveninent MCMC settings variables
n_dim = len(ini["var_par"])
n_steps = ini["n_steps"]
thin_by = ini["thin_by"]
n_walkers = ini["n_walkers"]


### Randomize initial walkers positions according to "start" & "width" columns in ini file
p0_start = [par[2] for par in ini["var_par"]]
std_start = [par[5] for par in ini["var_par"]]
p_start = np.zeros((n_walkers, n_dim))
for i in range(n_dim):
    p_start[:, i] = truncnorm.rvs(
        (uni_pri[i, 0] - p0_start[i]) / std_start[i],
        (uni_pri[i, 1] - p0_start[i]) / std_start[i],
        loc=p0_start[i],
        scale=std_start[i],
        size=n_walkers,
    )


### Read input file if provided and modify initial positions accordingly
if ini["input_type"] == "HDF5_chain":
    with MCMCsampler.backends.HDFBackend(f"{ini['input_fname']}.h5", read_only=True) as reader:
        # Get requested sample from chain
        input_p = reader.get_chain()[ini["ch_start"], :, :].copy()
        in_nw = input_p.shape[0]
        # Get parameters names from chain (from .ini file)
        in_ini = ECLAIR_parser.parse_ini_file(
            f"{ini['input_fname']}.ini",
            silent_mode=True)
        in_names = [par[1] for par in in_ini["var_par"]]
elif ini["input_type"] == "text_chain":
    # Get parameters names and number of walkers from chain (from .ini file)
    in_ini = ECLAIR_parser.parse_ini_file(
        f"{ini['input_fname']}.ini",
        silent_mode=True)
    in_names = [par[1] for par in in_ini["var_par"]]
    in_nw = in_ini["n_walkers"]
    # Get requested sample from chain
    input_p = np.loadtxt(f"{ini['input_fname']}.txt")[:, 2:len(in_names)+2]
    input_p = input_p.reshape(-1, in_nw, len(in_names))[ini["ch_start"], :, :]
elif ini["input_type"] == "walkers":
    # Get input walkers positions from file and their number
    input_p = np.loadtxt(ini["input_fname"])
    in_nw = input_p.shape[0]
    # Get parameters name from first line of input file
    with open(ini["input_fname"]) as f:
        in_names = f.readline()[1:].split()
if ini["input_type"] is not None:
    # Look for current chain parameters in input file
    ix_in_names = []
    for name in var_names:
        if name not in in_names:
            print(f"'{name}' parameter not found in input chain.")
            print("Will be randomized.")
            ix_in_names.append(-1)
        else:
            ix_in_names.append(in_names.index(name))
    # Fill up the intial walkers positions using p_start as a basis
    for n in range(n_walkers):
        for i, ix in enumerate(ix_in_names):
            # Replace in p_start only the parameters present in input file
            if ix != -1:
                    p_start[n, i] = input_p[n % in_nw, ix]


### Prepare some inputs for the MCMC
blobs_dtype = [("lnprior", float)]
blobs_dtype += [(f"lnlike_{name}", float) for name in ini["likelihoods"]]
blobs_dtype += [(deriv[0], float) for deriv in ini["derivs"]]
names = "  ".join(var_names)
blobs_names = "  ".join([b[0] for b in blobs_dtype])


### Initialize output file
if ini["output_format"] == "text":
    backend = None
    if not ini["continue_chain"]:
        with open(f"{ini['output_root']}.txt", "w") as output_file:
            output_file.write(
                "# 0:walker_id  1:lnprob  " +
                "  ".join(["%s:%s" % (i + 2, n) for i, n in enumerate(var_names)]) +
                "  " +
                "  ".join(["%s(d):%s" % (i + len(var_names) + 2, b[0]) for i, b in enumerate(blobs_dtype)]) +
                "\n"
            )
elif ini["output_format"] == "HDF5":
    backend = MCMCsampler.backends.HDFBackend(f"{ini['output_root']}.h5")
    if not ini["continue_chain"]:
        backend.reset(n_walkers, n_dim)
    open(f"{ini['output_root']}.lock", "w").close() # creates empty lock file


### Additional arguments for sampler
sampler_args = {}
if which_sampler == "emcee":
    sampler_args["moves"] = MCMCsampler.moves.StretchMove(a=ini["stretch"])
    sampler_args["backend"] = backend


### Do the actual MCMC
if (__name__ == "__main__") & (not ini["debug_mode"]):
    sampler = MCMCsampler.EnsembleSampler(
        n_walkers,
        n_dim,
        lnlike,
        pool=pool,
        blobs_dtype=blobs_dtype,
        **sampler_args,
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
                    "less than 2 of your walkers are starting at a finite value of "
                    "the posterior. Please check if your starting positions are "
                    "correct, and/or use debug mode to check your likelihoods."
                )
            elif n_finite < (0.5 * n_walkers):
                print(
                    "Warning, your chain will take time to converge: "
                    f"only {n_finite * 100. / n_walkers}% of your walkers are "
                    "starting at a finite value of the posterior. Please check if "
                    "your starting positions are correct, and/or use debug mode to "
                    "check your likelihoods."
                )
        # Always save the last MCMC step as plain text input file for future chain
        np.savetxt(
            f"{ini['output_root']}.input",
            np.hstack((
                result_coords,
                result_log_prob[:, None],
                result_blobs.view(dtype=np.float64).reshape(n_walkers, -1),
            )),
            header=names + "  log_prob  " + blobs_names,
        )
        # If not using the HDF5 format, save current state in plain text
        if ini["output_format"] == "text":
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
    if ini["output_format"] == "HDF5":
        os.remove(f"{ini['output_root']}.lock") # remove lock file
    if ini["parallel"][0] == "MPI":
        pool.close()
        sys.exit()
