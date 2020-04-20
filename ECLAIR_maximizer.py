import os
import sys
import emcee
import XYZ_parser
import numpy as np

if __name__ == "__main__":

    ### Parse the input ini file
    ini_fname = sys.argv[1]
    ini = XYZ_parser.parse_ini_file(ini_fname)


    ### If new chain is started, create a copy of ini file in output folder
    if not ini['continue_chain']:
        XYZ_parser.copy_ini_file(ini_fname, ini)


    ### Print the output path root
    print('Starting MCMC in %s' % ini['output_root'])


    ### Import requested variant of class python wrapper
    which_class = ini['which_class']
    exec('import %s as classy' % which_class)


    ### Conveninent MCMC settings variables
    n_steps = ini['n_steps']
    thin_by = ini['thin_by']
    n_walkers = ini['n_walkers']
    if ini['n_walkers_type'] == 'prop_to':
        n_walkers *= len(ini['var_par'])


    ### Import requested likelihoods
    # Grab likelihoods folder path (folder should be in same directory as this script)
    path_to_likes = os.path.dirname(os.path.realpath(sys.argv[0])) + '/XYZ_likelihoods'
    # Insert in python path
    sys.path.insert(0, path_to_likes)
    # Import and store in list all loglikelihoods
    likes = []
    for like_name in ini['likelihoods']:
        exec('import %s' % like_name)
        exec('likes.append(%s.get_loglike)' % like_name)


    ### Misc useful quantities
    # List of all variable names
    var_names = [p[1] for p in ini['var_par']]
    # List of uniform priors ([lower, upper] bounds)
    uni_pri = np.array([[p[3], p[4]] for p in ini['var_par']])
    # List of Gaussian priors ([mean, stddev])
    gauss_pri = np.array([[p[1], p[2]] for p in ini['gauss_priors']])
    # List of indices of variables with Gaussian priors
    ix_gauss_pri = np.array([var_names.index(p[0]) for p in ini['gauss_priors']])
    # List of names of derived parameters with Gaussian priors
    drv_gauss_pri = [n[0] for n in ini['drv_gauss_priors']]
    # "Bad result" to be returned by lnlike if evaluation fails in any way
    bad_res = tuple([-np.inf] * (4 + len(likes) + len(ini['derivs'])))


    ### Actual loglike function
    def lnlike(p, temp):

        # Deal with uniform priors
        if not np.all((uni_pri[:, 0] <= p) & (p <= uni_pri[:, 1])):
            return bad_res

        # Deal with Gaussian priors
        lnp = 0.
        if len(ini['gauss_priors']) > 0:
            lnp = np.sum(
                -0.5 * (p[ix_gauss_pri] - gauss_pri[:, 0])**2.
                / gauss_pri[:, 1]**2.)

        # Create parameters dictionnary for class and likelihoods
        class_input = ini['base_par_class'].copy()
        likes_input = ini['base_par_likes'].copy()

        # Loop over parameters
        for i, par in enumerate(ini['var_par']):
            if par[0] == 'var_class':
                class_input[par[1]] = p[i]
            else:
                likes_input[par[1]] = p[i]

        # Deal with constraints
        for cst in ini['constraints']:
            exec(cst)

        # Deal with parameter arrays
        final_class_input = class_input.copy()
        for n in ini['array_var'].keys():
            all_val = []
            for i in range(ini['array_var'][n]):
                all_val.append(str(final_class_input['%s_val_%s' % (n, i)]))
                final_class_input.pop('%s_val_%s' % (n, i))
            final_class_input[n] = ','.join(all_val)

        # Run class
        class_run = classy.Class()
        class_run.set(final_class_input)
        try:
            class_run.compute()
        except:
            class_run.struct_cleanup()
            class_run.empty()
            return bad_res

        # Compute likelihoods
        lnls = [0.]*len(likes)
        for i, like in enumerate(likes):
            try:
                lnls[i] = float(like(class_input, likes_input, class_run))
            except:
                class_run.struct_cleanup()
                class_run.empty()
                return bad_res
            if not np.isfinite(lnls[i]):
                class_run.struct_cleanup()
                class_run.empty()
                return bad_res

        # Computed derived parameters if requested
        derivs = []
        if len(ini['derivs']) > 0:
            bg = class_run.get_background() # in case it's needed
            for deriv in ini['derivs']:
                exec('derivs.append(%s)' % deriv[1])
                # Computed prior on derived parameter if requested
                if deriv[0] in drv_gauss_pri:
                    ix = drv_gauss_pri.index(deriv[0])
                    pri = ini['drv_gauss_priors'][ix]
                    lnp += -0.5 * (derivs[-1] - pri[1])**2. / pri[2]**2.

        # Clean up after CLASS
        class_run.struct_cleanup()
        class_run.empty()

        # Return log(likes*prior)/T, log(prior), log(likes*prior), T, log(likes), derivs
        res = [(sum(lnls) + lnp) / temp, lnp, sum(lnls) + lnp, temp] + lnls + derivs
        return tuple(res)


    ### Import additional modules for parallel computing if requested
    pool = None
    # Multithreading parallel computing via python-native multiprocessing module
    if ini['parallel'][0] == 'multiprocessing':
        from multiprocessing import Pool
        pool = Pool(int(ini['parallel'][1])) # number of threads chosen by user
    # MPI parallel computing via external schwimmbad module
    elif ini['parallel'][0] == 'MPI':
        from schwimmbad import MPIPool
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)


    ### Randomize initial walkers positions according to "var" inputs in ini file
    n_dim = len(ini['var_par'])
    p0_start = [par[2] for par in ini['var_par']]
    std_start = [par[5] for par in ini['var_par']]
    p_start = emcee.utils.sample_ball(p0_start, std_start, n_walkers)


    ### Read input file if provided and generate initial positions accordingly
    if ini['input_type'] == 'chain':
        with emcee.backends.HDFBackend(ini['input_fname'] + '.h5', read_only=True) as reader:
            # Get requested sample from chain
            input_p = reader.get_chain()[ini['ch_start'], :, :].copy()
            in_nw = input_p.shape[0]
            # Get parameters names from chain (from .ini file)
            in_ini = XYZ_parser.parse_ini_file(
                ini['input_fname'] + '.ini',
                ignore_errors=True)
            in_names = [par[1] for par in in_ini['var_par']]
    elif (ini['input_type'] == 'one_walker') | (ini['input_type'] == 'many_walkers'):
        # Get input walkers positions from file and their number
        input_p = np.loadtxt(ini['input_fname'])
        in_nw = input_p.shape[0]
        # Get parameters name from first line of input file
        with open(ini['input_fname']) as f:
            in_names = f.readline()[1:].split()
    if ini['input_type'] is not None:
        # Look for current chain parameters in input file
        ix_in_names = []
        for name in var_names:
            if name not in in_names:
                print('"%s" parameter not found in input chain.' % name)
                print('Will be randomized.')
                ix_in_names.append(-1)
            else:
                ix_in_names.append(in_names.index(name))
        # Fill up the intial walkers positions using p_start as a basis
        for n in range(n_walkers):
            for i, ix in enumerate(ix_in_names):
                # Replace in p_start only the parameters present in input file
                if ix != -1:
                    if ini['input_type'] == 'one_walker':
                        p_start[n, i] = (np.random.randn() * std_start[i]
                                        + input_p[n % in_nw, ix])
                    else:
                        p_start[n, i] = input_p[n % in_nw, ix]


    ### Prepare some inputs for the MCMC
    blobs_dtype = [("lnprior", float)]
    blobs_dtype += [("real_lnprob", float)]
    blobs_dtype += [("temperature", float)]
    blobs_dtype += [("lnlike_%s" % name, float) for name in ini['likelihoods']]
    for deriv in ini['derivs']:
        blobs_dtype += [("%s" % deriv[0], float)]
    names = '  '.join([par[1] for par in ini['var_par']])

    ### Parameters controlling the minimization
    n_T = 1000
    n_step_each_T = 20# * 2
    fact_T = np.sqrt(1.01)

    ### Do the minimization via MCMC
    current_temp = ini['temperature']
    backend = emcee.backends.HDFBackend(ini['output_root'] + '.h5')
    if not ini['continue_chain']:
        backend.reset(n_walkers, n_dim)
    pos = p_start
    for i in range(n_T):
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            lnlike,
            args=(current_temp,),
            moves=emcee.moves.StretchMove(a=ini['stretch']),
            pool=pool,
            backend=backend,
            blobs_dtype=blobs_dtype,
        )
        ct = 0
        print('Current temperature : %s' % current_temp)
        for result in sampler.sample(pos, iterations=n_steps, thin_by=thin_by):
            # Always save the last MCMC step as input file for future chain
            ct += 1
            print('> Current step : %s of %s' % (ct, n_steps))
            np.savetxt(ini['output_root'] + '.input',
                    np.hstack((result.coords, result.log_prob[:, None])),
                    header=names + '  log_prob')
            pos, lnl, blob, dum = result
            if ct >= n_step_each_T:
                current_temp /= fact_T
                break
    pool.close()
    sys.exit()
