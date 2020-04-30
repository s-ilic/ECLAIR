import os
import sys
import numpy as np
from ECLAIR_parser import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser,RawTextHelpFormatter


####################################################################################
####################################################################################

# Parsing input arguments

parser = ArgumentParser(
    description='Analyser for outputs from the ECLAIR suite.',
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    'file',
    action='store',
    nargs=1,
    type=str,
    help='Path to chain file to be analyzed.',
)
parser.add_argument(
    '-c',
    '--copy',
    metavar='PATH',
    action='store',
    help='Make a temporary copy of the chain in PATH before analyzing.\n'
         '> Default PATH: current directory.',
)
parser.add_argument(
    '-k',
    '--keep',
    metavar='LIST',
    action='store',
    nargs=1,
    help='Keep only some parameters for analysis; argument should\n'
         'be a list of comma-separated parameter indexes.\n'
         '> Example: "1,5,7-8"  (hyphens indicate a range)\n'
         '> Default: keep all parameters.',
)
parser.add_argument(
    '-p',
    '--plot',
    metavar='LIST',
    action='store',
    nargs=1,
    help='list of plots to produce; argument should be a list of\n'
         'comma-separated parameter indexes.\n'
         'List of possible plots:\n'
         '1 : -log(likelihood) values for all walkers along chain\n'
         '2 : parameters values for all walkers along chain\n'
         '3 : derived parameters for all walkers along chain\n'
         '4 : mean acceptance rate for each walker as chain progresses\n'
         '5 : acceptance rate averaged over walkers at each chain step\n'
         '> Example: "1,2,4-6"  (hyphens indicate a range)\n'
         '> Default: 1 (likelihood plot only)',
)
parser.add_argument(
    '-b',
    '--burn-in',
    metavar='N',
    action='store',
    nargs=1,
    help='If N >= 1, int(N) is the number of burned-in steps.\n'
         'If 0 <= N < 1, float(N) is the fraction of burned-in steps\n'
         '> Default: 0 (no burn-in)'
)
parser.add_argument(
    '-t',
    '--thin',
    metavar='N',
    action='store',
    nargs=1,
    help='Thin the chains so as to keep only every int(N)th step.\n'
         '> Default: 1 (no thinning)'
)
parser.add_argument(
    '-w',
    '--thin-walk',
    metavar='N',
    action='store',
    nargs=1,
    help='Thin the walkers so as to keep only every int(N)th walker.\n'
         '> Default: 1 (no walker thinning)'
)
args = parser.parse_args()


####################################################################################
####################################################################################

# Checking input arguments

if not os.path.isfile(args.file[0]): # Input chain path
    raise ValueError("Chain %s does not exist." % args.file[0])
else:
    test = (
        args.file[0].endswith('.h5')
        & os.path.isfile(args.file[0][:-3] + '.lock')
        & (args.copy is None)
    )
    if test:
        raise ValueError("Chain %s is HDF5 and running; use the -c option to prevent I/O errors that could compromise your MCMC run." % args.file[0])
    if args.file[0].endswith('.h5'):
        ftype = 'HDF5'
    elif args.file[0].endswith('.txt'):
        ftype = 'text'
    else:
        raise ValueError("Unrecognized file format for input chain %s, should be either .txt or .h5" % args.file[0])

if args.copy is not None: # Temp copy path
    if not os.path.isdir(args.copy):
        print("Copy path %s does not exist or is not a directory, defaulting to current directory." % args.copy)
        cp_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    else:
        cp_path = os.path.realpath(args.copy)


keep = [] # Parameter list
if args.keep is not None:
    tmp = args.keep[0].split(',')
    for t in tmp:
        if is_number(t):
            keep.append(int(t))
        else:
            st = t.split('-')
            test_1 = t.count('-') != 1
            test_2 = any([not is_number(tt) for tt in st])
            if test_1 | test_2:
                raise ValueError("Bad formatting for range %s" % t)
            test_3 = int(st[0]) >= int(st[1])
            if test_3:
                raise ValueError("Bad formatting for range %s" % t)
            keep += range(int(st[0]), int(st[1]) + 1)
keep = np.unique(keep)

plot = [] # Plot lists
if args.plot is not None:
    tmp = args.plot[0].split(',')
    tmp_plot = []
    for t in tmp:
        if is_number(t):
            tmp_plot.append(int(t))
        else:
            st = t.split('-')
            test_1 = t.count('-') != 1
            test_2 = any([not is_number(tt) for tt in st])
            if test_1 | test_2:
                raise ValueError("Bad formatting for range %s" % t)
            test_3 = int(st[0]) >= int(st[1])
            if test_3:
                raise ValueError("Bad formatting for range %s" % t)
            tmp_plot += range(int(st[0]), int(st[1]) + 1)
    for p in tmp_plot:
        if p not in [1,2,3,4,5]:
            print("Unrecognised plot option %s, ignored.")
        else:
            plot.append(p)
else:
    plot = [1]

if args.burn_in is None: # Burn-in
    burn_is_float = False
    burn = 0
else:
    if not is_number(args.burn_in[0]):
        raise ValueError("Input burn-in is not a number (%s)" % args.burn_in[0])
    elif float(args.burn_in[0]) < 0:
        raise ValueError("Input burn-in is negative (%s)" % args.burn_in[0])
    elif float(args.burn_in[0]) < 1:
        burn_is_float = True
        burn = float(args.burn_in[0])
    else:
        burn_is_float = False
        burn = int(args.burn_in[0])

if args.thin is None: # Sample thinning
    thin = 1
else:
    if not is_number(args.thin[0]):
        raise ValueError("Input sample thinning is not a number (%s)" % args.thin[0])
    elif float(args.thin[0]) < 1:
        raise ValueError("Input sample thinning is < 1 (%s)" % args.thin[0])
    else:
        thin = int(args.thin[0])

if args.thin_walk is None: # Walker thinning
    thinw = 1
else:
    if not is_number(args.thin_walk[0]):
        raise ValueError("Input walker thinning is not a number (%s)" % args.thin_walk[0])
    elif float(args.thin_walk[0]) < 1:
        raise ValueError("Input walker thinning is < 1 (%s)" % args.thin_walk[0])
    else:
        thinw = int(args.thin_walk[0])


####################################################################################
####################################################################################

# Read and parse chain + copy beforehand if requested

input_fname = os.path.realpath(args.file[0]) # Input chain filename

if input_fname.endswith('.h5'): # Check if HDF5 file
    import emcee

if args.copy is not None: # Do the copy if requested
    ran = str(np.random.rand())[2:]
    fname = "%s/%s" % (cp_path, ran)
    print("Copying %s into %s" % (input_fname, fname))
    os.system('cp -f %s %s' % (input_fname, fname))
else:
    fname = input_fname

# Read chain content
if ftype == "HDF5":
    ini_fname = input_fname[:-3] + '.ini'
    ini = parse_ini_file(ini_fname, silent_mode=True)
    par_names = np.array([par[1] for par in ini['var_par']])
    reader = emcee.backends.HDFBackend(fname, read_only=True)
    ln = reader.get_log_prob()
    ch = reader.get_chain()
    bl = reader.get_blobs()
    blobs_names = list(bl.dtype.names)
    n_blobs = len(blobs_names)
    n_steps, n_walkers, n_par = ch.shape
    bl = bl.view(dtype=np.float64).reshape(n_steps, n_walkers, -1)
else:
    ini_fname = input_fname[:-4] + '.ini'
    ini = parse_ini_file(ini_fname, silent_mode=True)
    par_names = np.array([par[1] for par in ini['var_par']])
    n_par = len(ini['var_par'])
    n_walkers = ini['n_walkers']
    with open(fname, 'r') as f:
        header = f.readline()
    all_names = [s.split(':')[1] for s in header[1:].split()]
    n_blobs = len(all_names) - n_par - 2
    blobs_names = all_names[-n_blobs:]
    tmp = np.loadtxt(fname)#.reshape(-1, n_walkers, len(all_names))
    ln = tmp[:, 1].reshape(-1, n_walkers)
    ch = tmp[:, 2:2+n_par].reshape(-1, n_walkers, n_par)
    bl = tmp[:, 2+n_par:].reshape(-1, n_walkers, n_blobs)
    n_steps = ln.shape[0]

# Read priors
lbs = np.array([par[3] for par in ini['var_par']])
ubs = np.array([par[4] for par in ini['var_par']])
if 'gauss_priors' in ini.keys():
    pri_dict = {}
    for p in ini['gauss_priors']:
        pri_dict[p[0]] = [p[1], p[2]]
if 'drv_gauss_priors' in ini.keys():
    drv_pri_dict = {}
    for p in ini['drv_gauss_priors']:
        drv_pri_dict[p[0]] = [p[1], p[2]]

# Burn chain
if burn_is_float:
    burn = int(n_steps * burn)
ln = ln[burn:, :]
ch = ch[burn:, :, :]
bl = bl[burn:, :, :]
print('Kept %s steps (out of %s) after burning' % (ln.shape[0], n_steps))

# Sample thinning
ln = ln[::thin, :]
ch = ch[::thin, :, :]
bl = bl[::thin, :, :]
print('Kept %s steps (out of %s) after thinning' % (ln.shape[0], n_steps))

# Walker thinning
ln = ln[:, ::thinw]
ch = ch[:, ::thinw, :]
bl = bl[:, ::thinw, :]
print('Kept %s walkers (out of %s) after walker thinning' % (ln.shape[1], n_walkers))
print('>>> Total samples used : %s' % (ln.shape[0] * ln.shape[1]))

# Parameter selection
if args.keep is None:
    keep = np.arange(n_par)
g = (keep >= 0) & (keep < n_par)
if sum(g) == 0:
    raise ValueError("No valid parameter index passed: %s" % keep)
else:
    ch = ch[:, :, keep[g]]
    par_names = par_names[keep[g]]
    lbs = lbs[keep[g]]
    ubs = ubs[keep[g]]
print('>>> Total number of parameters : %s' % ch.shape[2])
n_steps, n_walkers, n_par = ch.shape

# Check for temperature
if 'temperature' in blobs_names:
    temp = bl[:, :, blobs_names.index('temperature')]
else:
    temp = ln*0. + 1.


####################################################################################
####################################################################################

if 1 in plot:
    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(-1. * ln, alpha=0.1)
    for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
        ax1.axhline(
            np.percentile(-1. * ln, p, axis=1)[n_steps//2:].mean(),
            color='blue',
            lw=2,
            ls=ls,
        )
    for p in [2.5, 16, 50, 84, 97.5]:
        ax1.plot(
            np.percentile(-1. * ln, p, axis=1),
            color='black',
            lw=2,
            ls='--',
        )
    ax1.set_xlabel("MCMC step")
    ax1.set_ylabel("-log(likelihood)")

if 2 in plot:
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_par:
            break
        nc += 1
        if (nr*nc) >= n_par:
            break
        nr += 1
    fig2, ax2 = plt.subplots(nr, nc, squeeze=False, sharex=True)
    ix = 0
    for r in range(nr):
        for c in range(nc):
            ax2[r, c].plot(ch[:, :, ix], alpha=0.1)
            for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
                ax2[r, c].axhline(
                    np.percentile(ch[:, :, ix], p, axis=1)[n_steps//2:].mean(),
                    color='blue',
                    lw=2,
                    ls=ls,
                )
            for p in [2.5, 16, 50, 84, 97.5]:
                ax2[r, c].plot(
                    np.percentile(ch[:, :, ix], p, axis=1),
                    color='black',
                    lw=2,
                    ls='--',
                )
            ax2[r, c].set_ylabel(par_names[ix])
            curr_ylim = ax2[r, c].get_ylim()
            if (np.isfinite(lbs[ix])) & (lbs[ix] >= curr_ylim[0]):
                ax2[r, c].axhline(lbs[ix], ls='--', color='red')
            if (np.isfinite(ubs[ix])) & (ubs[ix] <= curr_ylim[1]):
                ax2[r, c].axhline(ubs[ix], ls='--', color='red')
            if 'gauss_priors' in ini.keys():
                if par_names[ix] in pri_dict.keys():
                    pri = pri_dict[par_names[ix]]
                    for m, ls in zip([-1.,0.,1.],['--','-','--']):
                        val = pri[0] + m * pri[1]
                        if (curr_ylim[0] <= val <= curr_ylim[1]):
                            ax2[r, c].axhline(val, ls=ls, color='orange')
            ix += 1
    fig2.subplots_adjust(
        left = 0.05,
        bottom = 0.03,
        right = 0.99,
        top = 0.98,
        wspace = 0.34,
        hspace = 0.,
    )

if 3 in plot:
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_blobs:
            break
        nc += 1
        if (nr*nc) >= n_blobs:
            break
        nr += 1
    fig3, ax3 = plt.subplots(nr, nc, squeeze=False, sharex=True)
    ix = 0
    for r in range(nr):
        for c in range(nc):
            ax3[r, c].plot(bl[:, :, ix], alpha=0.1)
            for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
                ax3[r, c].axhline(
                    np.percentile(bl[:, :, ix], p, axis=1)[n_steps//2:].mean(),
                    color='blue',
                    lw=2,
                    ls=ls,
                )
            for p in [2.5, 16, 50, 84, 97.5]:
                ax3[r, c].plot(
                    np.percentile(bl[:, :, ix], p, axis=1),
                    color='black',
                    lw=2,
                    ls='--',
                )
            ax3[r, c].set_ylabel(blobs_names[ix])
            curr_ylim = ax3[r, c].get_ylim()
            if 'drv_gauss_priors' in ini.keys():
                if blobs_names[ix] in drv_pri_dict.keys():
                    pri = drv_pri_dict[blobs_names[ix]]
                    for m, ls in zip([-1.,0.,1.],['--','-','--']):
                        val = pri[0] + m * pri[1]
                        if (curr_ylim[0] <= val <= curr_ylim[1]):
                            ax3[r, c].axhline(val, ls=ls, color='orange')
            ix += 1
    fig2.subplots_adjust(
        left = 0.05,
        bottom = 0.03,
        right = 0.99,
        top = 0.98,
        wspace = 0.34,
        hspace = 0.,
    )

if 4 in plot:
    fig4, ax4 = plt.subplots(1, 1)
    test = (ch[1:, :, 0] - ch[:-1, :, 0]) != 0.
    test = (
        np.cumsum(test, axis=0)
        / np.arange(1, test.shape[0]+1)[:, None].astype('float')
    )
    ax4.plot(test, alpha=0.1)
    for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
        plt.plot(
            np.percentile(test, p, axis=1),
            color='black',
            lw=2,
            ls=ls,
        )
    ax4.set_xlabel("MCMC step")
    ax4.set_ylabel("Mean acceptance rate")

if 5 in plot:
    fig5, ax5 = plt.subplots(1, 1)
    test = (ch[1:, :, 0] - ch[:-1, :, 0]) != 0.
    y = test.mean(axis=1)
    x = np.arange(len(y))
    ax5.plot(test.mean(axis=1), color='blue')
    ax5.plot(x, np.poly1d(np.polyfit(x,y,3))(x), color='black', lw=2)
    ax5.set_xlabel("MCMC step")
    ax5.set_ylabel("Mean acceptance rate")

if args.copy is not None:
    os.system('rm %s' % fname)

plt.show()
