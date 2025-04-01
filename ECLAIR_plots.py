import os
import numpy as np
from ECLAIR_tools import *
from itertools import product
import matplotlib.pyplot as plt
from argparse import ArgumentParser,RawTextHelpFormatter
from scipy.fft import fft, ifft


###################################
### Settings for tweaking plots ###
###################################
s = {}

# Generic settings
s["pix_x_size"] = 1920      # Optimize subplot grid for this screen pixel width
s["pix_y_size"] = 1080      # Optimize subplot grid for this screen pixel height

# Plot 1 (likelihood plot)
s["alpha_1"] = 0.1      # Opacity of plotted curves for walkers
s["n_bins_1"] = 32      # Number of bins along step axis for percentile overplot
s["pct_1"] = [2.5, 16, 50, 84, 97.5]    # Which percentiles to overplot
s["pct_ls_1"] = [':','--','-','--',':'] # Linestyle for each percentile overplot
s["pct_lw_1"] = [1]*5                   # Line width for percentile overplot
s["pct_col_1"] = ["black"]*5            # Color for percentile overplot

# Plot 2 (MCMC parameters plot)
s["alpha_2"] = 0.1      # Opacity of plotted curves for walkers
s["n_bins_2"] = 32      # Number of bins along step axis for cred. int. overplot
s["CI_2"] = [68., 95.]           # Which credible interval to overplot
s["CI_2_ls"] = ['--', ':']       # Linestyle for each credible interval overplot
s["CI_2_lw"] = [1, 1]            # Linestyle for each credible interval overplot
s["CI_2_c1"] = "black"           # Colour of binned credible intervals
s["CI_2_c2"] = "cyan"            # Colour of left-half credible intervals
s["CI_2_c3"] = "magenta"         # Colour of right-half credible intervals
s["upri_2_h"] = "/"              # Hatch pattern of uniform priors
s["upri_2_c"] = "red"            # Colour of uniform priors
s["upri_2_a"] = 0.5              # Alpha of uniform priors
s["gpri_2_h"] = "\\"             # Hatch pattern of gaussian priors
s["gpri_2_c"] = "orange"         # Colour of gaussian priors
s["gpri_2_a"] = 0.3              # Alpha of gaussian priors
s["margin_left_2"] = 0.05        # \
s["margin_bottom_2"] = 0.03      #  \
s["margin_right_2"] = 0.99       #   \ Margin
s["margin_top_2"] = 0.96         #   / settings
s["margin_wspace_2"] = 0.34      #  /
s["margin_hspace_2"] = 0.04      # /
s["quantile_instead_2"] = False  # Use quantiles instead of CI for overplot

# Plot 3 (Derived parameters plot)
s["alpha_3"] = 0.1      # Opacity of plotted curves for walkers
s["n_bins_3"] = 32      # Number of bins along step axis for cred. int. overplot
s["pct_3"] = [2.5, 16, 50, 84, 97.5]    # Which lnlike percentiles to overplot
s["pct_ls_3"] = [':','--','-','--',':'] # Linestyle for each percentile overplot
s["pct_lw_3"] = [1]*5                   # Line width for percentile overplot
s["pct_col_3"] = ["black"]*5            # Color for percentile overplot
s["CI_3"] = [68., 95.]           # Which credible interval to overplot
s["CI_3_ls"] = ['--', ':']       # Linestyle for each credible interval overplot
s["CI_3_lw"] = [1, 1]            # Linestyle for each credible interval overplot
s["CI_3_c1"] = "black"           # Colour of binned credible intervals
s["CI_3_c2"] = "cyan"            # Colour of left-half credible intervals
s["CI_3_c3"] = "magenta"         # Colour of right-half credible intervals
s["upri_3_h"] = "/"              # Hatch pattern of uniform priors
s["upri_3_c"] = "red"            # Colour of uniform priors
s["upri_3_a"] = 0.5              # Alpha of uniform priors
s["gpri_3_h"] = "\\"             # Hatch pattern of gaussian priors
s["gpri_3_c"] = "orange"         # Colour of gaussian priors
s["gpri_3_a"] = 0.3              # Alpha of gaussian priors
s["margin_left_3"] = 0.05        # \
s["margin_bottom_3"] = 0.03      #  \
s["margin_right_3"] = 0.99       #   \ Margin
s["margin_top_3"] = 0.96         #   / settings
s["margin_wspace_3"] = 0.34      #  /
s["margin_hspace_3"] = 0.04      # /
s["quantile_instead_3"] = False  # Use quantiles instead of CI for overplot

# Plot 4 (Mean acceptance plot)
s["n_bins_4"] = 32   # Number of bins along step axis for binned mean acceptance
s["mean_alpha"] = "blue"    # Colour of mean acceptance curve
s["mean_col"] = 0.25        # Alpha of mean acceptance curve
s["bmean_col"] = "black"    # Colour of binned mean acceptance curve
s["bmean_lw"] = 1.5         # Width of binned mean acceptance curve

# Plot 5 (Corner plot)
s["plot_density"] = False    # Display 2D density on corner plot
s["plot_datapoints"] = False # Display data points on corner plot
s["plot_contours"] = True    # Display 2D contour levels on corner plot
s["levels"] = [0.68, 0.95]   # Which levels to plot for 2D contours


###############################
### Parsing input arguments ###
###############################

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
    '-p',
    '--plot',
    metavar='LIST',
    action='store',
    nargs=1,
    help='list of plots to produce; argument should be a list of\n'
         'colon-separated parameter indexes.\n'
         'List of possible plots:\n'
         '1 : -log(likelihood) values for all walkers along chain\n'
         '2 : parameters values for all walkers along chain\n'
         '3 : derived parameters for all walkers along chain\n'
         '4 : acceptance rate averaged over walkers at each chain step\n'
         '5 : corner plot with all kept parameters (see -k and -kd options)\n'
         '> Example: 1:2:3-5  (hyphens indicate a range)\n'
)
parser.add_argument(
    '-pp',
    '--plot-parameters',
    metavar='LIST',
    action='store',
    nargs=1,
    help='list of parameters for customizing plots, see header of ECLAIR_plots.py\n'
         'for a detailed list; argument should be a list of colon-separated\n'
         'pairs in the form of parameter=value, written in python (i.e. with\n'
         'quotes for a string, brackets for an array, etc) and with no spaces.\n'
         'Note that quotes (for strings) need to be escaped with backslashes.'
)
parser.add_argument(
    '-b',
    '--burn-in',
    metavar='N',
    action='store',
    nargs=1,
    help='Apply burn-in to the chain.\n'
         'If N >= 1, int(N) is the number of burned-in steps.\n'
         'If 0 < N < 1, float(N) is the fraction of burned-in steps\n'
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
parser.add_argument(
    '-k',
    '--keep',
    metavar='LIST',
    action='store',
    nargs=1,
    help='Keep only some MCMC parameters for analysis; argument should be a \n'
         'list of colon-separated parameter indexes (use -sp to see list).\n'
         'Provide "None" as argument to remove all MCMC parameters.\n'
         '> Example: 0:5:7-8  (hyphens indicate a range)\n'
         '> Default: keep all MCMC parameters.',
)
parser.add_argument(
    '-kd',
    '--keep-derived',
    metavar='LIST',
    action='store',
    nargs=1,
    help='Keep only some derived parameters for analysis; argument should be \n'
         'a list of colon-separated parameter indexes (use -sp to see list).\n'
         'Provide "None" as argument to remove all derived parameters.\n'
         '> Example: 0:5:7-8  (hyphens indicate a range)\n'
         '> Default: keep all derived parameters.',
)
parser.add_argument(
    '-sp',
    '--show-parameters',
    action='store_true',
    help='Show the list of MCMC and derived parameters in the chain and \n'
         'their respective indexes.'
)
parser.add_argument(
    '-dvn',
    '--dict-var-names',
    metavar='DIR',
    action='store',
    help='Provide a dictionary (two-column text file) for switching\n'
         'the names of the variables.'
)
parser.add_argument(
    '-dl',
    '--dict-labels',
    metavar='DIR',
    action='store',
    help='Provide a dictionary (two-column text file) for defining\n'
         'parameter labels for plots.\n'
         '> Default: use the variable names'
)
parser.add_argument(
    '-og',
    '--output-getdist',
    action='store_true',
    help='Outputs a getdist-formatted version of the chains with\n'
         'same root file names.'
)
parser.add_argument(
    '-of',
    '--output-figures',
    action='store_true',
    help='Outputs requested figures in png format instead of displaying them\n'
         'on the screen. Figures will have the same root name as the chain\n'
         'with "_N.png" appended (where N is the figure number type).'
)

parser.add_argument(
    '-ps',
    '--print-summary',
    action='store_true',
    help='Outputs some summary statistics.'
)

args = parser.parse_args()


################################
### Checking input arguments ###
################################

# Check input chain path
if not os.path.isfile(args.file[0]):
    raise ValueError("Chain %s does not exist." % args.file[0])

# List of MCMC parameters to keep
keep = []
if args.keep is not None:
    if args.keep[0] == "None":
        pass
    else:
        keep = []
        tmp = args.keep[0].split(':')
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

# List of derived parameters to keep
keep_derived = []
if args.keep_derived is not None:
    if args.keep_derived[0] == "None":
        pass
    else:
        keep_derived = []
        tmp = args.keep_derived[0].split(':')
        for t in tmp:
            if is_number(t):
                keep_derived.append(int(t))
            else:
                st = t.split('-')
                test_1 = t.count('-') != 1
                test_2 = any([not is_number(tt) for tt in st])
                if test_1 | test_2:
                    raise ValueError("Bad formatting for range %s" % t)
                test_3 = int(st[0]) >= int(st[1])
                if test_3:
                    raise ValueError("Bad formatting for range %s" % t)
                keep_derived += range(int(st[0]), int(st[1]) + 1)
keep_derived = np.unique(keep_derived)

# List of plots to produce
plot = []
if args.plot is not None:
    tmp = args.plot[0].split(':')
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
            print("Unrecognised plot option %s, will be ignored." % p)
        else:
            plot.append(p)
plot.sort()

# List of plots parameters
if args.plot_parameters is not None:
    tmp = args.plot_parameters[0].split(":")
    for t in tmp:
        if t.count("=") != 1:
            print("Bad plot parameter format: %s, will be ignored." % t)
        else:
            st = t.split("=")
            if st[0] not in s.keys():
                print("Unrecognised plot parameter: %s, will be ignored." % st[0])
            else:
                exec(f"s['{st[0]}'] = {st[1]}")

# Burn-in settings
if args.burn_in is None:
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

# Thinning settings
if args.thin is None:
    thin = 1
else:
    if not is_number(args.thin[0]):
        raise ValueError("Input sample thinning is not a number (%s)" % args.thin[0])
    elif float(args.thin[0]) < 1:
        raise ValueError("Input sample thinning is < 1 (%s)" % args.thin[0])
    else:
        thin = int(args.thin[0])

# Walker thinning settings
if args.thin_walk is None:
    thinw = 1
else:
    if not is_number(args.thin_walk[0]):
        raise ValueError("Input walker thinning is not a number (%s)" % args.thin_walk[0])
    elif float(args.thin_walk[0]) < 1:
        raise ValueError("Input walker thinning is < 1 (%s)" % args.thin_walk[0])
    else:
        thinw = int(args.thin_walk[0])

# Dictionary for renaming variables
dvn = {}
if args.dict_var_names is not None:
    with open(args.dict_var_names, 'r') as f:
        lines = f.readlines()
    for line in lines:
        spline = line.split()
        is_comment_or_empty = spline == []
        if not is_comment_or_empty:
            is_comment_or_empty = not line.strip()[0].isalnum()
        if not is_comment_or_empty:
            if len(spline) == 2:
                if spline[0] in dvn.keys():
                    raise ValueError("Duplicate in variable names dictionary:\n%s" % line)
                dvn[spline[0]] = spline[1]
            else:
                raise ValueError("Formatting error in variable names dictionary:\n%s" % line)

# Dictionary for labelling variables
dl = {}
if args.dict_labels is not None:
    with open(args.dict_labels, 'r') as f:
        lines = f.readlines()
    for line in lines:
        spline = line.split()
        is_comment_or_empty = spline == []
        if not is_comment_or_empty:
            is_comment_or_empty = not line.strip()[0].isalnum()
        if not is_comment_or_empty:
            if len(spline) == 2:
                if spline[0] in dl.keys():
                    raise ValueError("Duplicate in labels names dictionary:\n%s" % line)
                dl[spline[0]] = spline[1]
            else:
                raise ValueError("Formatting error in labels names dictionary:\n%s" % line)


#################################
### Reading and parsing chain ###
#################################

fname = os.path.realpath(args.file[0]) # Input chain filename

# Read chain content, rename and relabel parameters if requested
ini_fname_nosuffix = fname.replace(".txt", "")
ini_fname = ini_fname_nosuffix + '.ini'
ini = parse_ini_file(ini_fname, silent_mode=True)
par_names = np.array([dvn[par[1]] if par[1] in dvn.keys() else par[1]
                      for par in ini['var_par']])
par_labels = np.array([dl[pn] if pn in dl.keys() else pn for pn in par_names])
n_par = len(ini['var_par'])
n_walkers = ini['n_walkers']
with open(fname, 'r') as f:
    header = f.readline()
all_names = [s.split(':')[1] for s in header[1:].split()]
n_blobs = len(all_names) - n_par - 2
blobs_names = np.array([dvn[n] if n in dvn.keys() else n
                        for n in all_names[-n_blobs:]])
blobs_labels = np.array([dl[n] if n in dl.keys() else n for n in blobs_names])
tmp = np.loadtxt(fname)#.reshape(-1, n_walkers, len(all_names))
ln = tmp[:, 1].reshape(-1, n_walkers)
ch = tmp[:, 2:2+n_par].reshape(-1, n_walkers, n_par)
bl = tmp[:, 2+n_par:].reshape(-1, n_walkers, n_blobs)
n_steps = ln.shape[0]

# Parameter printing, if requested
if args.show_parameters:
    print("MCMC parameters:")
    for ix, n in enumerate(par_names):
        print(f"- {ix} = {n}")
    print("Derived parameters:")
    for ix, n in enumerate(blobs_names):
        print(f"- {ix} = {n}")

# Read priors
lbs = np.array([par[3] for par in ini['var_par']])
ubs = np.array([par[4] for par in ini['var_par']])
pri_dict = {p[0]:[p[1], p[2]] for p in ini['gauss_priors']}
drv_gpri_dict = {p[0]:[p[1], p[2]] for p in ini['drv_gauss_priors']}
drv_upri_dict = {p[0]:[p[1], p[2]] for p in ini['drv_uni_priors']}

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
print('>>> Total number of samples plotted : %s (out of %s)' % (ln.shape[0] * ln.shape[1], n_walkers * n_steps))

# MCMC parameter selection
if args.keep is None:
    keep = np.arange(n_par)
g = (keep >= 0) & (keep < n_par)
if sum(g) == 0:
    # raise ValueError("No valid MCMC parameter index passed: %s" % keep)
    ch = ch[:, :, []]
    par_names = par_names[[]]
    par_labels = par_labels[[]]
    lbs = lbs[[]]
    ubs = ubs[[]]
else:
    ch = ch[:, :, keep[g]]
    par_names = par_names[keep[g]]
    par_labels = par_labels[keep[g]]
    lbs = lbs[keep[g]]
    ubs = ubs[keep[g]]
print('>>> Total number of MCMC parameters kept : %s (out of %s)' % (ch.shape[2], n_par))
n_steps, n_walkers, n_par = ch.shape

# Derived parameter selection
if args.keep_derived is None:
    keep_derived = np.arange(n_blobs)
g = (keep_derived >= 0) & (keep_derived < n_blobs)
if sum(g) == 0:
    # raise ValueError("No valid derived parameter index passed: %s" % keep_derived)
    bl = bl[:, :, []]
    blobs_names = blobs_names[[]]
    blobs_labels = blobs_labels[[]]
    drv_upri_dict = {}
else:
    bl = bl[:, :, keep_derived[g]]
    blobs_names = blobs_names[keep_derived[g]]
    blobs_labels = blobs_labels[keep_derived[g]]
    drv_to_del = []
    for k in drv_upri_dict.keys():
        if k not in blobs_names:
            drv_to_del.append(k)
    for k in drv_to_del:
        del drv_upri_dict[k]
print('>>> Total number of derived parameters kept : %s (out of %s)' % (bl.shape[2], n_blobs))
n_blobs = bl.shape[2]

# If requested, make getdist-formatted version of the chain
if args.output_getdist:
    from getdist.mcsamples import MCSamples
    gdist = MCSamples(
        ranges={**{par_names[i]: [lbs[i], ubs[i]] for i in range(n_par)}, **drv_upri_dict},
        samples=np.dstack((ch, bl)).reshape(-1, n_par + n_blobs),
        loglikes=(-1.*ln).reshape(-1),
        names=list(par_names) + list(blobs_names),
        labels=list(par_labels) + list(blobs_labels),
    )
    gdist.saveAsText(ini_fname_nosuffix + '_gdist')
    with open(ini_fname_nosuffix + '_gdist.settings', 'w') as f:
        f.write("Burn-in = %s\n" % burn)
        f.write("Sample thinning = %s\n" % thin)
        f.write("Walker thinning = %s\n" % thinw)


######################
### Misc functions ###
######################

# Find Smallest X Percent Credible Interval
def SC(samples, X):
    s = samples.flatten()
    l = len(s)
    f = X / 100. * l
    f_int = int(f)
    df = f - f_int
    sort_s = np.sort(s)
    diff = sort_s[f_int:]-sort_s[:-f_int]
    ix = diff.argmin()
    if ix == 0:
        lo = sort_s[0]
        hi = sort_s[f_int-1] + (sort_s[f_int] - sort_s[f_int-1]) * df
    elif ix == (l-f_int-1):
        lo = sort_s[l-f_int-1] - (sort_s[l-f_int] - sort_s[l-f_int-1]) * df
        hi = sort_s[-1]
    else:
        lo = sort_s[ix] + (sort_s[ix+1] - sort_s[ix]) * df / 2.
        hi = sort_s[f_int+ix] - (sort_s[f_int+ix+1] - sort_s[f_int+ix]) * df / 2.
    return [lo, hi]

# Return X percent symmetric quantiles
def SQ(samples, X):
    s = samples.flatten()
    half_X_pct = (100. - X) / 2. / 100.
    lo = np.quantile(s, half_X_pct)
    hi = np.quantile(s, 1. - half_X_pct)
    return [lo, hi]

# Compute Integrated Autocorrelation Time (adapted from Minas Karamanis)
def IAT(samples, c=5.0, norm=True):
    x = np.atleast_1d(samples.T.reshape((-1), order='C'))
    # Next largest power of 2
    n = 1
    while n < len(x):
        n = n << 1
    # Compute the auto-correlation function using FFT
    f = fft(x - np.mean(x), n=2 * n)
    acf = ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Normalize
    if norm:
        acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    # Automated windowing procedure following Sokal (1989)
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        window = np.argmin(m)
    else:
        window = len(taus) - 1
    return taus[window]


##################################
### Produce summary statistics ###
##################################

if args.print_summary:
    print(">>> Summary statistics:")
    print("MCMC parameters:")
    for ix, n in enumerate(par_names):
        iat = IAT(ch[:, :, ix])
        mean = np.mean(ch[:, :, ix])
        std = np.std(ch[:, :, ix])
        med = np.median(ch[:, :, ix])
        q1 = np.percentile(ch[:, :, ix], 16)
        q2 = np.percentile(ch[:, :, ix], 84)
        print(f"- {n}: {iat:.2f} (IAT), {mean:.3f} (mean), {std:.3f} (std), "
              f"{med:.3f} (median), [{q1:.3f}, {q2:.3f}] (68% CI)")
    print("Derived parameters:")
    for ix, n in enumerate(blobs_names):
        iat = IAT(bl[:, :, ix])
        mean = np.mean(bl[:, :, ix])
        std = np.std(bl[:, :, ix])
        med = np.median(bl[:, :, ix])
        q1 = np.percentile(bl[:, :, ix], 16)
        q2 = np.percentile(bl[:, :, ix], 84)
        print(f"- {n}: {iat:.2f} (IAT), {mean:.3f} (mean), {std:.3f} (std), "
              f"{med:.3f} (median), [{q1:.3f}, {q2:.3f}] (68% CI)")


#####################
### Produce plots ###
#####################

figs = []

# Likelihood plot
if 1 in plot:
    print("[[1]] Preparing likelihood plot...")
    fig1, ax1 = plt.subplots(1, 1,
                             figsize=(s['pix_x_size']/100,s['pix_y_size']/100))
    fig1.suptitle(fname)
    ax1.plot(-1. * ln, alpha=s["alpha_1"])
    bin_edges = np.ceil(np.linspace(0,n_steps-1,s["n_bins_1"]+1)).astype("int")
    for p, ls, lw, col in zip(s["pct_1"], s["pct_ls_1"], s["pct_lw_1"],
                              s["pct_col_1"]):
        x = np.array([], dtype='int')
        y = np.array([])
        for i_bin in range(s["n_bins_1"]):
            x = np.append(x, bin_edges[i_bin:i_bin+2])
            pct = np.percentile(-1. * ln[x[-2]:x[-1]+1, :], p)
            y = np.append(y, [pct, pct])
        ax1.plot(x, y, color=col, lw=lw, ls=ls)
    ax1.set_xlabel("MCMC step")
    ax1.set_ylabel("-log(likelihood)")
    figs.append((1, fig1))

# MCMC parameters plot
if (2 in plot) & (n_par == 0):
    print("Cannot do plot 2 because no MCMC parameters have been kept.")
elif (2 in plot):
    print("[[2]] Preparing MCMC parameter plot...")
    F = SQ if s["quantile_instead_2"] else SC
    nr, nc = 1, 1
    while (nr*nc) < n_par:
        if (nc/nr) <= (s["pix_x_size"]/s["pix_y_size"]):
            nc += 1
        else:
            nr += 1
    fig2, ax2 = plt.subplots(nr, nc, squeeze=False, sharex=True,
                             figsize=(s['pix_x_size']/100,s['pix_y_size']/100))
    fig2.suptitle(fname)
    b = np.round(np.linspace(0, n_steps-1, s["n_bins_2"]+1)).astype("int")
    for ix, (r, c) in enumerate(product(range(nr), range(nc))):
        if ix >= n_par:
            ax2[r, c].set_visible(False)
            continue
        ax2[r, c].plot(ch[:, :, ix], alpha=s["alpha_2"])
        y_lims = ax2[r,c].get_ylim()
        for CI, ls, lw in zip(s["CI_2"], s["CI_2_ls"], s["CI_2_lw"]):
            y = np.zeros((0, 2))
            for ib in range(s["n_bins_2"]):
                y = np.row_stack((y, F(ch[b[ib]:b[ib+1]+1, :, ix], CI)))
            y = np.row_stack((y, y[-1, :]))
            ax2[r,c].step(b, y[:, 0], where='post', color=s["CI_2_c1"],
                          lw=lw, ls=ls)
            ax2[r,c].step(b, y[:, 1], where='post', color=s["CI_2_c1"],
                          lw=lw, ls=ls)
            tmp = F(ch[:n_steps//2, :, ix], CI)
            ax2[r,c].axhline(tmp[0], xmax=0.5, color=s["CI_2_c2"], lw=lw, ls=ls)
            ax2[r,c].axhline(tmp[1], xmax=0.5, color=s["CI_2_c2"], lw=lw, ls=ls)
            tmp = F(ch[n_steps//2:, :, ix], CI)
            ax2[r,c].axhline(tmp[0], xmin=0.5, color=s["CI_2_c3"], lw=lw, ls=ls)
            ax2[r,c].axhline(tmp[1], xmin=0.5, color=s["CI_2_c3"], lw=lw, ls=ls)
        if np.isfinite(lbs[ix]):
            ax2[r, c].fill_between(range(ch.shape[0]), y_lims[0]-1, lbs[ix],
                                   hatch=s["upri_2_h"], color=s["upri_2_c"],
                                   alpha=s["upri_2_a"])
        if np.isfinite(ubs[ix]):
            ax2[r, c].fill_between(range(ch.shape[0]), ubs[ix], y_lims[1]+1,
                                   hatch=s["upri_2_h"], color=s["upri_2_c"],
                                   alpha=s["upri_2_a"])
        if par_names[ix] in pri_dict.keys():
            pri = pri_dict[par_names[ix]]
            ax2[r, c].fill_between(range(ch.shape[0]),
                                   pri[0] - pri[1], pri[0] + pri[1],
                                   hatch=s["gpri_2_h"], color=s["gpri_2_c"],
                                   alpha=s["gpri_2_a"])
        ax2[r, c].set_ylim(y_lims)
        ax2[r, c].set_ylabel(par_labels[ix])
        for label in ax2[r, c].get_yticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    fig2.subplots_adjust(
        left = s["margin_left_2"],
        bottom = s["margin_bottom_2"],
        right = s["margin_right_2"],
        top = s["margin_top_2"],
        wspace = s["margin_wspace_2"],
        hspace = s["margin_hspace_2"],
    )
    figs.append((2, fig2))

# Derived parameters plot
if (3 in plot) & (n_blobs == 0):
    print("Cannot do plot 3 because no derived parameters have been kept.")
elif (3 in plot):
    print("[[3]] Preparing derived parameter plot...")
    F = SQ if s["quantile_instead_3"] else SC
    nr, nc = 1, 1
    while (nr*nc) < n_blobs:
        if (nc/nr) <= (s["pix_x_size"]/s["pix_y_size"]):
            nc += 1
        else:
            nr += 1
    fig3, ax3 = plt.subplots(nr, nc, squeeze=False, sharex=True,
                             figsize=(s['pix_x_size']/100,s['pix_y_size']/100))
    fig3.suptitle(fname)
    b = np.round(np.linspace(0, n_steps-1, s["n_bins_3"]+1)).astype("int")
    for ix, (r, c) in enumerate(product(range(nr), range(nc))):
        if ix >= n_blobs:
            ax3[r, c].set_visible(False)
            continue
        ax3[r, c].plot(bl[:, :, ix], alpha=s["alpha_3"])
        y_lims = ax3[r,c].get_ylim()
        if ("lnprior" in blobs_names[ix]) or ("lnlike" in blobs_names[ix]):
            for p, ls, lw, col in zip(s["pct_3"], s["pct_ls_3"], s["pct_lw_3"],
                                      s["pct_col_3"]):
                y = []
                for ib in range(s["n_bins_3"]):
                    y.append(np.percentile(bl[b[ib]:b[ib+1]+1, :, ix], p))
                y.append(y[-1])
                ax3[r,c].step(b, y, where='post', color=col, lw=lw, ls=ls)
        else:
            for CI, ls, lw in zip(s["CI_3"], s["CI_3_ls"], s["CI_3_lw"]):
                y = np.zeros((0, 2))
                for ib in range(s["n_bins_3"]):
                    y = np.row_stack((y, F(bl[b[ib]:b[ib+1]+1, :, ix], CI)))
                y = np.row_stack((y, y[-1, :]))
                ax3[r,c].step(b, y[:, 0], where='post', color=s["CI_3_c1"],
                            lw=lw, ls=ls)
                ax3[r,c].step(b, y[:, 1], where='post', color=s["CI_3_c1"],
                            lw=lw, ls=ls)
                tmp = F(bl[:n_steps//3, :, ix], CI)
                ax3[r,c].axhline(tmp[0], xmax=0.5, color=s["CI_3_c2"],
                                 lw=lw, ls=ls)
                ax3[r,c].axhline(tmp[1], xmax=0.5, color=s["CI_3_c2"],
                                 lw=lw, ls=ls)
                tmp = F(bl[n_steps//2:, :, ix], CI)
                ax3[r,c].axhline(tmp[0], xmin=0.5, color=s["CI_3_c3"],
                                 lw=lw, ls=ls)
                ax3[r,c].axhline(tmp[1], xmin=0.5, color=s["CI_3_c3"],
                                 lw=lw, ls=ls)
        if blobs_names[ix] in drv_upri_dict.keys():
            pri = drv_upri_dict[blobs_names[ix]]
            ax3[r, c].fill_between(range(bl.shape[0]), y_lims[0]-1, pri[0],
                                   hatch=s["upri_3_h"], color=s["upri_3_c"],
                                   alpha=s["upri_3_a"])
            ax3[r, c].fill_between(range(bl.shape[0]), pri[1], y_lims[1]+1,
                                   hatch=s["upri_3_h"], color=s["upri_3_c"],
                                   alpha=s["upri_3_a"])
        if blobs_names[ix] in drv_gpri_dict.keys():
            pri = drv_gpri_dict[blobs_names[ix]]
            ax3[r, c].fill_between(range(bl.shape[0]),
                                   pri[0] - pri[1], pri[0] + pri[1],
                                   hatch=s["gpri_3_h"], color=s["gpri_3_c"],
                                   alpha=s["gpri_3_a"])
        ax3[r, c].set_ylim(y_lims)
        ax3[r, c].set_ylabel(blobs_labels[ix])
        for label in ax3[r, c].get_yticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    fig3.subplots_adjust(
        left = s["margin_left_3"],
        bottom = s["margin_bottom_3"],
        right = s["margin_right_3"],
        top = s["margin_top_3"],
        wspace = s["margin_wspace_3"],
        hspace = s["margin_hspace_3"],
    )
    figs.append((3, fig3))

# Mean acceptance plot
if (4 in plot) & (n_par == 0):
    print("Cannot do plot 4 because no MCMC parameters have been kept.")
elif (4 in plot):
    print("[[4]] Preparing mean acceptance plot...")
    fig4, ax4 = plt.subplots(1, 1,
                             figsize=(s['pix_x_size']/100,s['pix_y_size']/100))
    fig4.suptitle(fname)
    test = (ch[1:, :, 0] - ch[:-1, :, 0]) != 0.
    b = np.round(np.linspace(0, n_steps-2, s["n_bins_4"]+1)).astype("int")
    y = []
    for ib in range(s["n_bins_4"]):
        y.append(test[b[ib]:b[ib+1]+1, :].mean())
    y.append(y[-1])
    ax4.plot(test.mean(axis=1), color=s["mean_alpha"], alpha=s["mean_col"])
    ax4.step(b, y, where='post', color=s["bmean_col"], lw=s["bmean_lw"])
    ax4.set_xlabel("MCMC step")
    ax4.set_ylabel("Mean acceptance rate")
    figs.append((4, fig4))

# Corner plot
if (5 in plot) & (n_par == 0) & (n_blobs == 0):
    print("Cannot do plot 5 because no MCMC and derived parameters have been kept.")
elif (5 in plot):
    import corner
    print("[[5]] Preparing corner plot...")
    rch = np.zeros((n_steps * n_walkers, 0))
    if n_par != 0:
        rch = ch.reshape(-1, n_par)
    rbl = np.zeros((n_steps * n_walkers, 0))
    if n_blobs != 0:
        rbl = bl.reshape(-1, n_blobs)
    fig5 = corner.corner(
        np.column_stack((rch, rbl)),
        plot_density=s["plot_density"],
        plot_datapoints=s["plot_datapoints"],
        plot_contours=s["plot_contours"],
        levels=s["levels"],
        labels=np.concatenate((par_labels, blobs_labels)),
    )
    fig5.suptitle(fname)
    # fh = fig5.get_figheight()
    # fw = fig5.get_figwidth()
    # if s['pix_x_size'] > s['pix_y_size']:
    #     fig5.set_figheight(s['pix_y_size']/100)
    #     fig5.set_figwidth(s['pix_y_size']/100 * fw/fh)
    # else:
    #     fig5.set_figheight(s['pix_x_size']/100 * fh/fw)
    #     fig5.set_figwidth(s['pix_x_size']/100)
    figs.append((5, fig5))

# Show or save the plots
if len(figs) > 0:
    if args.output_figures:
        for ix_plot, fig in figs:
            figname = f"{ini_fname_nosuffix}_plot_{ix_plot}.png"
            print(f"Saving plot {ix_plot} to {figname}")
            fig.savefig(f"{ini_fname_nosuffix}_plot_{ix_plot}.png")
    else:
        plt.show()
