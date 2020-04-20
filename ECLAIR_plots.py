import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys
import os
import JAM_parser

do_copy = True #False

# Copy, read and parse chain
fname = sys.argv[1]
if do_copy:
    ran = str(np.random.rand())[2:]
    #os.system('cp -f %s /mnt/local-scratch/u/ilic/%s.h5' % (fname, ran))
    #reader = emcee.backends.HDFBackend('/mnt/local-scratch/u/ilic/%s.h5' % ran, read_only=True)
    os.system('cp -f %s /home/users/ilic/tmp/%s.h5' % (fname, ran))
    reader = emcee.backends.HDFBackend('/home/users/ilic/tmp/%s.h5' % ran, read_only=True)
else:
    reader = emcee.backends.HDFBackend(fname, read_only=True)
pars = JAM_parser.parse_ini_file(fname[:-3] + '.ini', ignore_errors=True)
names = np.array([par[1] for par in pars['var_par']])
lbs = np.array([par[3] for par in pars['var_par']])
ubs = np.array([par[4] for par in pars['var_par']])
if 'gauss_priors' in pars.keys():
    pri_dict = {}
    for p in pars['gauss_priors']:
        pri_dict[p[0]] = [p[1], p[2]]
if 'drv_gauss_priors' in pars.keys():
    drv_pri_dict = {}
    for p in pars['drv_gauss_priors']:
        drv_pri_dict[p[0]] = [p[1], p[2]]


# Store chain stuff
ln = reader.get_log_prob() #discard=burn_in, thin=thin)
ch = reader.get_chain() #discard=burn_in, thin=thin)
bl = reader.get_blobs() #discard=burn_in, thin=thin)
n_steps, n_walkers, n_par = ch.shape
n_blobs = len(bl.dtype.names)

# Grab plot parameters
burn = (sys.argv[2]).split('/')
burn_in, burn_out = float(burn[0]), float(burn[1])
thin = int(sys.argv[3])
thin_w = int(sys.argv[4])
print('###################################')
print('Input burn-in/out: %s/%s' % (burn_in, burn_out))
print('Input thinning : %s' % thin)
print('Input walker thinning : %s' % thin_w)
print('###################################')

# Burn & thin chain
if burn_in < 1.:
    n_burn_in = int(n_steps * burn_in)
else:
    n_burn_in = int(burn_in)
ix1 = n_burn_in
if burn_out < 1.:
    n_burn_out = int(n_steps * burn_out)
else:
    n_burn_out = int(burn_out)
ix2 = n_steps - n_burn_out
ln = ln[ix1:ix2, :]
ch = ch[ix1:ix2, :, :]
bl = bl[ix1:ix2, :]
print('Kept %s steps (out of %s) after burning' % (ln.shape[0], n_steps))
n_steps = ln.shape[0]
ln = ln[::thin, :]
ch = ch[::thin, :, :]
bl = bl[::thin, :]
print('Kept %s steps (out of %s) after thinning' % (ln.shape[0], n_steps))
ln = ln[:, ::thin_w]
ch = ch[:, ::thin_w, :]
bl = bl[:, ::thin_w]
print('Kept %s walkers (out of %s) after thinning' % (ln.shape[1], n_walkers))
print('###################################')
print('Total samples used : %s' % (ln.shape[0] * ln.shape[1]))
print('Total number of parameters : %s' % ch.shape[2])
print('###################################')

if len(sys.argv)>6:
    keep_par = [int(ix) for ix in (sys.argv[6]).split(',')]
    ch = ch[:, ::thin_w, keep_par]
    names = names[keep_par]
    lbs = lbs[keep_par]
    ubs = ubs[keep_par]
n_steps, n_walkers, n_par = ch.shape
print('Number of parameters kept : %s' % n_par)
print('###################################')


to_plot = (sys.argv[5]).split(':')

if 'temperature' in bl.dtype.names:
    print('Has field "temperature"')
    temp = bl['temperature']
elif 'real_lnprob' in bl.dtype.names:
    print('Has field "real_lnprob"')
    temp = bl['real_lnprob']/ln
elif 'real_lnl' in bl.dtype.names:
    print('Has field "real_lnl"')
    temp = bl['real_lnl']/ln
else:
    temp = ln*0. + 1.
print("Last temperature is : %s" % temp[-1, -1])

if 'R' in to_plot:
    rs = np.zeros(n_par)
    for i in range(n_par):
        rs[i] = ch[:, :, i].mean(axis=1).std()/ch[:, :, i].std(axis=1).mean()
        print("Equivalent R for parameter %s = %s" % (names[i], rs[i]))
    ix = rs.argmax()
    print("=======================================")
    print("Maximum R is for parameter %s : R = %s" % (names[ix], rs[ix]))
    print("=======================================")

if '1' in to_plot:
    fig = plt.figure()
    plt.plot(ln,alpha=0.1)
    if 'mini' not in fname:
        plt.axhline(np.percentile(ln, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        plt.axhline(np.percentile(ln, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        plt.axhline(np.percentile(ln, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
        plt.axhline(np.percentile(ln, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        plt.axhline(np.percentile(ln, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
    plt.plot(np.percentile(ln, 2.5, axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ln, 16, axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ln, 50, axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ln, 100-16., axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ln, 100.-2.5, axis=1), color='black', lw=2, ls='--')

if '1max' in to_plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,  gridspec_kw={'wspace': 0})
    ax1.plot(ln*temp,alpha=0.1)
    if 'mini' not in fname:
        ax1.axhline(np.percentile(ln*temp, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        ax1.axhline(np.percentile(ln*temp, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        ax1.axhline(np.percentile(ln*temp, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
        ax1.axhline(np.percentile(ln*temp, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        ax1.axhline(np.percentile(ln*temp, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
    ax1.plot(np.percentile(ln*temp, 2.5, axis=1), color='black', lw=2, ls='--')
    ax1.plot(np.percentile(ln*temp, 16, axis=1), color='black', lw=2, ls='--')
    ax1.plot(np.percentile(ln*temp, 50, axis=1), color='black', lw=2, ls='--')
    ax1.plot(np.percentile(ln*temp, 100-16., axis=1), color='black', lw=2, ls='--')
    ax1.plot(np.percentile(ln*temp, 100.-2.5, axis=1), color='black', lw=2, ls='--')
    ax1.plot((ln.mean(axis=1)+ch.shape[2]/2.)*temp[:, 0], color='grey')
    ax1.axhline((((ln.mean(axis=1)+ch.shape[2]/2.)*temp[:, 0])[-1]), color='red', ls='--')
    ####
    fakstd = np.max(ln*temp, axis=1)-np.median(ln*temp, axis=1)
    malntemp = np.ma.masked_array(ln*temp, mask=((ln*temp) <= (np.median(ln*temp, axis=1)-3.*fakstd)[:, None]))
    maln = np.ma.masked_array(ln, mask=((ln*temp) <= (np.median(ln*temp, axis=1)-3.*fakstd)[:, None]))
    matemp = np.ma.masked_array(temp, mask=((ln*temp) <= (np.median(ln*temp, axis=1)-3.*fakstd)[:, None]))
    ax2.plot(malntemp,alpha=0.1)
    if 'mini' not in fname:
        ax2.axhline(np.percentile(malntemp, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        ax2.axhline(np.percentile(malntemp, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        ax2.axhline(np.percentile(malntemp, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
        ax2.axhline(np.percentile(malntemp, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        ax2.axhline(np.percentile(malntemp, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
    ax2.plot(np.percentile(malntemp, 2.5, axis=1), color='black', lw=2, ls='--')
    ax2.plot(np.percentile(malntemp, 16, axis=1), color='black', lw=2, ls='--')
    ax2.plot(np.percentile(malntemp, 50, axis=1), color='black', lw=2, ls='--')
    ax2.plot(np.percentile(malntemp, 100-16., axis=1), color='black', lw=2, ls='--')
    ax2.plot(np.percentile(malntemp, 100.-2.5, axis=1), color='black', lw=2, ls='--')
    ax2.plot((maln.mean(axis=1)+ch.shape[2]/2.)*matemp[:, 0], color='purple')
    ax2.axhline((((maln.mean(axis=1)+ch.shape[2]/2.)*matemp[:, 0])[-1]), color='purple', ls='--')



if '-1' in to_plot:
    plt.plot(-ln,alpha=0.1)
    if 'mini' not in fname:
        plt.axhline(-np.percentile(ln, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        plt.axhline(-np.percentile(ln, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        plt.axhline(-np.percentile(ln, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
        plt.axhline(-np.percentile(ln, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
        plt.axhline(-np.percentile(ln, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
    plt.plot(-np.percentile(ln, 2.5, axis=1), color='black', lw=2, ls='--')
    plt.plot(-np.percentile(ln, 16, axis=1), color='black', lw=2, ls='--')
    plt.plot(-np.percentile(ln, 50, axis=1), color='black', lw=2, ls='--')
    plt.plot(-np.percentile(ln, 100-16., axis=1), color='black', lw=2, ls='--')
    plt.plot(-np.percentile(ln, 100.-2.5, axis=1), color='black', lw=2, ls='--')


if '2' in to_plot:
    fig = plt.figure()
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_par:
            break
        nc += 1
        if (nr*nc) >= n_par:
            break
        nr += 1
    for i in range(n_par):
        plt.subplot(nr,nc,i+1)
        plt.plot(ch[:, :, i],alpha=0.1)
        if 'mini' not in fname:
            plt.axhline(np.percentile(ch[:, :, i], 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            plt.axhline(np.percentile(ch[:, :, i], 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(ch[:, :, i], 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
            plt.axhline(np.percentile(ch[:, :, i], 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(ch[:, :, i], 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        plt.plot(np.percentile(ch[:, :, i], 2.5, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(ch[:, :, i], 16, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(ch[:, :, i], 50, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(ch[:, :, i], 100-16., axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(ch[:, :, i], 100.-2.5, axis=1), color='black', lw=2, ls='--')
        plt.ylabel(names[i])
        curr_ylim = plt.ylim()
        if (np.isfinite(lbs[i])) & (lbs[i] >= curr_ylim[0]):
            plt.axhline(lbs[i], ls='--', color='red')
        if (np.isfinite(ubs[i])) & (ubs[i] <= curr_ylim[1]):
            plt.axhline(ubs[i], ls='--', color='red')
        if 'gauss_priors' in pars.keys():
            if names[i] in pri_dict.keys():
                pri = pri_dict[names[i]]
                for m, ls in zip([-1.,0.,1.],['--','-','--']):
                            val = pri[0] + m * pri[1]
                            if (curr_ylim[0] <= val <= curr_ylim[1]):
                                plt.axhline(val, ls=ls, color='orange')
    fig.subplots_adjust(
        left = 0.05,
        bottom = 0.03,
        right = 0.99,
        top = 0.98,
        wspace = 0.34,
        hspace = 0.2,
    )

if '3' in to_plot:
    plt.figure()
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_blobs:
            break
        nc += 1
        if (nr*nc) >= n_blobs:
            break
        nr += 1
    nb = bl.dtype.names
    for i, n in enumerate(nb):
        plt.subplot(nr,nc,i+1)
        plt.plot(bl[n],alpha=0.1)
        if not 'mini' in fname:
            plt.axhline(np.percentile(bl[n], 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            plt.axhline(np.percentile(bl[n], 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(bl[n], 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
            plt.axhline(np.percentile(bl[n], 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(bl[n], 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
        plt.plot(np.percentile(bl[n], 2.5, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(bl[n], 16, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(bl[n], 50, axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(bl[n], 100-16., axis=1), color='black', lw=2, ls='--')
        plt.plot(np.percentile(bl[n], 100.-2.5, axis=1), color='black', lw=2, ls='--')
        plt.ylabel(n)
        curr_ylim = plt.ylim()
        if 'drv_gauss_priors' in pars.keys():
            if n in drv_pri_dict.keys():
                pri = drv_pri_dict[n]
                for m, ls in zip([-1.,0.,1.],['--','-','--']):
                            val = pri[0] + m * pri[1]
                            if (curr_ylim[0] <= val <= curr_ylim[1]):
                                plt.axhline(val, ls=ls, color='orange')
    if ('supersmartbin' in fname) | ('only3' in fname):
        plt.figure()
        if ('supersmartbin' in fname):
            imax, n1, n2 = 8, 2, 4
        else:
            imax, n1, n2 = 3, 1, 2
        for i in range(1, imax):
            plt.subplot(n1,n2,i)
            ix = names.index('f_k_%s' % i)
            if i==1:
                logtmp = ch[:, ::thin_w, ix]*(np.log10(0.18)-np.log10(0.00056))+np.log10(0.00056)
                tmp = 10.**logtmp
            else:
                tmp = 10.**(ch[:, ::thin_w, ix]*(np.log10(0.18)-logtmp)+logtmp)
                logtmp = ch[:, ::thin_w, ix]*(np.log10(0.18)-logtmp)+logtmp
            plt.plot(tmp,alpha=0.1)
            plt.axhline(np.percentile(tmp, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            plt.axhline(np.percentile(tmp, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(tmp, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
            plt.axhline(np.percentile(tmp, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            plt.axhline(np.percentile(tmp, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            plt.plot(np.percentile(tmp, 2.5, axis=1), color='black', lw=2, ls='--')
            plt.plot(np.percentile(tmp, 16, axis=1), color='black', lw=2, ls='--')
            plt.plot(np.percentile(tmp, 50, axis=1), color='black', lw=2, ls='--')
            plt.plot(np.percentile(tmp, 100-16., axis=1), color='black', lw=2, ls='--')
            plt.plot(np.percentile(tmp, 100.-2.5, axis=1), color='black', lw=2, ls='--')
            plt.ylabel('k_values_val_%s' % i)
        plt.figure()
        for i in range(1, imax):
            ix = names.index('f_k_%s' % i)
            if i==1:
                logtmp = ch[:, ::thin_w, ix]*(np.log10(0.18)-np.log10(0.00056))+np.log10(0.00056)
                tmp = 10.**logtmp
            else:
                tmp = 10.**(ch[:, ::thin_w, ix]*(np.log10(0.18)-logtmp)+logtmp)
                logtmp = ch[:, ::thin_w, ix]*(np.log10(0.18)-logtmp)+logtmp
            plt.semilogy(tmp,alpha=0.1,color='C%s' % (i-1))
            # plt.axhline(np.percentile(tmp, 2.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            # plt.axhline(np.percentile(tmp, 16, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            # plt.axhline(np.percentile(tmp, 50, axis=1)[n_steps//2:].mean(), color='blue', lw=2)
            # plt.axhline(np.percentile(tmp, 84, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='--')
            # plt.axhline(np.percentile(tmp, 97.5, axis=1)[n_steps//2:].mean(), color='blue', lw=2, ls='-.')
            # plt.semilogy(np.percentile(tmp, 2.5, axis=1), color='black', lw=2, ls='--')
            # plt.semilogy(np.percentile(tmp, 16, axis=1), color='black', lw=2, ls='--')
            # plt.semilogy(np.percentile(tmp, 50, axis=1), color='black', lw=2, ls='--')
            # plt.semilogy(np.percentile(tmp, 100-16., axis=1), color='black', lw=2, ls='--')
            # plt.semilogy(np.percentile(tmp, 100.-2.5, axis=1), color='black', lw=2, ls='--')
        plt.axhline(0.00056, color='black', lw=2)
        plt.axhline(0.18, color='black', lw=2)
        plt.ylabel('k_values')


if '4' in to_plot:
    plt.figure()
    test = (ch[1:, ::thin_w, 0] - ch[:-1, ::thin_w, 0]) != 0.
    ttest = np.cumsum(test, axis=0)/np.arange(1, test.shape[0]+1)[:, None].astype('float')
    plt.plot(ttest, alpha=0.1)
    plt.plot(np.percentile(ttest, 2.5, axis=1), color='black', lw=2, ls='-.')
    plt.plot(np.percentile(ttest, 16., axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ttest, 50, axis=1), color='black', lw=2)
    plt.plot(np.percentile(ttest, 84, axis=1), color='black', lw=2, ls='--')
    plt.plot(np.percentile(ttest, 97.5, axis=1), color='black', lw=2, ls='-.')

if '5' in to_plot:
    plt.figure()
    test = (ch[1:, ::thin_w, 0] - ch[:-1, ::thin_w, 0]) != 0.
    y = test.mean(axis=1)
    x = np.arange(len(y))
    plt.plot(test.mean(axis=1), color='blue')
    plt.plot(x, np.poly1d(np.polyfit(x,y,3))(x), color='black', lw=2)

if '6' in to_plot:
    plt.figure()
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_par:
            break
        nc += 1
        if (nr*nc) >= n_par:
            break
        nr += 1
    for i in range(n_par):
        plt.subplot(nr,nc,i+1)
        plt.plot(ch[:, :, i], ln, '+')
        plt.xlabel(names[i])

if '7_hist' in to_plot:
    plt.figure()
    nr, nc = 1, 1
    while True:
        if (nr*nc) >= n_par:
            break
        nc += 1
        if (nr*nc) >= n_par:
            break
        nr += 1
    for i in range(n_par):
        plt.subplot(nr,nc,i+1)
        plt.hist(ch[:, :, i].flatten(), bins=20, histtype='step')
        plt.xlabel(names[i])


if '7' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch[-1, :, :], plot_contours=False, plot_density=False, labels=names)

if '7_c' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=True, plot_datapoints=False, plot_density=False, labels=names)

if '7_p' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=False, plot_datapoints=True, plot_density=False, labels=names)

if '7_d' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=False, plot_datapoints=False, plot_density=True, labels=names)

if '7_cp' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=True, plot_datapoints=True, plot_density=False, labels=names)

if '7_cd' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=True, plot_datapoints=False, plot_density=True, labels=names)

if '7_dp' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=False, plot_datapoints=True, plot_density=True, labels=names)

if '7_cdp' in to_plot:
    plt.figure()
    import corner
    corner.corner(ch.reshape(-1, ch.shape[2]), plot_contours=True, plot_datapoints=True, plot_density=True, labels=names)



for ind, k in enumerate(to_plot):
    if '8cov' in k:
        n_par_max = int(k[5:])
        fig = plt.figure()
        n_comb = n_par_max*(n_par_max-1)/2
        nr, nc = 1, 1
        while True:
            if (nr*nc) >= n_comb:
                break
            nc += 1
            if (nr*nc) >= n_comb:
                break
            nr += 1
        ct = 1
        for i in range(n_par_max):
            for j in range(i+1, n_par_max):
                plt.subplot(nr,nc,ct)
                ct += 1
                quant = [np.cov(ch[ix, :, i], ch[ix, :, j])[0,1] for ix in range(n_steps)]
                plt.plot(quant)
                plt.ylabel("%s x %s" % (names[i], names[j]))
        fig.subplots_adjust(
            left = 0.05,
            bottom = 0.03,
            right = 0.99,
            top = 0.98,
            wspace = 0.34,
            hspace = 0.2,
        )

for ind, k in enumerate(to_plot):
    if '8cor' in k:
        n_par_max = int(k[5:])
        fig = plt.figure()
        n_comb = n_par_max*(n_par_max-1)/2
        nr, nc = 1, 1
        while True:
            if (nr*nc) >= n_comb:
                break
            nc += 1
            if (nr*nc) >= n_comb:
                break
            nr += 1
        ct = 1
        for i in range(n_par_max):
            for j in range(i+1, n_par_max):
                plt.subplot(nr,nc,ct)
                ct += 1
                quant = [np.corrcoef(ch[ix, :, i], ch[ix, :, j])[0,1] for ix in range(n_steps)]
                plt.plot(quant)
                plt.ylabel("%s x %s" % (names[i], names[j]))
        fig.subplots_adjust(
            left = 0.05,
            bottom = 0.03,
            right = 0.99,
            top = 0.98,
            wspace = 0.34,
            hspace = 0.2,
        )

if "9" not in to_plot:
    plt.show()


#sys.exit()

if do_copy:
    #os.system('rm /mnt/local-scratch/u/ilic/%s.h5' % ran)
    os.system('rm /home/users/ilic/tmp/%s.h5' % ran)

sys.exit()
#################################
def recur(D,N):
    if N == 1:
        return [[i] for i in range(D+1)]
    else:
        out = []
        for dN in range(D+1):
            sols = recur(D-dN, N-1)
            for sol in sols:
                out.append(sol + [dN])
        return out

n_deg = 2
all_pwr = np.array(recur(n_deg, n_par))
tmp_all_tmp = np.unique(temp)#[1:] # Exclude potentially incomplete last T step
all_tmp = [tmp_all_tmp[-1]]
for v in tmp_all_tmp[::-1]:
    if not np.isclose(all_tmp[-1], v):
        all_tmp.append(v)
all_tmp = np.array(all_tmp[::-1])


all_x, all_y = [], []

new_lbs = ch.reshape(-1, 51).min(axis=0)
new_ubs = ch.reshape(-1, 51).max(axis=0)

for ix in tqdm(range(5)):

    ix_T0, ix_T1 = 1, 10
    g = np.where((all_tmp[ix_T0+ix*10] <= temp) & (temp <= all_tmp[ix_T1+ix*10]) & ((ln*temp) >= (np.median(ln*temp, axis=1)-3.*fakstd)[:, None]))
    all_x.append([g[0].min() , g[0].max()])
    if 'real_lnprob' in bl.dtype.names:
        lnl = bl['real_lnprob'][g]
    else:
        lnl = bl['real_lnl'][g]

    quot = (ch[g].max(axis=0)-ch[g].min(axis=0))
    diff = ch[g].min(axis=0)
    ch2 = (ch[g]-diff)/quot
    # ch2 = ch[g]
    part_ch = np.zeros((ch2.shape[0], len(all_pwr)))
    for i, pwr in enumerate(all_pwr):
        part_ch[:, i] = np.prod(ch2**pwr,axis=1)

    from scipy.linalg import lstsq
    res = lstsq(part_ch, lnl)

    fitted_lnl = np.dot(part_ch, res[0])

    def new_mlnprob(p, res):
        if np.any(p<=new_lbs) | np.any(p>=new_ubs):
            return np.inf
        pp = (p-diff)/quot
        vec = np.prod(pp**all_pwr,axis=1)
        return -np.dot(vec, res[0])

    from scipy.optimize import minimize
    yay = minimize(new_mlnprob, ch2[0]*quot+diff, method='Nelder-Mead', args=(res,), options={"maxfev":1000000})

    all_y.append([-yay["fun"], -yay["fun"]])
