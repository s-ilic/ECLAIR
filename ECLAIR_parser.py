import numpy as np
from os.path import isfile


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def splt(s, sep=None, maxsplit=-1):
    ctp = s.count('"')
    if ctp == 0:
        return s.split(sep, maxsplit)
    if ctp % 2 == 1:
        # return ValueError("Error: even number of quotes: %s" % s)
        return None
    tmp = s.split(sep, maxsplit)
    fs = []
    in_quote = False
    store = ''
    for t in tmp:
        if ('"' in t) and not in_quote:
            in_quote = True
            store += t + ' '
        elif ('"' in t) and in_quote:
            in_quote = False
            store += t
            fs.append(store.replace('"',''))
            store = ''
        elif in_quote:
            store += t + ' '
        else:
            fs.append(t)
    return fs


def parse_ini_file(fname, silent_mode=False):

    ### Strings containing all warnings and errors
    str_warn = '\n################\n### WARNINGS ###\n################\n'
    str_err  = '################\n#### ERRORS ####\n################\n'

    ### Open the .ini file
    with open(fname) as f:
        lines = f.readlines()
    out =  {}
    error_ct = 0 # Error count, should stay 0

    ### Read all lines and options (==1st word on line)
    slines = []
    flines = []
    options = []
    for line in lines:
        sline = splt(line.replace('\n',''))
        if sline == None:
            str_err += 'Error: odd number of quotes\n> %s\n' % line
            error_ct += 1
        empty_line = sline == []
        is_comment = line[0] == '#'
        if not empty_line and not is_comment and sline is not None:
            slines.append(sline)
            flines.append(line.replace('\n',''))
            options.append(sline[0])

    ### Deal with debug mode
    ct = options.count('debug_mode')
    out['debug_mode'] = False
    if ct == 0:
        str_warn += '"debug_mode" not found. Assumed "no".\n'
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "debug_mode" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('debug_mode')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "debug_mode".\n'
            error_ct += 1
        elif slines[ix][1] not in ['yes', 'no']:
            str_err += '"debug_mode" should be "yes" or "no".\n'
            error_ct += 1
        elif slines[ix][1] == 'yes':
            out['debug_mode'] = True

    ### Deal with output format
    ct = options.count('output_format')
    out['output_format'] = 'text'
    suffix = '.txt'
    if ct == 0:
        str_warn += '"output_format" not found. Assumed "text".\n'
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "output_format" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('output_format')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "output_format".\n'
            error_ct += 1
        elif slines[ix][1] not in ['text', 'HDF5']:
            str_err += '"output_format" should be "text" or "HDF5".\n'
            error_ct += 1
        elif slines[ix][1] == 'HDF5':
            try:
                import h5py
            except:
                str_err += 'You need to install the "h5py" module in order to use the HDF5 file format.\n'
                error_ct += 1
            out['output_format'] = 'HDF5'
            suffix = '.h5'

    ### Deal with output_root
    ct = options.count('output_root')
    if ct == 0:
        str_err += '"output_root" not found.\n'
        out['output_root'] = None
        error_ct += 1
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "output_root" found.\n' % ct
        out['output_root'] = None
        error_ct += 1
    else:
        ix = options.index('output_root')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "output_root".\n'
            out['output_root'] = None
            error_ct += 1
        elif is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "output_root".\n'
            out['output_root'] = None
            error_ct += 1
        else:
            out['output_root'] = slines[ix][1]

    ### Deal with input_type
    if ('input_type' in options) & ('input_fname' not in options):
        str_warn += '"input_type" found, but not "input_fname". Ignored.\n'
        out['input_type'] = None
    elif ('input_type' not in options) & ('input_fname' in options):
        str_err += '"input_fname" found, but not "input_type".\n'
        out['input_type'] = None
        error_ct += 1
    elif ('input_type' in options) & ('input_fname' in options):
        ct = options.count('input_type')
        if ct > 1:
            str_err += 'Multiple (%s) instances of "input_type" found.\n' % ct
            out['input_type'] = None
            error_ct += 1
        else:
            ix = options.index('input_type')
            if not (1 < len(slines[ix]) < 4):
                str_err += 'Wrong number of arguments for "input_type".\n'
                out['input_type'] = None
                error_ct += 1
            elif slines[ix][1] not in ['text_chain', 'HDF5_chain', 'walkers']:
                str_err += 'Unrecognizd argument for "input_type" : %s.\n' % slines[ix][1]
                out['input_type'] = None
                error_ct += 1
            elif 'chain' in slines[ix][1]:
                out['input_type'] = slines[ix][1]
                if len(slines[ix]) == 2:
                   out['ch_start'] = -1
                elif not is_number(slines[ix][2]):
                    str_err += 'Wrong argument type for "input_type".\n'
                    out['input_type'] = None
                    error_ct += 1
                else:
                    out['ch_start'] = int(slines[ix][2])
            elif len(slines[ix]) != 2:
                str_err += 'Wrong number of arguments for "input_type".\n'
                out['input_type'] = None
                error_ct += 1
            else:
                out['input_type'] = slines[ix][1]
    else:
        out['input_type'] = None

    ### Deal with input_fname
    ct = options.count('input_fname')
    if ct == 0:
        str_warn += '"input_fname" not found, starting with random positions.\n'
        out['input_fname'] = None
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "input_fname" found.\n' % ct
        out['input_fname'] = None
        error_ct += 1
    else:
        ix = options.index('input_fname')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "input_fname".\n'
            out['input_fname'] = None
            error_ct += 1
        elif is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "input_fname".\n'
            out['input_fname'] = None
            error_ct += 1
        elif out['input_type'] == 'text_chain':
            out['input_fname'] = slines[ix][1][:-4]
        elif out['input_type'] == 'HDF5_chain':
            out['input_fname'] = slines[ix][1][:-3]
        else:
            out['input_fname'] = slines[ix][1]

    ### Deal with continue_chain
    ct = options.count('continue_chain')
    out['continue_chain'] = False
    if ct == 0:
        str_err += '"continue_chain" not found. Assumed "no".\n'
        error_ct += 1
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "continue_chain" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('continue_chain')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "continue_chain".\n'
            error_ct += 1
        elif slines[ix][1] not in ['yes', 'no']:
            str_err += 'Wrong argument for "continue_chain" : %s.\n' % slines[ix][1]
            error_ct += 1
        elif slines[ix][1] == 'yes':
            if not isfile(out['output_root']+suffix):
                str_err += 'The chain you want to continue from (%s) does not exist.\n' % (out['output_root']+suffix)
                error_ct += 1
            else:
                out['continue_chain'] = True
        elif isfile(out['output_root']+suffix):
            str_err += 'The output chain (%s) already exists.\n' % (out['output_root']+suffix)
            error_ct += 1


    ### Deal with parallel
    if 'parallel' not in options:
        str_warn += '"parallel" not found, assuming "none".\n'
        out['parallel'] = ['none']
    else:
        ct = options.count('parallel')
        if ct > 1:
            str_err += 'Multiple (%s) instances of "parallel" found.\n' % ct
            error_ct += 1
        else:
            ix = options.index('parallel')
            if slines[ix][1] not in ['none', 'multiprocessing', 'MPI']:
                str_err += 'Unrecognizd argument for "parallel" : %s.\n' % slines[ix][1]
                error_ct += 1
            elif slines[ix][1] == 'multiprocessing':
                if len(slines[ix]) != 3:
                    str_err += 'Wrong number of arguments for "parallel".\n'
                    error_ct += 1
                elif not is_number(slines[ix][2]):
                    str_err += 'Wrong argument type in "parallel".\n'
                    error_ct += 1
                else:
                    out['parallel'] = slines[ix][1:]
            else:
                if len(slines[ix]) != 2:
                    str_err += 'Wrong number of arguments for "parallel".\n'
                    error_ct += 1
                else:
                    out['parallel'] = [slines[ix][1]]

    ### Deal with n_walkers
    ct = options.count('n_walkers')
    if ct == 0:
        str_warn += '"n_walkers" not found, assuming 2 times the number of parameters.\n'
        out['n_walkers_type'] = 'prop_to'
        out['n_walkers'] = 2
    elif ct > 1:
        str_err += 'Multiple (%s) instances of "n_walkers" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('n_walkers')
        if len(slines[ix]) != 3:
            str_err += 'Wrong number of arguments for "n_walkers".\n'
            error_ct += 1
        elif slines[ix][1] not in ['custom', 'prop_to']:
            str_err += 'Unrecognizd argument for "n_walkers" : %s.\n' % slines[ix][1]
            error_ct += 1
        elif not is_number(slines[ix][2]):
            str_err += 'Wrong argument type for "n_walkers".\n'
            error_ct += 1
        else:
            out['n_walkers_type'] = slines[ix][1]
            out['n_walkers'] = int(slines[ix][2])

    ### Deal with n_steps
    ct = options.count('n_steps')
    if ct == 0:
        str_warn += '"n_steps" not found, assuming 10000 steps.\n'
        out['n_steps'] = 10000
    elif ct > 2:
        str_err += 'Multiple (%s) instances of "n_steps" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('n_steps')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "n_steps".\n'
            error_ct += 1
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "n_steps".\n'
            error_ct += 1
        else:
            out['n_steps'] = int(slines[ix][1])

    ### Deal with thin_by
    ct = options.count('thin_by')
    if ct == 0:
        str_warn += '"thin_by" not found, assuming no thinning.\n'
        out['thin_by'] = 1
    elif ct > 2:
        str_err += 'Multiple (%s) instances of "thin_by" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('thin_by')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "thin_by".\n'
            error_ct += 1
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "thin_by".\n'
            error_ct += 1
        else:
            out['thin_by'] = int(slines[ix][1])

    ### Deal with temperature
    ct = options.count('temperature')
    if ct == 0:
        str_warn += '"temperature" not found, assuming temperature = 1.\n'
        out['temperature'] = 1
    elif ct > 2:
        str_err += 'Multiple (%s) instances of "temperature" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('temperature')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "temperature".\n'
            error_ct += 1
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "temperature".\n'
            error_ct += 1
        else:
            out['temperature'] = float(slines[ix][1])

    ### Deal with stretch
    ct = options.count('stretch')
    if ct == 0:
        str_warn += '"stretch" not found, assuming default stretch of 2.\n'
        out['stretch'] = 1
    elif ct > 2:
        str_err += 'Multiple (%s) instances of "stretch" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('stretch')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "stretch".\n'
            error_ct += 1
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "stretch".\n'
            error_ct += 1
        elif float(slines[ix][1]) <= 1:
            str_err += 'Wrong value for "stretch": has to be > 1.\n'
            error_ct += 1
        else:
            out['stretch'] = float(slines[ix][1])

    ### Deal with which_class
    ct = options.count('which_class')
    if ct == 0:
        str_warn += '"which_class" not found, assuming regular class.\n'
        out['which_class'] = 'classy'
    elif ct > 2:
        str_err += 'Multiple (%s) instances of "which_class" found.\n' % ct
        error_ct += 1
    else:
        ix = options.index('which_class')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "which_class".\n'
            error_ct += 1
        elif is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "which_class".\n'
            error_ct += 1
        else:
            out['which_class'] = slines[ix][1]

    ### Check if any likelihood
    ct = options.count('likelihood')
    if ct == 0:
        str_err += 'No likelihood specified.\n'
        error_ct += 1

    ### Check if any free variable
    ct = options.count('var') + options.count('var_class')
    if ct == 0:
        str_err += 'No free parameters specified.\n'
        error_ct += 1

    ### Loop to catch parameters that can have multiple instances
    out['constraints'] = []
    out['likelihoods'] = []
    out['derivs'] = []
    out['gauss_priors'] = []
    out['drv_gauss_priors'] = []
    out['var_par'] = []
    out['array_var'] = {}
    out['base_par_class'] = {}
    out['base_par_likes'] = {}
    vp_names = []  # all varying parameter names
    bpc_names = [] # non-varying class parameters names
    bpl_names = [] # non-varying other parameters names
    arr_names = [] # array parameter names
    cst_names = [] # constrained parameter names
    drv_names = [] # derived parameter names
    for ix, sline in enumerate(slines):
        # Deal with constraints
        if sline[0] == 'constraint':
            if len(sline) < 4:
                str_err += 'Wrong number of arguments for "constraint".\n'
                error_ct += 1
            else:
                good1 = sline[2] == '='
                good2 = ' '.join(sline[1:]).count('=') == 1
                if good1 & good2:
                    tmp_cst = sline[1:]
                    for i in [0,2]:
                        tmp_cst[i] = tmp_cst[i].replace("class", "class_input")
                        tmp_cst[i] = tmp_cst[i].replace("likes", "likes_input")
                        tmp_cst[i] = tmp_cst[i].replace("[","['")
                        tmp_cst[i] = tmp_cst[i].replace("]","']")
                    out['constraints'].append(' '.join(tmp_cst))
                    tmp_cst_name = sline[1].replace('class','').replace('likes','')
                    tmp_cst_name = tmp_cst_name.replace('[','').replace(']','')
                    cst_names.append(tmp_cst_name)
                    if '_val_' in sline[1]:
                        tmp_arr_name = sline[1]
                        tmp_arr_name = tmp_arr_name.replace("class", "")
                        tmp_arr_name = tmp_arr_name.replace("likes", "")
                        tmp_arr_name = tmp_arr_name.replace("[","")
                        tmp_arr_name = tmp_arr_name.replace("]","")
                        arr_names.append(tmp_arr_name)
                        tmp = tmp_arr_name.split('_val_')
                        if tmp[0] not in out['array_var'].keys():
                            out['array_var'][tmp[0]] = 0
                else:
                    str_err += 'Wrong format for "constraint": \n> %s\n' % flines[ix]
                    error_ct += 1
        # Deal with likelihood
        elif sline[0] == 'likelihood':
            if len(sline) != 2:
                str_err += 'Wrong number of arguments for "likelihood".\n'
                error_ct += 1
            else:
                out['likelihoods'].append(sline[1])
        # Deal with deriv
        elif sline[0] == 'deriv':
            fline = splt(flines[ix])
            if len(fline) != 3:
                str_err += 'Wrong number of arguments for "deriv": \n> %s\n' % flines[ix]
                error_ct += 1
            elif is_number(fline[1]) | is_number(fline[2]):
                str_err += 'Wrong argument type for "deriv": \n> %s\n' % flines[ix]
                error_ct += 1
            else:
                out['derivs'].append([fline[1], fline[2]])
                drv_names.append(fline[1])
        # Deal with var/var_class
        elif (sline[0] == 'var_class') | (sline[0] == 'var'):
            good1 = len(sline) == 6
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                str_err += 'Wrong "var/var_class" format:\n> %s\n' % flines[ix]
                error_ct += 1
            else:
                out['var_par'].append(sline[:2] + [float(x) for x in sline[2:]])
                vp_names.append(sline[1])
                if '_val_' in sline[1]:
                    arr_names.append(sline[1])
                    tmp = sline[1].split('_val_')
                    if tmp[0] not in out['array_var'].keys():
                        out['array_var'][tmp[0]] = 0
        # Deal with gauss_prior
        elif sline[0] == 'gauss_prior':
            good1 = len(sline) == 4
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                str_err += 'Wrong "gauss_prior" format:\n> %s\n' % flines[ix]
                error_ct += 1
            else:
                out['gauss_priors'].append([sline[1]] + [float(x) for x in sline[2:]])
        # Deal with fix_class
        elif sline[0] == 'fix_class':
            if len(sline) != 3:
                str_err += 'Wrong "fix_class" format:\n> %s\n' % flines[ix]
                error_ct += 1
            elif is_number(sline[2]):
                out['base_par_class'][sline[1]] = float(sline[2])
                bpc_names.append(sline[1])
                if '_val_' in sline[1]:
                    arr_names.append(sline[1])
                    tmp = sline[1].split('_val_')
                    if tmp[0] not in out['array_var'].keys():
                        out['array_var'][tmp[0]] = 0
            else:
                out['base_par_class'][sline[1]] = sline[2]
                bpc_names.append(sline[1])
        # Deal with fix
        elif sline[0] == 'fix':
            if len(sline) != 3:
                str_err += 'Wrong "fix" format:\n> %s\n' % flines[ix]
                error_ct += 1
            elif is_number(sline[1]) | (not is_number(sline[2])):
                str_err += 'Wrong "fix" format:\n> %s\n' % flines[ix]
                error_ct += 1
            else:
                out['base_par_likes'][sline[1]] = float(sline[2])
                bpl_names.append(sline[1])
        # Warn about unknown options
        else:
            known_other_options = [
                'debug_mode',
                'output_format',
                'output_root',
                'input_fname',
                'input_type',
                'continue_chain',
                'parallel',
                'n_walkers',
                'n_steps',
                'thin_by',
                'temperature',
                'stretch',
                'which_class',
            ]
            if sline[0] not in known_other_options:
                str_warn += 'Unrecognized option "%s". Ignored.\n' % sline[0]
    # Adjust number of walkers if "proportional" option is requested
    if out['n_walkers_type'] == 'prop_to':
        out['n_walkers'] *= len(out['var_par'])

    ### Check for duplicate parameters
    for n in vp_names:
        if vp_names.count(n) > 1:
            str_err += 'Duplicate "var" parameter: %s\n' % n
            error_ct +=1
        if (n in bpc_names) or (n in bpl_names):
            str_err += 'Error: parameter "%s" is both fixed and varying.\n' % n
            error_ct +=1
    for n in bpc_names:
        if bpc_names.count(n) > 1:
            str_err += 'Duplicate "fix_class" parameter: %s\n' % n
            error_ct +=1
    for n in bpl_names:
        if bpl_names.count(n) > 1:
            str_err += 'Duplicate "fix" parameter: %s\n' % n
            error_ct +=1

    ### Checks for parameter arrays
    for n in out['array_var'].keys():
        ixs = []
        for av in arr_names:
            tmp = av.split('_val_')
            if tmp[0] == n:
                ixs.append(int(tmp[1]))
        for i in range(max(ixs)+1):
            not_good1 = '%s_val_%s' % (n, i) not in arr_names
            not_good2 = '%s_val_%s' % (n, i) not in cst_names
            if not_good1 & not_good2:
                str_err += 'Error: element %s of parameter array "%s" missing.\n' % (i, n)
                error_ct +=1
        out['array_var'][n] = max(ixs)+1

    ### Checks constraints
    for cst_name in cst_names:
        if cst_name in (vp_names + bpc_names + bpl_names):
            str_err += 'Error: %s is both constrained and variable/fixed.\n' % cst_name
            error_ct +=1

    ### Checks for derived parameters
    for der in out['derivs']:
        if  ('mPk' not in out['base_par_class']['output']) & ('sigma8' in der):
            str_err += 'sigma_8 asked as a derived parameter, but mPk not in output.\n'
            error_ct +=1

    ### Checks for Gaussian priors
    for i, p in enumerate(out['gauss_priors']):
        if p[0] not in (vp_names + drv_names):
            str_err += 'Error: parameter %s has prior but is not an MCMC or derived parameter.\n' % p[0]
            error_ct +=1
        if p[0] in drv_names:
            out['drv_gauss_priors'].append(out['gauss_priors'].pop(i))

    ### Raise error if any problem detected, else return final dictionary
    if error_ct == 0:
        str_err += 'None.\n'
    if not silent_mode:
        print(str_warn)
        print(str_err)
    if silent_mode:
        return out
    elif error_ct >= 1:
        raise ValueError('Check your .ini file for the error(s) detected above.')
    else:
        return out


def copy_ini_file(fname, params):

    ### Write copy of ini file
    with open(fname) as f:
        lines = f.readlines()
    with open(params['output_root'] + '.ini', 'w') as f:
        for line in lines:
            empty_line = line.split() == []
            comment = line[0] == '#'
            if not empty_line and not comment:
                f.write(line)
    return None
