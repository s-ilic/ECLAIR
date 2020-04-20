import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_ini_file(fname, ignore_errors=False):

    ### Open the .ini file
    with open(fname) as f:
        lines = f.readlines()
    out =  {}
    error_ct = 0

    ### Read all lines and options (==1st word on line)
    slines = []
    flines = []
    options = []
    for line in lines:
        sline = line.replace('\n','').split()
        empty_line = sline == []
        is_comment = line[0] == '#'
        if not empty_line and not is_comment:
            slines.append(sline)
            flines.append(line.replace('\n',''))
            options.append(sline[0])

    ### Deal with output_root
    ct = options.count('output_root')
    if ct == 0:
        print('"output_root" not found.')
        out['output_root'] = None
        error_ct += 1
    elif ct > 1:
        print('%s instances of "output_root" found.' % ct)
        out['output_root'] = None
        error_ct += 1
    else:
        ix = options.index('output_root')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "output_root".')
            out['output_root'] = None
            error_ct += 1
        elif is_number(slines[ix][1]):
            print('Wrong argument type for "output_root".')
            out['output_root'] = None
            error_ct += 1
        else:
            out['output_root'] = slines[ix][1]

    ### Deal with input_type
    if ('input_type' in options) & ('input_fname' not in options):
        print('"input_type" found, but not "input_fname". Ignored.')
        out['input_type'] = None
    elif ('input_type' not in options) & ('input_fname' in options):
        print('"input_fname" found, but not "input_type".')
        out['input_type'] = None
        error_ct += 1
    elif ('input_type' in options) & ('input_fname' in options):
        ct = options.count('input_type')
        if ct > 1:
            print('%s instances of "input_type" found.' % ct)
            out['input_type'] = None
            error_ct += 1
        else:
            ix = options.index('input_type')
            if not (1 < len(slines[ix]) < 4):
                print('Wrong number of arguments for "input_type".')
                out['input_type'] = None
                error_ct += 1
            elif slines[ix][1] not in ['chain', 'one_walker', 'many_walkers']:
                print('Unrecognizd argument for "input_type" : %s.' % slines[ix][1])
                out['input_type'] = None
                error_ct += 1
            elif slines[ix][1] == 'chain':
                out['input_type'] = 'chain'
                if len(slines[ix]) == 2:
                   out['ch_start'] = -1
                elif not is_number(slines[ix][2]):
                    print('Wrong argument type for "input_type".')
                    out['input_type'] = None
                    error_ct += 1
                else:
                    out['ch_start'] = int(slines[ix][2])
            elif len(slines[ix]) != 2:
                print('Wrong number of arguments for "input_type".')
                out['input_type'] = None
                error_ct += 1
            else:
                out['input_type'] = slines[ix][1]
    else:
        out['input_type'] = None

    ### Deal with input_fname
    ct = options.count('input_fname')
    if ct == 0:
        print('"input_fname" not found, starting with random positions.')
        out['input_fname'] = None
    elif ct > 1:
        print('%s instances of "input_fname" found.' % ct)
        out['input_fname'] = None
        error_ct += 1
    else:
        ix = options.index('input_fname')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "input_fname".')
            out['input_fname'] = None
            error_ct += 1
        elif is_number(slines[ix][1]):
            print('Wrong argument type for "input_fname".')
            out['input_fname'] = None
            error_ct += 1
        elif out['input_type'] == 'chain':
            out['input_fname'] = slines[ix][1][:-3]
        else:
            out['input_fname'] = slines[ix][1]

    ### Deal with continue_chain
    ct = options.count('continue_chain')
    if ct == 0:
        print('"continue_chain" not found.')
        error_ct += 1
    elif ct > 1:
        print('%s instances of "continue_chain" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('continue_chain')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "continue_chain".')
            error_ct += 1
        elif slines[ix][1] not in ['yes', 'no']:
            print('Wrong argument for "continue_chain" : %s.' % slines[ix][1])
            error_ct += 1
        else:
            if slines[ix][1] == 'yes':
                out['continue_chain'] = True
            else:
                out['continue_chain'] = False

    ### Deal with parallel
    if 'parallel' not in options:
        print('"parallel" not found, assuming "none".')
        out['parallel'] = ['none']
    else:
        ct = options.count('parallel')
        if ct > 1:
            print('%s instances of "parallel" found.' % ct)
            error_ct += 1
        else:
            ix = options.index('parallel')
            if slines[ix][1] not in ['none', 'multiprocessing', 'MPI']:
                print('Unrecognizd argument for "parallel" : %s.' % slines[ix][1])
                error_ct += 1
            elif slines[ix][1] == 'multiprocessing':
                if len(slines[ix]) != 3:
                    print('Wrong number of arguments for "parallel".')
                    error_ct += 1
                elif not is_number(slines[ix][2]):
                    print('Wrong argument type in "parallel".')
                    error_ct += 1
                else:
                    out['parallel'] = slines[ix][1:]
            else:
                if len(slines[ix]) != 2:
                    print('Wrong number of arguments for "parallel".')
                    error_ct += 1
                else:
                    out['parallel'] = [slines[ix][1]]

    ### Deal with n_walkers
    ct = options.count('n_walkers')
    if ct == 0:
        print('"n_walkers" not found, assuming 2 times the number of parameters.')
        out['n_walkers_type'] = 'prop_to'
        out['n_walkers'] = 2
    elif ct > 1:
        print('%s instances of "n_walkers" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('n_walkers')
        if len(slines[ix]) != 3:
            print('Wrong number of arguments for "n_walkers".')
            error_ct += 1
        elif slines[ix][1] not in ['custom', 'prop_to']:
            print('Unrecognizd argument for "n_walkers" : %s.' % slines[ix][1])
            error_ct += 1
        elif not is_number(slines[ix][2]):
            print('Wrong argument type for "n_walkers".')
            error_ct += 1
        else:
            out['n_walkers_type'] = slines[ix][1]
            out['n_walkers'] = int(slines[ix][2])

    ### Deal with n_steps
    ct = options.count('n_steps')
    if ct == 0:
        print('"n_steps" not found, assuming 10000 steps.')
        out['n_steps'] = 10000
    elif ct > 2:
        print('%s instances of "n_steps" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('n_steps')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "n_steps".')
            error_ct += 1
        elif not is_number(slines[ix][1]):
            print('Wrong argument type for "n_steps".')
            error_ct += 1
        else:
            out['n_steps'] = int(slines[ix][1])

    ### Deal with thin_by
    ct = options.count('thin_by')
    if ct == 0:
        print('"thin_by" not found, assuming no thinning.')
        out['thin_by'] = 1
    elif ct > 2:
        print('%s instances of "thin_by" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('thin_by')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "thin_by".')
            error_ct += 1
        elif not is_number(slines[ix][1]):
            print('Wrong argument type for "thin_by".')
            error_ct += 1
        else:
            out['thin_by'] = int(slines[ix][1])

    ### Deal with temperature
    ct = options.count('temperature')
    if ct == 0:
        print('"temperature" not found, assuming temperature = 1.')
        out['temperature'] = 1
    elif ct > 2:
        print('%s instances of "temperature" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('temperature')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "temperature".')
            error_ct += 1
        elif not is_number(slines[ix][1]):
            print('Wrong argument type for "temperature".')
            error_ct += 1
        else:
            out['temperature'] = float(slines[ix][1])

    ### Deal with stretch
    ct = options.count('stretch')
    if ct == 0:
        print('"stretch" not found, assuming default stretch of 2.')
        out['stretch'] = 1
    elif ct > 2:
        print('%s instances of "stretch" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('stretch')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "stretch".')
            error_ct += 1
        elif not is_number(slines[ix][1]):
            print('Wrong argument type for "stretch".')
            error_ct += 1
        elif float(slines[ix][1]) <= 1:
            print('Wrong value for "stretch": has to be > 1.')
            error_ct += 1
        else:
            out['stretch'] = float(slines[ix][1])

    ### Deal with which_class
    ct = options.count('which_class')
    if ct == 0:
        print('"which_class" not found, assuming regular class.')
        out['which_class'] = 'classy'
    elif ct > 2:
        print('%s instances of "which_class" found.' % ct)
        error_ct += 1
    else:
        ix = options.index('which_class')
        if len(slines[ix]) != 2:
            print('Wrong number of arguments for "which_class".')
            error_ct += 1
        elif is_number(slines[ix][1]):
            print('Wrong argument type for "which_class".')
            error_ct += 1
        else:
            out['which_class'] = slines[ix][1]

    ### Check if any likelihood
    ct = options.count('likelihood')
    if ct == 0:
        print('No likelihood specified.')
        error_ct += 1

    ### Check if any free variable
    ct = options.count('var') + options.count('var_class')
    if ct == 0:
        print('No free parameters specified.')
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
                print('Wrong number of arguments for "constraint".')
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
                    print('Wrong format for "constraint": \n> %s' % sline)
                    error_ct += 1
        # Deal with likelihood
        elif sline[0] == 'likelihood':
            if len(sline) != 2:
                print('Wrong number of arguments for "likelihood".')
                error_ct += 1
            else:
                out['likelihoods'].append(sline[1])
        # Deal with deriv
        elif sline[0] == 'deriv':
            fline = flines[ix].split(None, 2)
            if len(fline) != 3:
                print('Wrong number of arguments for "deriv".')
                error_ct += 1
            elif is_number(fline[1]) | is_number(fline[2]):
                print('Wrong argument type for "deriv": \n> %s' % sline)
            else:
                out['derivs'].append([fline[1], fline[2]])
                drv_names.append(fline[1])
        # Deal with var/var_class
        elif (sline[0] == 'var_class') | (sline[0] == 'var'):
            good1 = len(sline) == 6
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                print('Wrong "var/var_class" format:\n%s' % '  '.join(sline))
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
                print('Wrong "gauss_prior" format:\n%s' % '  '.join(sline))
                error_ct += 1
            else:
                out['gauss_priors'].append([sline[1]] + [float(x) for x in sline[2:]])
        # Deal with fix_class
        elif sline[0] == 'fix_class':
            if sline[1] == 'output':
                out['base_par_class']['output'] = ' '.join(sline[2:])
                bpc_names.append(sline[1])
            elif len(sline) != 3:
                print('Wrong "fix_class" format:\n%s' % '  '.join(sline))
                error_ct += 1
            elif sline[1] == 'non_linear':
                out['base_par_class']['non linear'] = sline[2]
                bpc_names.append(sline[1])
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
                print('Wrong "fix" format:\n%s' % '  '.join(sline))
                error_ct += 1
            elif is_number(sline[1]) | (not is_number(sline[2])):
                print('Wrong "fix" format:\n%s' % '  '.join(sline))
                error_ct += 1
            else:
                out['base_par_likes'][sline[1]] = float(sline[2])
                bpl_names.append(sline[1])
        # Warn about unknown options
        else:
            known_other_options = [
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
                print('Unrecognized option "%s". Ignored.' % sline[0])

    ### Check for duplicate parameters
    for n in vp_names:
        if vp_names.count(n) > 1:
            print('Duplicate "var" parameter: %s' % n)
            error_ct +=1
        if (n in bpc_names) or (n in bpl_names):
            print('Error: parameter "%s" is both fixed and varying.' % n)
            error_ct +=1
    for n in bpc_names:
        if bpc_names.count(n) > 1:
            print('Duplicate "fix_class" parameter: %s' % n)
            error_ct +=1
    for n in bpl_names:
        if bpl_names.count(n) > 1:
            print('Duplicate "fix" parameter: %s' % n)
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
                print('Error: element %s of parameter array "%s" missing.' % (i, n))
                error_ct +=1
        out['array_var'][n] = max(ixs)+1

    ### Checks constraints
    for cst_name in cst_names:
        if cst_name in (vp_names + bpc_names + bpl_names):
            print('Error: %s is both constrained and variable/fixed.' % cst_name)
            error_ct +=1

    ### Checks for derived parameters
    for der in out['derivs']:
        if  ('mPk' not in out['base_par_class']['output']) & ('sigma8' in der):
            print('sigma_8 asked as a derived parameter, but mPk not in output.')
            error_ct +=1

    ### Checks for Gaussian priors
    for i, p in enumerate(out['gauss_priors']):
        if p[0] not in (vp_names + drv_names):
            print('Error: parameter %s has prior but is not an MCMC or derived parameter.' % p[0])
            error_ct +=1
        if p[0] in drv_names:
            out['drv_gauss_priors'].append(out['gauss_priors'].pop(i))

    ### Raise error if any problem detected, else return final dictionary
    if ignore_errors:
        print('%s problem detected.' % error_ct)
        return out
    elif error_ct == 1:
        raise ValueError('Check your .ini file (1 problem detected).')
    elif error_ct > 1:
        raise ValueError('Check your .ini file (%s problems detected)' % error_ct)
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
