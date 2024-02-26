import os
import datetime
import subprocess
import numpy as np
from os.path import isfile
from collections.abc import Iterable

# Returns boolean according to whether input is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Returns boolean according to whether input string finished with "(N)" where N
# is an integer, and if yes returns also the base string + the said integer
def is_arrval(s):
    if (s.count("(") != 1) or (s.count(")") != 1):
        return [False, None, None]
    if not s.endswith(")"):
        return [False, None, None]
    i1, i2 = s.index("("), s.index(")")
    if i1 > i2:
        return [False, None, None]
    maybe_int = s[i1+1:i2]
    if not maybe_int.isnumeric():
        return [False, None, None]
    return [True, s[:i1], int(maybe_int)]

# Special string splitting function, which ignores the presence of
# the separator character when it is between two double quotes
def splt(s, sep=None, maxsplit=-1):
    ctp = s.count('"')
    if ctp == 0:
        return s.split(sep, maxsplit)
    if ctp % 2 == 1:
        return None
    final_splt = []
    tmp_str = ''
    in_quote = False
    for n in s:
        if (n == '"') & (not in_quote):
            if tmp_str.strip() != '':
                final_splt += tmp_str.strip().split()
            tmp_str = ''
            in_quote = True
        elif (n == '"') & in_quote:
            final_splt.append(tmp_str.strip())
            tmp_str = ''
            in_quote = False
        else:
            tmp_str += n
    if tmp_str.strip() != '':
        final_splt += tmp_str.strip().split()
    return final_splt

class counter():
    def __init__(self):
        self.count = 0
    def return_count(self):
        return self.count
    def increase_count(self):
        self.count += 1
        return

def parse_ini_file(fname, silent_mode=False):

    ### Strings containing all warnings and errors, to be printed at the end
    str_warn = ''
    str_err = ''  # should ideally stay empty

    ### Open the .ini file
    with open(fname) as f:
        lines = f.readlines()
    out =  {}

    ### Read all lines and options (i.e. 1st word on each line)
    slines = [] # split lines
    flines = [] # non-split lines
    options = []
    for line in lines:
        sline = splt(line.replace('\n',''))
        if sline == None:
            str_err += f'Error: odd number of double quotes\n> {line}\n'
        empty_line = sline == []
        if not empty_line and sline is not None:
            is_comment = not line.strip()[0].isalpha()
            if not is_comment:
                slines.append(sline)
                flines.append(line.replace('\n','').strip())
                options.append(sline[0])

    ### Deal with debug_mode
    ct = options.count('debug_mode')
    out['debug_mode'] = False
    if ct == 0:
        str_warn += '"debug_mode" not found. Assumed "no".\n'
    elif ct > 1:
        str_err += f'Multiple ({ct}) instances of "debug_mode" found.\n'
    else:
        ix = options.index('debug_mode')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "debug_mode".\n'
        elif slines[ix][1] not in ['yes', 'no']:
            str_err += '"debug_mode" should be "yes" or "no".\n'
        elif slines[ix][1] == 'yes':
            out['debug_mode'] = True

    ### Deal with output_root
    ct = options.count('output_root')
    if ct == 0:
        str_err += '"output_root" not found.\n'
        out['output_root'] = None
    elif ct > 1:
        str_err += f'Multiple ({ct}) instances of "output_root" found.\n'
        out['output_root'] = None
    else:
        ix = options.index('output_root')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "output_root".\n'
            out['output_root'] = None
        elif is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "output_root".\n'
            out['output_root'] = None
        else:
            out['output_root'] = slines[ix][1]

    ### Deal with input_fname
    ct = options.count('input_fname')
    if ct == 0:
        str_warn += '"input_fname" not found, starting with random positions.\n'
        out['input_fname'] = None
    elif ct > 1:
        str_err += f'Multiple ({ct}) instances of "input_fname" found.\n'
        out['input_fname'] = None
    else:
        ix = options.index('input_fname')
        l = len(slines[ix])
        if (l != 2) & (l != 3):
            str_err += 'Wrong number of arguments for "input_fname".\n'
            out['input_fname'] = None
        else:
            test1 = is_number(slines[ix][1])
            test2 = False
            if l == 3:
                test2 = not is_number(slines[ix][2])
            if test1 or test2:
                str_err += 'Wrong argument(s) type(s) for "input_fname".\n'
                out['input_fname'] = None
            else:
                out['input_fname'] = slines[ix][1]
                if not isfile(out['input_fname']):
                    str_err += ('File chosen as "input_fname" (namely '
                                f'{out["input_fname"]}) does not exist.\n')
                    out['input_fname'] = None
                else:
                    out['ch_start'] = int(slines[ix][2]) if l == 3 else -1
                    str_warn += (f"Will use step {out['ch_start']} from "
                                 f"previous chain {out['input_fname']} as "
                                 "starting point.\n")

    ### Deal with continue_chain
    ct = options.count('continue_chain')
    out['continue_chain'] = False
    if ct == 0:
        str_warn += '"continue_chain" not found. Assumed "no".\n'
    elif ct > 1:
        str_err += f'Multiple ({ct}) instances of "continue_chain" found.\n'
    else:
        ix = options.index('continue_chain')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "continue_chain".\n'
        elif slines[ix][1] not in ['yes', 'no']:
            str_err += 'Argument for "continue_chain" should be yes or no.\n'
        else:
            outname = out['output_root'] + '.txt'
            if slines[ix][1] == 'yes':
                if not isfile(outname):
                    str_err += ("The chain you want to continue from "
                                f"({outname}) does not exist.\n")
                else:
                    out['continue_chain'] = True
    if not out['continue_chain'] and out['output_root'] is not None:
        outname = out['output_root'] + '.txt'
        if isfile(outname):
            str_err += (f'The output chain ({outname}) already exists. Set '
                        '"continue_chain" to yes if you want to append new '
                        'samples to it.\n')

    ### Deal with choice of MCMC sampler
    ct = options.count('which_sampler')
    if ct == 0:
        str_warn += '"which_sampler" not found, assuming emcee.\n'
        out['which_sampler'] = 'emcee'
    elif ct > 2:
        str_err += f'Multiple ({ct}) instances of "which_sampler" found.\n'
    else:
        ix = options.index('which_sampler')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "which_sampler".\n'
        elif slines[ix][1] not in ['emcee', 'zeus']:
            str_err += 'Unrecognized sampler: should be either emcee or zeus.\n'
        else:
            out['which_sampler'] = slines[ix][1]

    ### Deal with parallel
    if 'parallel' not in options:
        str_warn += '"parallel" not found, assuming "none".\n'
        out['parallel'] = ['none']
    else:
        ct = options.count('parallel')
        if ct > 1:
            str_err += f'Multiple ({ct}) instances of "parallel" found.\n'
        else:
            ix = options.index('parallel')
            if slines[ix][1] not in ['none', 'multiprocessing', 'MPI']:
                str_err += 'Unrecognized argument for "parallel".\n'
            elif slines[ix][1] == 'multiprocessing':
                if len(slines[ix]) != 3:
                    str_err += 'Wrong number of arguments for "parallel".\n'
                elif not is_number(slines[ix][2]):
                    str_err += 'Wrong argument type in "parallel".\n'
                else:
                    out['parallel'] = slines[ix][1:]
            else:
                if len(slines[ix]) != 2:
                    str_err += 'Wrong number of arguments for "parallel".\n'
                else:
                    out['parallel'] = [slines[ix][1]]

    ### Deal with n_walkers
    ct = options.count('n_walkers')
    if ct == 0:
        str_warn += ('"n_walkers" not found, assuming 2 times the number of '
                     'parameters.\n')
        out['n_walkers_type'] = 'prop_to'
        out['n_walkers'] = 2
    elif ct > 1:
        str_err += f'Multiple ({ct}) instances of "n_walkers" found.\n'
    else:
        ix = options.index('n_walkers')
        if len(slines[ix]) != 3:
            str_err += 'Wrong number of arguments for "n_walkers".\n'
        elif slines[ix][1] not in ['custom', 'prop_to']:
            str_err += 'Unrecognizd argument for "n_walkers".\n'
        elif not is_number(slines[ix][2]):
            str_err += 'Wrong argument type for "n_walkers".\n'
        else:
            out['n_walkers_type'] = slines[ix][1]
            out['n_walkers'] = int(slines[ix][2])

    ### Deal with n_steps
    ct = options.count('n_steps')
    if ct == 0:
        str_err += '"n_steps" not found.\n'
    elif ct > 2:
        str_err += f'Multiple ({ct}) instances of "n_steps" found.\n'
    else:
        ix = options.index('n_steps')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "n_steps".\n'
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "n_steps".\n'
        else:
            out['n_steps'] = int(slines[ix][1])

    ### Deal with thin_by
    ct = options.count('thin_by')
    if ct == 0:
        str_warn += '"thin_by" not found, assuming no thinning.\n'
        out['thin_by'] = 1
    elif ct > 2:
        str_err += f'Multiple ({ct}) instances of "thin_by" found.\n'
    else:
        ix = options.index('thin_by')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "thin_by".\n'
        elif not is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "thin_by".\n'
        else:
            out['thin_by'] = int(slines[ix][1])

    ### Deal with temperature
    ct = options.count('temperature')
    if ct == 0:
        str_warn += '"temperature" not found, assuming fixed to 1.\n'
        out['temperature'] = [[1., out['n_steps']]]
        out['temperature_is_notinv'] = True
    elif ct > 2:
        str_err += f'Multiple ({ct}) instances of "temperature" found.\n'
    else:
        ix = options.index('temperature')
        if len(slines[ix]) < 3:
            str_err += 'Wrong number of arguments for "temperature".\n'
        elif slines[ix][1] not in ['inv', 'notinv']:
            str_err += ('Second argument of "temperature" should be "inv" or '
                        '"notinv"\n')
        elif (len(slines[ix]) == 3) and is_number(slines[ix][2]):
            out['temperature'] = [[float(slines[ix][2]), out['n_steps']]]
            out['temperature_is_notinv'] = slines[ix][1] == 'notinv'
        else:
            tmp_str = (flines[ix].lstrip("temperature").lstrip()
                       .lstrip("not").lstrip("inv").strip())
            fail = False
            try:
                exec(f"global tmparr; tmparr = {tmp_str}")
                if not isinstance(tmparr, Iterable):
                    fail = True
                elif any([not isinstance(arr, Iterable) for arr in tmparr]):
                    fail = True
                elif any([len(arr) != 2 for arr in tmparr]):
                    fail = True
                elif any([not is_number(arr[0]) for arr in tmparr]):
                    fail = True
                elif any([arr[0]<0 for arr in tmparr]):
                    fail = True
                elif any([type(arr[1]) != int for arr in tmparr]):
                    fail = True
                elif any([arr[1]<=0 for arr in tmparr]):
                    fail = True
                else:
                    n_temp_steps = 0
                    for arr in tmparr:
                        n_temp_steps += arr[1]
            except:
                fail = True
            if fail:
                out['temperature'] = None
                str_err += (f'Wrong syntax for "temperature": {tmp_str} is not '
                             'a list of [float, int] lists.\n')
            else:
                if n_temp_steps != out['n_steps']:
                    str_err += (f'Wrong syntax for "temperature": the total '
                                'number of temperature values provided '
                                f'({n_temp_steps}) should be the same as the '
                                f'number of MCMC steps ({out["n_steps"]}).\n')
                else:
                    out['temperature'] = tmparr
                    out['temperature_is_notinv'] = slines[ix][1] == 'notinv'

    ### Deal with which_class
    ct = options.count('which_class')
    if ct == 0:
        str_warn += '"which_class" not found, assuming regular class.\n'
        out['which_class'] = 'classy'
    elif ct > 2:
        str_err += f'Multiple ({ct}) instances of "which_class" found.\n'
    else:
        ix = options.index('which_class')
        if len(slines[ix]) != 2:
            str_err += 'Wrong number of arguments for "which_class".\n'
        elif is_number(slines[ix][1]):
            str_err += 'Wrong argument type for "which_class".\n'
        else:
            out['which_class'] = slines[ix][1]

    ### Check if any likelihood
    ct = options.count('likelihood')
    if ct == 0:
        str_err += 'No likelihood specified.\n'

    ### Check if any free variables
    ct = options.count('var') + options.count('var_class')
    if ct == 0:
        str_err += 'No free parameters specified.\n'

    ### Loop to catch parameters that can have multiple instances
    out['keep_input'] = []
    out['sampler_kwargs'] = {}
    out['constraints'] = []
    out['likelihoods'] = []
    out['derivs'] = []
    out['gauss_priors'] = []
    out['drv_gauss_priors'] = []
    out['drv_uni_priors'] = []
    out['var_par'] = []
    out['array_var'] = {}
    out['base_par_class'] = {}
    out['base_par_lkl'] = {}
    out['profiles'] = []
    vp_names  = [] # all varying parameter names
    bpc_names = [] # non-varying class parameters names
    bpl_names = [] # non-varying other parameters names
    arr_names = [] # array parameter names
    cst_names = [] # constrained parameter names
    drv_names = [] # derived parameter names
    for sline, fline in zip(slines, flines):
        ### Deal with keep_input
        if sline[0] == 'keep_input':
            if len(sline) < 2:
                str_err += 'Wrong number of arguments for "keep_input":\n'
                str_err += f'> {fline}\n'
                continue
            ct_gt = fline.count(">")
            ct_lt = fline.count("<")
            if (ct_gt + ct_lt) != 1:
                str_err += 'Wrong syntax for "keep_input":\n'
                str_err += f'> {fline}\n'
                continue
            sign = ">" if ct_gt == 1 else "<"
            tmp = fline.lstrip('keep_input').split(sign)
            if len(tmp) != 2:
                str_err += 'Wrong syntax for "keep_input":\n'
                str_err += f'> {fline}\n'
                continue
            if not is_number(tmp[1]):
                str_err += 'Wrong syntax for "keep_input":\n'
                str_err += f'> {fline}\n'
                continue
            out['keep_input'].append([tmp[0].strip(), sign, tmp[1].strip()])
        # Deal with options of MCMC sampler
        elif sline[0] == 'sampler_kwarg':
            if len(sline) < 3:
                str_err += 'Wrong number of arguments for "sampler_kwarg":\n'
                str_err += f'> {fline}\n'
            else:
                if is_number(sline[1]):
                    str_err += 'Wrong argument type for "sampler_kwarg":\n'
                    str_err += f'> {fline}\n'
                else:
                    arg = fline.lstrip('sampler_kwarg').lstrip(sline[1])
                    out['sampler_kwargs'][sline[1]] = arg
        # Deal with constraints
        elif sline[0] == 'constraint':
            if fline.count("=") != 1:
                str_err += 'More/less than 1 equal sign in "constraint":\n'
                str_err += f'> {fline}\n'
            elif len(fline.split("=")[0].split()) != 2:
                str_err += 'Wrong syntax for "constraint":\n'
                str_err += f'> {fline}\n'
            elif fline.split("=")[1].strip() == '':
                str_err += 'Wrong syntax for "constraint":\n'
                str_err += f'> {fline}\n'
            else:
                tmp_cst = fline.replace('constraint','').strip()
                out['constraints'].append(tmp_cst)
                tmp_cst_name = tmp_cst.split("=")[0].strip()
                i1 = tmp_cst_name.index("[")
                i2 = tmp_cst_name.index("]")
                cst_names.append(tmp_cst_name[i1+2:i2-1])
                test_arr = is_arrval(tmp_cst_name[i1+2:i2-1])
                if test_arr[0]:
                    arr_names.append(tmp_cst_name[i1+2:i2-1])
                    if test_arr[1] not in out['array_var'].keys():
                        out['array_var'][test_arr[1]] = 0
        # Deal with likelihood
        elif sline[0] == 'likelihood':
            if len(sline) != 2:
                str_err += 'Wrong number of arguments for "likelihood".\n'
                str_err += f'> {fline}\n'
            else:
                out['likelihoods'].append(sline[1])
        # Deal with deriv
        elif sline[0] == 'deriv':
            if len(sline) != 3:
                str_err += 'Wrong number of arguments for "deriv":\n'
                str_err += f'> {fline}\n'
            elif is_number(sline[1]) | is_number(sline[2]):
                str_err += 'Wrong argument type for "deriv":\n'
                str_err += f'> {fline}\n'
            else:
                out['derivs'].append([sline[1], sline[2]])
                drv_names.append(sline[1])
        # Deal with var/var_class
        elif (sline[0] == 'var_class') | (sline[0] == 'var'):
            good1 = len(sline) == 6
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                str_err += 'Wrong "var/var_class" format:\n'
                str_err += f'> {fline}\n'
            else:
                out['var_par'].append(sline[:2] + [float(x) for x in sline[2:]])
                vp_names.append(sline[1])
                test_arr = is_arrval(sline[1])
                if test_arr[0]:
                    arr_names.append(sline[1])
                    if test_arr[1] not in out['array_var'].keys():
                        out['array_var'][test_arr[1]] = 0
        # Deal with gauss_prior
        elif sline[0] == 'gauss_prior':
            good1 = len(sline) == 4
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                str_err += 'Wrong "gauss_prior" format:\n'
                str_err += f'> {fline}\n'
            else:
                out['gauss_priors'].append(
                    [sline[1]] + [float(x) for x in sline[2:]])
        # Deal with uni_prior
        elif sline[0] == 'uni_prior':
            good1 = len(sline) == 4
            good2 = not is_number(sline[1])
            good3 = all([is_number(x) for x in sline[2:]])
            if not (good1 & good2 & good3):
                str_err += 'Wrong "uni_prior" format:\n'
                str_err += f'> {fline}\n'
            else:
                out['drv_uni_priors'].append(
                    [sline[1]] + [float(x) for x in sline[2:]])
        # Deal with fix_class
        elif sline[0] == 'fix_class':
            if (len(sline) != 3) or is_number(sline[1]):
                str_err += 'Wrong "fix_class" format:\n'
                str_err += f'> {fline}\n'
            elif is_number(sline[2]):
                out['base_par_class'][sline[1]] = float(sline[2])
                bpc_names.append(sline[1])
                test_arr = is_arrval(sline[1])
                if test_arr[0]:
                    arr_names.append(sline[1])
                    if test_arr[1] not in out['array_var'].keys():
                        out['array_var'][test_arr[1]] = 0
            else:
                out['base_par_class'][sline[1]] = sline[2]
                bpc_names.append(sline[1])
        # Deal with fix
        elif sline[0] == 'fix':
            if len(sline) != 3:
                str_err += 'Wrong "fix" format:\n'
                str_err += f'> {fline}\n'
            elif is_number(sline[1]):
                str_err += 'Wrong "fix" format:\n'
                str_err += f'> {fline}\n'
            else:
                if is_number(sline[2]):
                    out['base_par_lkl'][sline[1]] = float(sline[2])
                else:
                    out['base_par_lkl'][sline[1]] = sline[2]
                bpl_names.append(sline[1])
        elif sline[0] == 'profile':
            if len(sline) < 3:
                str_err += 'Wrong "profile" format:\n'
                str_err += f'> {fline}\n'
            elif sline[1] not in vp_names:
                str_err += ('Your requested profile parameter is not a varying '
                            'MCMC parameter\n')
                str_err += f'> {fline}\n'
            else:
                tmp_str = fline.lstrip("profile").lstrip().lstrip(sline[1]).strip()
                fail = False
                try:
                    exec(f"global tmparr; tmparr = {tmp_str}")
                    if not isinstance(tmparr, Iterable):
                        fail = True
                    elif any([not is_number(arr) for arr in tmparr]):
                        fail = True
                    else:
                        n_temp_steps = len(tmparr)
                except:
                    fail = True
                if fail:
                    str_err += (f'Wrong syntax for "profile": {tmp_str} is not '
                                'a list of floats.\n')
                else:
                    out['profiles'].append([sline[1], tmparr])
        # Warn about unknown options
        else:
            known_other_options = [
                'continue_chain',
                'debug_mode',
                'input_fname',
                'n_steps',
                'n_walkers',
                'output_root',
                'parallel',
                'temperature',
                'thin_by',
                'which_class',
                'which_sampler',
            ]
            if sline[0] not in known_other_options:
                str_warn += f'Unrecognized option "{sline[0]}". Ignored.\n'

    ### Adjust number of walkers if "proportional" option is requested
    if out['n_walkers_type'] == 'prop_to':
        out['n_walkers'] *= len(out['var_par'])

    ### Check profiled parameters (if requested)
    if len(out['profiles']) > 0:
        for p in out['profiles']:
            if len(p[1]) != out['n_walkers']:
                str_err += (f'Wrong syntax for "profile" for {p[0]}: the total '
                            f'number of provided profiled values ({len(p[1])}) '
                            'is not equal to the number of walkers '
                            f'({out["n_walkers"]}).\n')
        if out['which_sampler'] != 'emcee':
            str_err += ('Profiles are only possible with the emcee sampler '
                        f'(you requested {out["which_sampler"]}).\n')
        else:
            import emcee
            if not hasattr(emcee.moves, "StretchMove_except_some"):
                str_err += ('Profiles are only possible with a custom version '
                            'of the emcee sampler which you can get at '
                            'https://github.com/s-ilic/emcee.\n')
        if "moves" in out['sampler_kwargs']:
            str_warn += ('The moves options for emcee will be overrided for '
                         'a special move to allow for profiling parameters.\n')

    ### Check for duplicate parameters
    for n in vp_names:
        if vp_names.count(n) > 1:
            str_err += f'Duplicate "var_class/var" parameter: {n}\n'
        if (n in bpc_names) or (n in bpl_names):
            str_err += f'Error: parameter "{n}" is both fixed and varying.\n'
    for n in bpc_names:
        if bpc_names.count(n) > 1:
            str_err += f'Duplicate "fix_class" parameter: {n}\n'
    for n in bpl_names:
        if bpl_names.count(n) > 1:
            str_err += f'Duplicate "fix" parameter: {n}\n'

    ### Checks for parameter arrays
    for n in out['array_var'].keys():
        ixs = []
        for av in arr_names:
            tmp = is_arrval(av)
            if tmp[1] == n:
                ixs.append(tmp[2])
        for i in range(max(ixs)+1):
            not_good1 = f'{n}({i})' not in arr_names
            not_good2 = f'{n}({i})' not in cst_names
            if not_good1 & not_good2:
                str_err += (f'Error: element {i} of parameter array "{n}" '
                            'missing.\n')
        out['array_var'][n] = max(ixs) + 1

    ### Checks constraints
    for cst_name in cst_names:
        if cst_name in (vp_names + bpc_names + bpl_names):
            str_err += (f'Error: {cst_name} is both constrained and '
                        'variable/fixed.\n')

    ### Checks for uniform priors
    for i, p in enumerate(out['drv_uni_priors']):
        if p[0] not in (vp_names + drv_names):
            str_err += (f'Error: parameter {p[0]} has prior but is not an MCMC '
                        'or derived parameter.\n')

    ### Checks for Gaussian priors
    for i, p in enumerate(out['gauss_priors']):
        if p[0] not in (vp_names + drv_names):
            str_err += (f'Error: parameter {p[0]} has prior but is not an MCMC '
                        'or derived parameter.\n')
        if p[0] in drv_names:
            out['drv_gauss_priors'].append(out['gauss_priors'].pop(i))

    ### Raise error if any problem detected, else return final dictionary
    has_errors = str_err != ''
    if not has_errors:
        str_err += 'None.\n'
    if str_warn == '':
        str_warn += 'None.\n'
    if not silent_mode:
        print("\n"
              "################\n"
              "### WARNINGS ###\n"
              "################\n"
              + str_warn)
        print("################\n"
              "#### ERRORS ####\n"
              "################\n"
              + str_err)
    if silent_mode:
        return out
    elif has_errors:
        raise ValueError('Check your .ini file for the above error(s).')
    else:
        return out


def copy_ini_file(fname, params):

    ### Write temporay copy of ini file
    with open(fname) as f:
        lines = f.readlines()
    with open(params['output_root'] + '.ini.tmp', 'w') as f:
        for line in lines:
            sline = splt(line.replace('\n',''))
            empty_line = sline == []
            if not empty_line and sline is not None:
                is_comment = not line.strip()[0].isalpha()
                if not is_comment:
                    f.write(line)

    ### Write inside log file
    with open(params['output_root'] + '.log', 'a') as f:
        f.write("******"+str(datetime.datetime.now())+"******\n")
        if not params["continue_chain"]:
            f.write("Chain is started.\n\n")
        else:
            sh = np.loadtxt(params['output_root'] + '.txt').shape
            f.write("Chain is restarted.\nIt currently has "
                    f"{sh[0]//params['n_walkers']} samples.\n")
            f.write("Changes in ini file:\n")
            res = subprocess.run(
                ["diff",
                params['output_root'] + '.ini',
                params['output_root'] + '.ini.tmp'],
                stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
            if res == "":
                f.write("None.\n")
            else:
                f.write(res)
            f.write("\n")

    ### Copy to final file
    os.system(f"mv {params['output_root']}.ini.tmp {params['output_root']}.ini")

    return None
